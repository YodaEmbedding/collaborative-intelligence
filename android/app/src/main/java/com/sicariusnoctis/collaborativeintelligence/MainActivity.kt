package com.sicariusnoctis.collaborativeintelligence

import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.renderscript.RenderScript
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.sicariusnoctis.collaborativeintelligence.ui.ModelUiController
import com.sicariusnoctis.collaborativeintelligence.ui.OptionsUiController
import com.sicariusnoctis.collaborativeintelligence.ui.StatisticsUiController
import io.fotoapparat.parameter.Resolution
import io.fotoapparat.preview.Frame
import io.reactivex.Completable
import io.reactivex.Flowable
import io.reactivex.disposables.CompositeDisposable
import io.reactivex.internal.schedulers.IoScheduler
import io.reactivex.processors.PublishProcessor
import io.reactivex.rxkotlin.subscribeBy
import io.reactivex.rxkotlin.zipWith
import kotlinx.android.synthetic.main.activity_main.*
import java.time.Duration
import java.time.Instant
import java.util.concurrent.CompletableFuture

class MainActivity : AppCompatActivity() {
    private val TAG = MainActivity::class.qualifiedName

    private lateinit var camera: Camera
    private lateinit var clientProcessor: ClientProcessor
    private lateinit var networkManager: NetworkManager
    private lateinit var previewResolution: Resolution
    private lateinit var rs: RenderScript
    private lateinit var modelUiController: ModelUiController
    private lateinit var optionsUiController: OptionsUiController
    private lateinit var statisticsUiController: StatisticsUiController

    private var prevFrameTime: Instant? = null
    private val frameProcessor: PublishProcessor<Frame> = PublishProcessor.create()
    private val statistics = Statistics()
    private val subscriptions = CompositeDisposable()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        rs = RenderScript.create(this)
        camera = Camera(this, cameraView) { frame -> this.frameProcessor.onNext(frame) }
        modelUiController = ModelUiController(
            modelSpinner,
            layerSeekBar,
            compressionSpinner
        )
        optionsUiController = OptionsUiController(
            uploadRateLimitSeekBar
        )
        statisticsUiController = StatisticsUiController(
            statistics,
            predictionsText,
            fpsText,
            uploadText,
            uploadAvgText,
            preprocessText,
            clientInferenceText,
            encodingText,
            networkReadText,
            serverInferenceText,
            networkWriteText,
            totalText,
            totalAvgText,
            framesProcessedText,
            lineChart
        )
    }

    override fun onStart() {
        super.onStart()

        clientProcessor = ClientProcessor(rs)
        networkManager = NetworkManager(statistics, statisticsUiController)

        val cameraReady = CompletableFuture<Unit>()
        camera.start()
        camera.getCurrentParameters().whenAvailable {
            this.previewResolution = it!!.previewResolution
            cameraReady.complete(Unit)
        }

        Completable
            .mergeArray(
                Completable.fromFuture(cameraReady),
                clientProcessor.initInference(),
                networkManager.connectNetworkAdapter()
            )
            .subscribeOn(IoScheduler())
            .andThen(initPostprocessor())
            .andThen(networkManager.subscribeNetworkIo())
            .andThen(networkManager.subscribePingGenerator())
            .andThen(switchModel(modelUiController.modelConfig))
            .andThen(subscribeFrameProcessor())
            .subscribeBy({ it.printStackTrace() })
    }

    override fun onStop() {
        super.onStop()
        camera.stop()
        clientProcessor.close()
        networkManager.close()
        networkManager.dispose()
    }

    override fun onDestroy() {
        subscriptions.dispose()
        super.onDestroy()
    }

    private fun initPostprocessor() = Completable.fromRunnable {
        val rotationCompensation = CameraPreviewPostprocessor.getRotationCompensation(
            this, CameraCharacteristics.LENS_FACING_BACK
        )

        clientProcessor.initPostprocessor(
            previewResolution.width,
            previewResolution.height,
            rotationCompensation
        )
    }

    // TODO this is kind of using references to things that stop existing after onStop
    private fun subscribeFrameProcessor() = Completable.fromRunnable {
        // clientProcessor.subscribeFrameProcessor(frameProcessor)

        val frameProcessorSubscription = frameProcessor
            .subscribeOn(IoScheduler())
            .onBackpressureDrop()
            .observeOn(IoScheduler(), false, 1)
            .map { it to modelUiController.modelConfig }
            .onBackpressureLimitRate(
                onDrop = { statistics[it.second].frameDropped() },
                limit = { shouldProcessFrame(it.second) }
            )
            .zipWith(Flowable.range(0, Int.MAX_VALUE)) { (frame, modelConfig), frameNumber ->
                prevFrameTime = Instant.now()
                statistics[modelConfig].makeSample(frameNumber)
                FrameRequest(frame, FrameRequestInfo(frameNumber, modelConfig))
            }
            .doOnNext { Log.i(TAG, "Started processing frame ${it.info.frameNumber}") }
            .mapTimed(ModelStatistics::setPreprocess) { it.map(clientProcessor.postprocessor::process) }
            .observeOn(clientProcessor.inferenceScheduler, false, 1)
            .mapTimed(ModelStatistics::setInference) { clientProcessor.inference!!.run(it) }
            .observeOn(networkManager.networkWriteScheduler, false, 1)
            .doOnNext { networkManager.uploadLimitRate = optionsUiController.uploadRateLimit }
            .doOnNextTimed(ModelStatistics::setNetworkWrite) {
                networkManager.writeFrameRequest(it)
            }
            .doOnNext {
                statistics[it.info.modelConfig].setUpload(it.info.frameNumber, it.obj.size)
            }
            .subscribeBy({ it.printStackTrace() })

        subscriptions.add(frameProcessorSubscription)
    }

    private fun switchModel(modelConfig: ModelConfig) = Completable
        .fromRunnable { Log.i(TAG, "Switching model begin") }
        .andThen(
            Completable.mergeArray(
                clientProcessor.switchModelInference(modelConfig),
                networkManager.switchModelServer(modelConfig)
            )
        )
        .doOnComplete { Log.i(TAG, "Switching model end") }

    private fun shouldProcessFrame(modelConfig: ModelConfig): Boolean {
        val stats = statistics[modelConfig]

        // Load new model if config changed
        if (modelConfig != clientProcessor.modelConfig) {
            val prevStats = statistics[clientProcessor.modelConfig]

            // Wait till latest sample has been processed client-side first
            if (prevStats.currentSample != null && prevStats.currentSample!!.networkWriteEnd == null) {
                Log.i(TAG, "Dropped frame because waiting to switch models")
                return false
            }

            switchModel(modelConfig).blockingAwait()
            statistics[modelConfig] = ModelStatistics()
            Log.i(TAG, "Dropped frame after switching model")
            return false
        }

        if (!stats.isFirstExist)
            return true

        // If first server response is not yet received, drop frame
        if (stats.isEmpty) {
            Log.i(TAG, "Dropped frame because waiting for first server response")
            return false
        }

        if (stats.currentSample?.inferenceEnd == null) {
            Log.i(TAG, "Dropped frame because frame is currently being processed")
            return false
        }

        if (stats.currentSample?.networkWriteEnd == null) {
            Log.i(TAG, "Dropped frame because frame is currently being written over network")
            return false
        }

        if (stats.currentSample?.networkReadEnd == null) {
            // Log.i(TAG, "Dropped frame because frame is currently being processed by server")
            return false
        }

        // TODO imagine we had the current upload bytes or bandwidth... then what?
        // maybe use the total upload bytes to subtract from what has currently been sent?
        // that tells us how much remains in the buffer...
        // if it's greater than an average frame size, we probably shouldn't send more...

        // val remainingBytes = totalSent - currSent
        // vs uploadSpeed???? that should roughly predict how much time needed before remaining frame is completely sent... and if that time is "negative", speed up!
        // remainingBytes

        // but how do we ramp up to "max" capacity? actually, we don't really because we decide using the above candidates whether or not to drop frame...
        // do we even need to phrase it in terms of "limits", actually?
        // sort of... because clientInference might take longer than network sometimes.

        // OK, but how to ramp up? Maybe leave a little gap? Or perhaps pre-measure network upload speed (and have user "reset" automatically?) sure

        val sample = stats.sample

        // TODO watch out for encoding time...
        // val preWriteTime = Duration.between(sample.preprocessStart, sample.inferenceEnd)
        // val enoughTime = preWriteTime.multipliedBy(2).dividedBy(3) + Duration.ofMillis(20)
        val enoughTime = Duration.ofMillis(0)
        if (networkManager.timeUntilWriteAvailable > enoughTime) {
            Log.i(TAG, "Dropped frame because of slow upload speed!")
            return false
        }
        // val extrapolatedBytes = networkAdapter!!.uploadBytesPerSecond * t.toMillis() / 1000
        // val s = "${(networkAdapter!!.uploadBytesPerSecond / 1024).toInt()}KB/s, " +
        //         "remaining: ${networkAdapter!!.uploadRemainingBytes}B, " +
        //         "uploaded: ${networkAdapter!!.uploadStats.uploadedBytes}, " +
        //         "goal: ${networkAdapter!!.uploadStats.goalBytes}"
        //         Log.i(TAG, s)
        // if (networkAdapter!!.uploadRemainingBytes > extrapolatedBytes) {
        //     Log.i(TAG, "Dropped frame because of slow upload speed!")
        //     return false
        // }

        // TODO app-test network bandwidth uploaded accuracy, max bandwidth, see if it ramps correctly, etc

        // TODO also, make sure it's app bandwidth (this can be fixed later, though! just use TrafficStats for now)

        // TODO this maintains reasonable FPS, but doesn't help much with latency (which can accumulate)
        // TODO rate limit upload in a smarter way... maybe subtract ping and check upload? idk
        // val fpsLimit = Duration.ofMillis((1000 / stats.fps / 1.3).toLong())
        // val fpsLimit = stats.sample.total.multipliedBy(7).dividedBy(10)
        // val networkLimit = minOf(sample.networkWrite, Duration.ofMillis(50))
        // TODO watch out for encoding time...

        val timeSinceLastFrameRequest = Duration.between(prevFrameTime, Instant.now())

        if (timeSinceLastFrameRequest < sample.clientInference) {
            Log.i(TAG, "Dropped frame because of slow client inference time!")
            return false
        }

        if (timeSinceLastFrameRequest < sample.serverInference) {
            Log.i(TAG, "Dropped frame because of slow server inference time!")
            return false
        }

        return true
    }

    // TODO nullify prevFrameTime onModelConfigChanged? (for smoother transition)
    // TODO immediately send onModelConfigChanged request to server, instead of waiting for response?
    // no, because server doesn't cancel inferences... yet
    private fun <T> Flowable<T>.onBackpressureLimitRate(
        onDrop: (T) -> Unit,
        limit: (T) -> Boolean
    ): Flowable<T> {
        return this
            // .onBackpressureDrop(onDrop)
            .filter {
                if (limit(it)) {
                    true
                } else {
                    onDrop(it)
                    false
                }
            }
    }

    private fun <T> Flowable<FrameRequest<T>>.doOnNextTimed(
        timeFunc: (ModelStatistics, Int, Instant, Instant) -> Unit,
        onNext: (FrameRequest<T>) -> Unit
    ): Flowable<FrameRequest<T>> {
        return this.doOnNext { x ->
            val (_, start, end) = timed { onNext(x) }
            timeFunc(statistics[x.info.modelConfig], x.info.frameNumber, start, end)
        }
    }

    private fun <R, T> Flowable<FrameRequest<T>>.mapTimed(
        timeFunc: (ModelStatistics, Int, Instant, Instant) -> Unit,
        mapper: (FrameRequest<T>) -> FrameRequest<R>
    ): Flowable<FrameRequest<R>> {
        return this.map { x ->
            val (result, start, end) = timed { mapper(x) }
            timeFunc(statistics[x.info.modelConfig], x.info.frameNumber, start, end)
            result
        }
    }
}

// TODO Switch to using networkWrite.toSerialized()
// TODO Late-bind variables in RxJava?
