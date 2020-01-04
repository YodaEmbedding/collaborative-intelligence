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

        // TODO watch out for encoding time...
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

        val sample = stats.sample

        val enoughTime = Duration.ofMillis(0)
        if (networkManager.timeUntilWriteAvailable > enoughTime) {
            Log.i(TAG, "Dropped frame because of slow upload speed!")
            return false
        }

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
