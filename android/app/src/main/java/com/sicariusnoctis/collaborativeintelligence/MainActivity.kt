package com.sicariusnoctis.collaborativeintelligence

import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.renderscript.RenderScript
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import io.fotoapparat.parameter.Resolution
import io.fotoapparat.preview.Frame
import io.reactivex.Completable
import io.reactivex.Flowable
import io.reactivex.Observable
import io.reactivex.Scheduler
import io.reactivex.android.schedulers.AndroidSchedulers
import io.reactivex.disposables.CompositeDisposable
import io.reactivex.internal.schedulers.IoScheduler
import io.reactivex.observables.ConnectableObservable
import io.reactivex.processors.PublishProcessor
import io.reactivex.rxkotlin.subscribeBy
import io.reactivex.rxkotlin.zipWith
import io.reactivex.schedulers.Schedulers
import kotlinx.android.synthetic.main.activity_main.*
import java.time.Duration
import java.time.Instant
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private val TAG = MainActivity::class.qualifiedName

    private lateinit var camera: Camera
    private lateinit var inference: Inference
    private lateinit var inferenceExecutor: ExecutorService
    private lateinit var inferenceScheduler: Scheduler
    private lateinit var networkWriteExecutor: ExecutorService
    private lateinit var networkWriteScheduler: Scheduler
    private lateinit var postprocessor: CameraPreviewPostprocessor
    private lateinit var previewResolution: Resolution
    private lateinit var rs: RenderScript
    private lateinit var modelUiController: ModelUiController
    private lateinit var statisticsUiController: StatisticsUiController

    private var networkAdapter: NetworkAdapter? = null
    private var prevFrameTime: Instant? = null
    private val frameProcessor: PublishProcessor<Frame> = PublishProcessor.create()
    private val statistics = Statistics()
    private val subscriptions = CompositeDisposable()

    private var networkRead: Observable<Response>? = null
    private var networkWrite: PublishProcessor<Any>? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        rs = RenderScript.create(this)
        camera = Camera(this, cameraView) { frame -> this.frameProcessor.onNext(frame) }
        modelUiController = ModelUiController(this, modelSpinner, layerSeekBar, compressionSpinner)
        statisticsUiController = StatisticsUiController(
            statistics,
            predictionsText,
            fpsText,
            uploadText,
            preprocessText,
            clientInferenceText,
            encodingText,
            networkReadText,
            serverInferenceText,
            networkWriteText,
            totalText,
            framesProcessedText,
            lineChart
        )
    }

    override fun onStart() {
        super.onStart()

        inferenceExecutor = Executors.newSingleThreadExecutor()
        inferenceScheduler = Schedulers.from(inferenceExecutor)
        networkWriteExecutor = Executors.newSingleThreadExecutor()
        networkWriteScheduler = Schedulers.from(networkWriteExecutor)

        val cameraReady = CompletableFuture<Unit>()
        camera.start()
        camera.getCurrentParameters().whenAvailable {
            this.previewResolution = it!!.previewResolution
            cameraReady.complete(Unit)
        }

        networkAdapter = NetworkAdapter()

        Completable
            .mergeArray(
                Completable.fromFuture(cameraReady),
                initInference(),
                connectNetworkAdapter()
            )
            .subscribeOn(IoScheduler())
            .andThen(initPostprocessor())
            .andThen(subscribeNetworkIo())
            .subscribeBy({ it.printStackTrace() })
    }

    override fun onStop() {
        super.onStop()
        camera.stop()
        // TODO shouldn't this be part of "doFinally" for frameProcessor?
        inferenceExecutor.submit { inference.close() }
        // TODO release inferenceExecutor?
        networkAdapter?.close()
    }

    override fun onDestroy() {
        subscriptions.dispose()
        super.onDestroy()
    }

    private fun initInference() = Completable
        .fromRunnable { inference = Inference() }
        .subscribeOn(inferenceScheduler)

    private fun initPostprocessor() = Completable.fromRunnable {
        // TODO this kind of doesn't work nice when you onStop/onStart the app
        if (::postprocessor.isInitialized)
            return@fromRunnable

        val width = previewResolution.width
        val height = previewResolution.height
        val rotationCompensation = CameraPreviewPostprocessor.getRotationCompensation(
            this, CameraCharacteristics.LENS_FACING_BACK
        )

        postprocessor =
            CameraPreviewPostprocessor(rs, width, height, 224, 224, rotationCompensation)
    }

    private fun connectNetworkAdapter() = Completable
        .fromRunnable { networkAdapter!!.connect() }
        .subscribeOn(IoScheduler())

    // TODO Prevent duplicate subscriptions! (e.g. if onStart called multiple times); unsubscribe?
    // TODO .onTerminateDetach()
    private fun subscribeNetworkIo() = Completable.fromRunnable {
        // TODO don't really NEED a networkWrite observable... if we make a single executor thread
        networkWrite = PublishProcessor.create()
        val networkWriteRequests = networkWrite!!.subscribeOn(networkWriteScheduler)

        val networkWriteModelConfigSubscription = networkWriteRequests
            .filter { it is ModelConfig }
            .doOnNext { networkAdapter!!.writeModelConfig(it as ModelConfig) }
            .subscribeBy({ it.printStackTrace() })
        subscriptions.add(networkWriteModelConfigSubscription)

        val networkWriteFrameRequestSubscription = networkWriteRequests
            .filter { it is FrameRequest<*> && it.obj is ByteArray }
            .doOnNext { networkAdapter!!.writeFrameRequest(it as FrameRequest<ByteArray>) }
            .subscribeBy({ it.printStackTrace() })
        subscriptions.add(networkWriteFrameRequestSubscription)

        val networkWriteSampleSubscription = networkWriteRequests
            .filter { it is FrameRequest<*> && it.obj is Sample }
            .doOnNext { networkAdapter!!.writeSample(it as FrameRequest<Sample>) }
            .subscribeBy({ it.printStackTrace() })
        subscriptions.add(networkWriteSampleSubscription)

        networkRead = Observable.fromIterable(Iterable {
            iterator {
                // TODO Wait until thread is alive using CountDownLatch?
                // TODO thread.isAlive()? socket.isOpen? volatile boolean flag?
                while (true) {
                    val (result, start, end) = timed { networkAdapter!!.readResultResponse() as Response }  // TODO
                    if (result == null) break
                    if (result is ResultResponse) {
                        // TODO does this even make sense?
                        // TODO Also, have a "ping"/ack return when all data is received
                        val stats = statistics[result.frameNumber]
                        stats.setNetworkRead(result.frameNumber, start, end)
                        stats.setResultResponse(result.frameNumber, result)
                    }
                    yield(result)
                }
            }
        })
            .subscribeOn(IoScheduler())
            .publish()

        // val networkConfirmationResponses = networkRead!!
        networkRead!!
            .filter { it is ConfirmationResponse }
            .map { it as ConfirmationResponse }
            // TODO

        // val networkResultResponses = networkRead!!
        val networkResultResponsesSubscription = networkRead!!
            .filter { it is ResultResponse }
            .map { it as ResultResponse }
            .map { statistics[it.frameNumber].sample }
            .observeOn(AndroidSchedulers.mainThread())
            .doOnNext { Log.i(TAG, "Finished processing frame ${it.frameNumber}") }
            .subscribeBy(
                { it.printStackTrace() },
                { },
                { sample -> statisticsUiController.addSample(sample) }
            )
        subscriptions.add(networkResultResponsesSubscription)

        (networkRead!! as ConnectableObservable).connect()

        subscriptions.add(subscribeFrameProcessor())
    }

    // TODO Reduce requests if connection speed is not fast enough? (Backpressure here too!!!!!)
    private fun subscribeFrameProcessor() = frameProcessor
        .subscribeOn(IoScheduler())
        .map { it to modelUiController.modelConfig }
        .onBackpressureLimitRate(
            onDrop = { statistics[it.second].frameDropped() },
            limit = { shouldProcessFrame(statistics[it.second]) }
        )
        .doOnNext { prevFrameTime = Instant.now() }
        .zipWith(Flowable.range(0, Int.MAX_VALUE)) { (frame, modelConfig), i ->
            FrameRequest(frame, FrameRequestInfo(i, modelConfig))
        }
        .mapTimed(ModelStatistics::setPreprocess) { it.map(postprocessor::process) }
        .observeOn(inferenceScheduler, false, 1)
        .mapTimed(ModelStatistics::setInference) { inference.run(this, it) }
        .observeOn(networkWriteScheduler)
        .doOnNextTimed(ModelStatistics::setNetworkWrite) { networkAdapter!!.writeFrameRequest(it) }
        .doOnNext { statistics[it.info.modelConfig].setUpload(it.info.frameNumber, it.obj.size) }
        .subscribeBy({ it.printStackTrace() })

    private fun shouldProcessFrame(stats: ModelStatistics): Boolean {
        if (!stats.isFirstExist)
            return true

        // If first server response is not yet received, drop frame
        if (stats.isEmpty)
            return false

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
        val t = Duration.between(sample.preprocessStart, sample.inferenceEnd)
        if (networkAdapter!!.timeUntilWriteAvailable > t) {
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
        val rateLimit = maxOf(sample.serverInference, sample.clientInference)

        return Duration.between(prevFrameTime, Instant.now()) >= rateLimit
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

    companion object {
        private fun <R> timed(
            func: () -> R
        ): Triple<R, Instant, Instant> {
            val start = Instant.now()
            val result = func()
            val end = Instant.now()
            return Triple(result, start, end)
        }
    }
}