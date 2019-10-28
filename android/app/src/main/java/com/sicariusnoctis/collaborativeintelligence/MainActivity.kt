package com.sicariusnoctis.collaborativeintelligence

import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.renderscript.RenderScript
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import io.fotoapparat.parameter.camera.CameraParameters
import io.fotoapparat.preview.Frame
import io.reactivex.Flowable
import io.reactivex.Scheduler
import io.reactivex.Single
import io.reactivex.android.schedulers.AndroidSchedulers
import io.reactivex.disposables.Disposable
import io.reactivex.internal.schedulers.IoScheduler
import io.reactivex.processors.PublishProcessor
import io.reactivex.rxkotlin.subscribeBy
import io.reactivex.rxkotlin.zipWith
import io.reactivex.schedulers.Schedulers
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.serialization.ImplicitReflectionSerializer
import java.time.Duration
import java.time.Instant
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private val TAG = MainActivity::class.qualifiedName

    private lateinit var camera: Camera
    private lateinit var inference: Inference
    private lateinit var inferenceExecutor: ExecutorService
    private lateinit var inferenceScheduler: Scheduler
    private lateinit var postprocessor: CameraPreviewPostprocessor
    private lateinit var rs: RenderScript
    private lateinit var modelUiController: ModelUiController
    private lateinit var statisticsUiController: StatisticsUiController
    private val frameProcessor: PublishProcessor<Frame> = PublishProcessor.create()
    private var networkAdapter: NetworkAdapter? = null
    private val statistics = Statistics()
    private var subscriptions = listOf<Disposable>()
    private var prevFrameTime: Instant? = null

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
        camera.start(this::onCameraParametersAvailable)
        inferenceExecutor = Executors.newSingleThreadExecutor()
        inferenceScheduler = Schedulers.from(inferenceExecutor)
        inferenceExecutor.submit { inference = Inference() }
        networkAdapter = NetworkAdapter()
        connectNetworkAdapter()
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
        subscriptions.forEach { x -> x.dispose() }
        super.onDestroy()
    }

    private fun onCameraParametersAvailable(cameraParameters: CameraParameters?) {
        val resolution = cameraParameters!!.previewResolution

        if (::postprocessor.isInitialized)
            return

        val rotationCompensation =
            CameraPreviewPostprocessor.getRotationCompensation(
                this, CameraCharacteristics.LENS_FACING_BACK
            )

        postprocessor = CameraPreviewPostprocessor(
            rs, resolution.width, resolution.height, 224, 224, rotationCompensation
        )
    }

    private fun connectNetworkAdapter() {
        Single.just(networkAdapter!!)
            .subscribeOn(IoScheduler())
            .doOnSuccess { it.connect() }
            .subscribeBy(
                { it.printStackTrace() },
                { subscribeNetworkIo() }
            )
    }

    @UseExperimental(ImplicitReflectionSerializer::class)
    private fun subscribeNetworkIo() {
        val networkReadSubscription = Flowable
            .fromIterable(Iterable {
                iterator {
                    // TODO Wait until thread is alive using CountDownLatch?
                    // TODO thread.isAlive()? socket.isOpen? volatile boolean flag?
                    while (true) {
                        val (result, start, end) = timed { networkAdapter!!.readData() }
                        if (result == null) break
                        statistics.setNetworkRead(result.frameNumber, start, end)
                        statistics.setResultResponse(result.frameNumber, result)
                        yield(statistics.sample)
                    }
                }
            })
            .subscribeOn(IoScheduler())
            .observeOn(AndroidSchedulers.mainThread())
            .doOnNext { Log.i(TAG, "Finished processing frame ${it.frameNumber}") }
            .subscribeBy(
                { it.printStackTrace() },
                { },
                { sample -> statisticsUiController.addSample(sample) }
            )

        // TODO Reduce requests if connection speed is not fast enough? (Backpressure here too!!!!!)
        // TODO Prevent duplicate subscriptions! (e.g. if onStart called multiple times); unsubscribe?
        // TODO .onTerminateDetach()
        val networkWriteSubscription = frameProcessor
            .onBackpressureLimitRate { statistics.frameDropped() }
            .zipWith(Flowable.range(0, Int.MAX_VALUE)) { frame, i ->
                Log.i(TAG, "zip($i, ${modelUiController.modelConfig})")
                FrameRequest(frame, FrameRequestInfo(i, modelUiController.modelConfig))
            }
            .subscribeOn(IoScheduler())
            .mapTimed(statistics::setPreprocess) { it.map(postprocessor::process) }
            .observeOn(inferenceScheduler, false, 1)
            .mapTimed(statistics::setInference) { inference.run(this, it) }
            .observeOn(IoScheduler())
            .doOnNextFrameTimed(statistics::setNetworkWrite) { networkAdapter!!.writeFrameRequest(it) }
            .doOnNext { statistics.setUpload(it.info.frameNumber, it.obj.size) }
            .subscribeBy(
                { it.printStackTrace() },
                { },
                { }
            )

        subscriptions = listOf(networkReadSubscription, networkWriteSubscription)
    }

    // TODO nullify prevFrameTime onModelConfigChanged? (for smoother transition)
    // TODO immediately send onModelConfigChanged request to server, instead of waiting for response?
    // no, because server doesn't cancel inferences... yet
    private fun <T> Flowable<T>.onBackpressureLimitRate(onDrop: (T) -> Unit): Flowable<T> {
        return this
            .onBackpressureDrop(onDrop)
            .filter {
                if (prevFrameTime == null)
                    return@filter true

                // TODO reset statistics on model switch...

                // If first server response is not yet received, drop frame
                if (statistics.samples.isEmpty()) {
                    onDrop(it)
                    return@filter false
                }

                // TODO this maintains a reasonable FPS, but doesn't help as much with latency (which can accumulate)

                // TODO rate limit upload in a smarter way... maybe subtract ping and check upload? idk
                val sample = statistics.sample
                val fpsLimit = Duration.ofMillis((1000 / (statistics.fps * 1.3)).toLong())
                // val networkLimit = minOf(sample.networkWrite, Duration.ofMillis(50))
                val rateLimit = maxOf(fpsLimit, sample.serverInference)

                if (Duration.between(prevFrameTime, Instant.now()) < rateLimit) {
                    onDrop(it)
                    return@filter false
                }

                true
            }.doOnNext {
                prevFrameTime = Instant.now()
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

        private fun <T> Flowable<FrameRequest<T>>.doOnNextFrameTimed(
            timeFunc: (Int, Instant, Instant) -> Unit,
            onNext: (FrameRequest<T>) -> Unit
        ): Flowable<FrameRequest<T>> {
            return this.doOnNext { x ->
                val (_, start, end) = timed { onNext(x) }
                timeFunc(x.info.frameNumber, start, end)
            }
        }

        private fun <R, T> Flowable<FrameRequest<T>>.mapTimed(
            timeFunc: (Int, Instant, Instant) -> Unit,
            mapper: (FrameRequest<T>) -> FrameRequest<R>
        ): Flowable<FrameRequest<R>> {
            return this.map { x ->
                val (result, start, end) = timed { mapper(x) }
                timeFunc(result.info.frameNumber, start, end)
                result
            }
        }
    }
}