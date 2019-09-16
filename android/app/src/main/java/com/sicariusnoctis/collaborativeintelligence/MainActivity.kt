package com.sicariusnoctis.collaborativeintelligence

import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.renderscript.RenderScript
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import io.fotoapparat.Fotoapparat
import io.fotoapparat.configuration.CameraConfiguration
import io.fotoapparat.log.logcat
import io.fotoapparat.log.loggers
import io.fotoapparat.parameter.FpsRange
import io.fotoapparat.parameter.Resolution
import io.fotoapparat.parameter.ScaleType
import io.fotoapparat.parameter.camera.CameraParameters
import io.fotoapparat.preview.Frame
import io.fotoapparat.selector.*
import io.reactivex.*
import io.reactivex.android.schedulers.AndroidSchedulers
import io.reactivex.disposables.Disposable
import io.reactivex.internal.schedulers.IoScheduler
import io.reactivex.processors.PublishProcessor
import io.reactivex.rxkotlin.subscribeBy
import io.reactivex.rxkotlin.zipWith
import io.reactivex.schedulers.Schedulers
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.android.synthetic.main.bottom_sheet.*
import org.reactivestreams.Subscriber
import java.time.Instant
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private val TAG = MainActivity::class.qualifiedName

    private lateinit var fotoapparat: Fotoapparat
    private lateinit var inference: Inference
    private lateinit var inferenceExecutor: ExecutorService
    private lateinit var inferenceScheduler: Scheduler
    private lateinit var postprocessor: CameraPreviewPostprocessor
    private lateinit var rs: RenderScript
    private val frameProcessor: PublishProcessor<Frame> = PublishProcessor.create()
    private var networkAdapter: NetworkAdapter? = null
    // private var networkReadThread: NetworkReadThread? = null
    private var statistics = Statistics()
    private var subscriptions = listOf<Disposable>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        rs = RenderScript.create(this)
        initFotoapparat()
    }

    override fun onStart() {
        super.onStart()
        fotoapparat.start()
        fotoapparat.getCurrentParameters().whenAvailable(this::onCameraParametersAvailable)
        inferenceExecutor = Executors.newSingleThreadExecutor()
        inferenceScheduler = Schedulers.from(inferenceExecutor)
        inferenceExecutor.submit { inference = Inference(this) }
        networkAdapter = NetworkAdapter()
        connectNetworkAdapter()
    }

    override fun onStop() {
        super.onStop()
        fotoapparat.stop()
        // TODO shouldn't this be part of "doFinally" for frameProcessor?
        inferenceExecutor.submit { inference.close() }
        // TODO release inferenceExecutor?
        // networkReadThread?.interrupt()
        subscriptions.forEach { x -> x.dispose() }
        networkAdapter?.close()
    }

    private fun initFotoapparat() {
        val cameraConfiguration = CameraConfiguration(
            pictureResolution = highestResolution(),
            previewResolution = this::selectPreviewResolution,
            previewFpsRange = highestFps(),
            focusMode = firstAvailable(
                continuousFocusPicture(),
                continuousFocusVideo(),
                autoFocus(),
                fixed()
            ),
            antiBandingMode = firstAvailable(
                auto(),
                hz50(),
                hz60(),
                none()
            ),
            jpegQuality = manualJpegQuality(90),
            sensorSensitivity = lowestSensorSensitivity(),
            // frameProcessor = this.frameProcessor::onNext
            frameProcessor = { frame: Frame ->
                // Log.i(TAG, "Received preview frame")
                this.frameProcessor.onNext(frame)
            }
        )

        fotoapparat = Fotoapparat(
            context = this,
            view = cameraView,
            scaleType = ScaleType.CenterCrop,
            lensPosition = back(),
            cameraConfiguration = cameraConfiguration,
            logger = loggers(
                logcat()
            ),
            cameraErrorCallback = { error ->
                Log.e(TAG, "$error")
            }
        )
    }

    private fun selectPreviewResolution(iterable: Iterable<Resolution>): Resolution? {
        return iterable
            .sortedBy { it.area }
            .firstOrNull { it.width >= 224 && it.height >= 224 }
    }

    private fun onCameraParametersAvailable(cameraParameters: CameraParameters?) {
        val resolution = cameraParameters!!.previewResolution

        if (::postprocessor.isInitialized)
            return

        val rotationCompensation =
            CameraPreviewPostprocessor.getRotationCompensation(
                this,
                CameraCharacteristics.LENS_FACING_BACK
            )

        postprocessor = CameraPreviewPostprocessor(
            rs,
            resolution.width,
            resolution.height,
            224,
            224,
            rotationCompensation
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

    private fun subscribeNetworkIo() {

        // TODO
        // Observable.range(0, Int.MAX_VALUE)
        //     .subscribeOn(IoScheduler())
        //     .doOnNextTimed { read() }
        //     // .doOnNext(read)
        //     // .doOnNext(process_read_item)
        //     .subscribeBy { }

        // TODO catch exception and emit onError(exception)

        val networkReadSubscription = Flowable
            .fromIterable(Iterable {
                iterator {
                    // TODO Wait until thread is alive using CountDownLatch?
                    // TODO thread.isAlive()? socket.isOpen? volatile boolean flag?
                    var i = 0
                    // TODO this code looks horrid...
                    while (true) {
                        // TODO pointless argument t = null
                        val item = timeWrapper(
                            i,
                            null,
                            { x -> networkAdapter!!.readData() },
                            statistics::setNetworkRead
                        )
                            ?: break
                        yield(Pair(i, item))
                        ++i
                    }
                }
            })
            // TODO note that server shouldn't guarantee in-order frame sends if not TCP...
            // .zipWith(Flowable.range(0, Int.MAX_VALUE)) { item, i -> Pair(i, item) }
            .subscribeOn(IoScheduler())
            .observeOn(AndroidSchedulers.mainThread())
            .subscribeBy {
                Log.i(TAG, "Received frame ${it.first}:\n${it.second}")
                bottomSheetResults.text = it.second
                bottomSheetStatistics.text = statistics.display()
            }

        // TODO add doOnNextTimed for read...

        // networkReadThread = NetworkReadThread(networkAdapter!!)
        // val networkReadSubscription = networkReadThread!!.outputStream
        //     .subscribeOn(IoScheduler())
        //     .observeOn(AndroidSchedulers.mainThread())
        //     .subscribeBy {
        //         Log.i(TAG, "Received message: $it")
        //         bottomSheetResults.text = it
        //         bottomSheetStatistics.text = statistics.display()
        //     }
        // networkReadThread!!.start()

        // TODO .onTerminateDetach()

        // TODO Prevent duplicate subscriptions! (e.g. if onStart called multiple times); unsubscribe?
        // TODO Kill observable after some time?
        // TODO Backpressure strategies? Also check if frames are dropped...
        val networkWriteSubscription = frameProcessor
            .zipWith(Flowable.range(0, Int.MAX_VALUE)) { frame, i -> Pair(i, frame) }
            // TODO Don't do everything on inference thread...
            // .subscribeOn(ComputationScheduler())
            .subscribeOn(inferenceScheduler)
            // TODO always keep one frame in buffer so that we don't have to wait?
            .onBackpressureDrop { statistics.frameDropped() }
            .doOnNext { Log.i(TAG, "Starting processing frame ${it.first}") }
            .mapTimed(statistics::setPreprocess) { postprocessor.process(it) }
            // TODO implement pull strategy so backpressure doesn't build due to inference
            // .observeOn(inferenceScheduler)
            .mapTimed(statistics::setInference) { inference.run(it) }
            .observeOn(IoScheduler())
            // TODO Handle java.net.SocketException (if disconnection occurs mid-write)
            .doOnNextTimed(statistics::setNetworkWrite) { networkAdapter!!.writeData(it) }
            .subscribeBy(
                { it.printStackTrace() },
                { },
                // TODO save to stats on main thread? or observable...
                { Log.i(TAG, "Finished processing frame ${it.first}") }
            )

        subscriptions = listOf(networkReadSubscription, networkWriteSubscription)
    }

    // private fun <R, S, T, U, V> timed(
    //     timeFunc: (S, Instant, Instant) -> Unit,
    //     mapper: (T) -> R,
    //     thisFunc: ((Pair<S, T>) -> U) -> Flowable<Pair<S, V>>,
    //     resultFunc: (S, T, R) -> U
    // ): Flowable<Pair<S, V>> {
    //     return thisFunc { (s, t) ->
    //         // Consider using Clock.systemUTC().instant()
    //         // or Instant.now(Clock.systemUTC())
    //         val start = Instant.now()
    //         val r = mapper(t)
    //         val end = Instant.now()
    //         timeFunc(s, start, end)
    //         resultFunc(s, t, r)
    //     }
    // }

    // TODO replace S with Int...

    private fun <R, S, T> timeWrapper(
        s: S,
        t: T,
        mapper: (T) -> R,
        timeFunc: (S, Instant, Instant) -> Unit
    ): R {
        // Consider using Clock.systemUTC().instant()
        // or Instant.now(Clock.systemUTC())
        val start = Instant.now()
        val result = mapper(t)
        val end = Instant.now()
        timeFunc(s, start, end)
        return result
    }

    private fun <S, T> Flowable<Pair<S, T>>.doOnNextTimed(
        timeFunc: (S, Instant, Instant) -> Unit,
        onNext: (T) -> Unit
    ): Flowable<Pair<S, T>> {
        return this.doOnNext { (s, t) ->
            timeWrapper(s, t, onNext, timeFunc)
        }
        // return timed(timeFunc, onNext, this::doOnNext, { _, _, _ -> })
    }

    private fun <R, S, T> Flowable<Pair<S, T>>.mapTimed(
        timeFunc: (S, Instant, Instant) -> Unit,
        mapper: (T) -> R
    ): Flowable<Pair<S, R>> {
        return this.map { (s, t) ->
            Pair(s, timeWrapper(s, t, mapper, timeFunc))
        }
        // return timed(timeFunc, mapper, this::map, { s, _, r -> Pair(s, r) })
    }

    companion object {
        @JvmStatic
        private fun selectPreviewResolution(iterable: Iterable<Resolution>): Resolution? {
            return iterable
                .sortedBy { it.area }
                .firstOrNull { it.width >= 224 && it.height >= 224 }
        }

        @JvmStatic
        private fun selectFpsRange(iterable: Iterable<FpsRange>): FpsRange? {
            return iterable
                .sortedBy { it.min }
                .firstOrNull { it.min >= 5000 }
        }
    }

    // TODO I think the predictions are incorrect... check if rewinding... or uint8 types...
    // or correct resnet34.tflite... or image orientation!
    // or correct weights from memory?
    // or thread synchronization (overwrites?)... try disabling threading for a bit
    // check what data is computed by looking at bytes
    // compare various data (e.g. red color, fully black image, etc) with working version

    // TODO Stats: timer, battery, # frames dropped
    // TODO Video

    // TODO E/BufferQueueProducer: [SurfaceTexture-0-22526-3] cancelBuffer: BufferQueue has been abandoned
    // TODO why does the stream freeze on server-side?
    // TODO run expensive operations in pipeline in parallel
}