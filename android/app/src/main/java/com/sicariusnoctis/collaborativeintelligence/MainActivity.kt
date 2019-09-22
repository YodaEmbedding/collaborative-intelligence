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
import kotlinx.android.synthetic.main.bottom_sheet.*
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
        networkAdapter?.close()
    }

    override fun onDestroy() {
        subscriptions.forEach { x -> x.dispose() }
        super.onDestroy()
    }

    private fun initFotoapparat() {
        val cameraConfiguration = CameraConfiguration(
            pictureResolution = highestResolution(),
            previewResolution = ::selectPreviewResolution,
            previewFpsRange = ::selectFpsRange,
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

    private fun subscribeNetworkIo() {
        // TODO WRONG! some of the samples might be lost in transmission...
        // so you should explicitly have readData return sample number and item!
        // Note: if server changes the frame number, the client can crash... HMMM

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
                            i, null, { x -> networkAdapter!!.readData() },
                            statistics::setNetworkRead
                        ) ?: break
                        yield(Pair(i, item))
                        ++i
                    }
                }
            })
            // TODO note that server shouldn't guarantee in-order frame sends if not TCP...
            // .zipWith(Flowable.range(0, Int.MAX_VALUE)) { item, i -> Pair(i, item) }
            .subscribeOn(IoScheduler())
            .observeOn(AndroidSchedulers.mainThread())
            .subscribeBy(
                { it.printStackTrace() },
                { },
                { (i, message) ->
                    Log.i(TAG, "Received message $i:\n$message")
                    bottomSheetResults.text = message
                    bottomSheetStatistics.text = statistics.display()
                })

        // TODO Prevent duplicate subscriptions! (e.g. if onStart called multiple times); unsubscribe?
        // TODO .onTerminateDetach()
        val networkWriteSubscription = frameProcessor
            // TODO use IndexedValue<Frame>
            .zipWith(Flowable.range(0, Int.MAX_VALUE)) { frame, i -> Pair(i, frame) }
            // .subscribeOn(inferenceScheduler)
            // .subscribeOn(IoScheduler(), false)
            // .subscribeOn(IoScheduler(), true)
            .subscribeOn(IoScheduler())
            // .onBackpressureLatest()
            .onBackpressureDrop { statistics.frameDropped() }
            .doOnNext { Log.i(TAG, "Starting processing frame ${it.first}") }
            .mapTimed(statistics::setPreprocess) { postprocessor.process(it) }
            .doOnNext { statistics.appendSampleString(it.first, it.second.toPreviewString()) }
            // .observeOn(inferenceScheduler)
            .observeOn(inferenceScheduler, false, 1)
            .mapTimed(statistics::setInference) { inference.run(it) }
            .doOnNext { (i, x) -> statistics.appendSampleString(i, "\n${x.toPreviewString()}") }
            .observeOn(IoScheduler())
            .doOnNextTimed(statistics::setNetworkWrite) { networkAdapter!!.writeData(it) }
            .subscribeBy(
                { it.printStackTrace() },
                { },
                { Log.i(TAG, "Finished processing frame ${it.first}") }
            )

        subscriptions = listOf(networkReadSubscription, networkWriteSubscription)
    }

    // private fun <R, T, U, V> timed(
    //     timeFunc: (Int, Instant, Instant) -> Unit,
    //     mapper: (T) -> R,
    //     thisFunc: ((Pair<Int, T>) -> U) -> Flowable<Pair<Int, V>>,
    //     resultFunc: (Int, T, R) -> U
    // ): Flowable<Pair<Int, V>> {
    //     return thisFunc { (frameNum, x) ->
    //         val start = Instant.now()
    //         val result = mapper(x)
    //         val end = Instant.now()
    //         timeFunc(frameNum, start, end)
    //         resultFunc(frameNum, x, result)
    //     }
    // }
    //
    // Usages:
    // doOnNextTimed: return timed(timeFunc, onNext, this::doOnNext, { _, _, _ -> })
    // mapTimed:      return timed(timeFunc, mapper, this::map, { s, _, r -> Pair(s, r) })

    private fun <R, T> timeWrapper(
        frameNum: Int,
        x: T,
        mapper: (T) -> R,
        timeFunc: (Int, Instant, Instant) -> Unit
    ): R {
        val start = Instant.now()
        val result = mapper(x)
        val end = Instant.now()
        timeFunc(frameNum, start, end)
        return result
    }

    private fun <T> Flowable<Pair<Int, T>>.doOnNextTimed(
        timeFunc: (Int, Instant, Instant) -> Unit,
        onNext: (T) -> Unit
    ): Flowable<Pair<Int, T>> {
        return this.doOnNext { (frameNum, x) ->
            timeWrapper(frameNum, x, onNext, timeFunc)
        }
    }

    private fun <R, T> Flowable<Pair<Int, T>>.mapTimed(
        timeFunc: (Int, Instant, Instant) -> Unit,
        mapper: (T) -> R
    ): Flowable<Pair<Int, R>> {
        return this.map { (frameNum, x) ->
            Pair(frameNum, timeWrapper(frameNum, x, mapper, timeFunc))
        }
    }

    private fun ByteArray.toHexString() = joinToString("") { "%02x".format(it) }

    private fun ByteArray.toPreviewString() =
        take(12).toByteArray().toHexString() + "..." + takeLast(3).toByteArray().toHexString()

    companion object {
        @JvmStatic
        private fun selectPreviewResolution(iterable: Iterable<Resolution>): Resolution? {
            return iterable
                .sortedBy { it.area }
                .firstOrNull { it.width >= 224 * 2 && it.height >= 224 * 2 }
        }

        @JvmStatic
        private fun selectFpsRange(iterable: Iterable<FpsRange>): FpsRange? {
            return iterable
                .sortedBy { it.min }
                .firstOrNull { it.min >= 5000 }
        }
    }

    // TODO Stats: timer, battery, # frames dropped
    // TODO Video input

    // TODO E/BufferQueueProducer: [SurfaceTexture-0-22526-3] cancelBuffer: BufferQueue has been abandoned
    // TODO why does the stream freeze on server-side?
    // TODO run expensive operations in pipeline in parallel
}