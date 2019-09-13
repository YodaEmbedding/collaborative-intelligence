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
import io.fotoapparat.parameter.Resolution
import io.fotoapparat.parameter.ScaleType
import io.fotoapparat.parameter.camera.CameraParameters
import io.fotoapparat.preview.Frame
import io.fotoapparat.selector.*
import io.reactivex.Scheduler
import io.reactivex.Single
import io.reactivex.internal.schedulers.ComputationScheduler
import io.reactivex.internal.schedulers.IoScheduler
import io.reactivex.processors.PublishProcessor
import io.reactivex.rxkotlin.subscribeBy
import io.reactivex.schedulers.Schedulers
import kotlinx.android.synthetic.main.activity_main.*
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
    private var networkReadThread: NetworkReadThread? = null

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
        networkReadThread?.interrupt()
    }

    private fun initFotoapparat() {
        val cameraConfiguration = CameraConfiguration(
            pictureResolution = highestResolution(),
            previewResolution = highestResolution(),
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
                Log.i(TAG, "Received preview frame")
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
        networkReadThread = NetworkReadThread(networkAdapter!!)
        networkReadThread!!.outputStream
            .subscribeOn(IoScheduler())
            .subscribeBy { Log.i(TAG, "Received message: $it") }
        networkReadThread!!.start()

        // TODO Prevent duplicate subscriptions!
        // TODO Kill observable after some time?
        // TODO Backpressure strategies? Also check if frames are dropped...
        frameProcessor
            .subscribeOn(ComputationScheduler())
            // .doOnNext { Log.i(TAG, "Received preview frame") }
            .map { postprocessor.process(it) }
            .observeOn(inferenceScheduler)
            .map { inference.run(it) }
            .observeOn(IoScheduler())
            // TODO Handle java.net.SocketException (if disconnection occurs mid-write)
            .doOnNext { networkAdapter!!.writeData(it) }
            .subscribeBy(
                { it.printStackTrace() },
                { },
                // TODO save to stats on main thread? or observable...
                { })
    }

    // TODO I think the predictions are incorrect... check if rewinding... or uint8 types...
    // or correct resnet34.tflite... or image orientation!
    // or thread synchronization (overwrites?)... try disabling threading for a bit

    // TODO Stats: timer, battery, # frames dropped
    // TODO Video

    // TODO E/BufferQueueProducer: [SurfaceTexture-0-22526-3] cancelBuffer: BufferQueue has been abandoned
    // TODO why does the stream freeze on server-side?
    // TODO run expensive operations in pipeline in parallel
}