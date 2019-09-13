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
import io.fotoapparat.parameter.ScaleType
import io.fotoapparat.preview.Frame
import io.fotoapparat.selector.*
import io.reactivex.internal.schedulers.ComputationScheduler
import io.reactivex.internal.schedulers.IoScheduler
import io.reactivex.processors.PublishProcessor
import io.reactivex.rxkotlin.subscribeBy
import kotlinx.android.synthetic.main.activity_main.*

class MainActivity : AppCompatActivity() {
    private val TAG = MainActivity::class.qualifiedName

    private lateinit var fotoapparat: Fotoapparat
    private lateinit var inference: Inference
    private lateinit var postprocessor: CameraPreviewPostprocessor
    private lateinit var rs: RenderScript
    private val frameProcessor: PublishProcessor<Frame> = PublishProcessor.create()
    private var networkThread: NetworkThread? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        rs = RenderScript.create(this)
        initFotoapparat()
    }

    override fun onStart() {
        super.onStart()
        fotoapparat.start()
        // TODO extract to reduce clutter...
        fotoapparat.getCurrentParameters().whenAvailable { cameraParameters ->
            val resolution = cameraParameters?.previewResolution!!
            if (!::postprocessor.isInitialized) {
                postprocessor = CameraPreviewPostprocessor(
                    rs,
                    resolution.width,
                    resolution.height,
                    224,
                    224,
                    CameraPreviewPostprocessor.getRotationCompensation(
                        this,
                        CameraCharacteristics.LENS_FACING_BACK
                    )
                )
            }
        }

        inference = Inference(this)

        networkThread = NetworkThread()
        networkThread?.start()

        // TODO Backpressure strategies? Also check if frames are dropped...
        frameProcessor
            .map { postprocessor.process(it) }
            .map { inference.run(it) }
            .observeOn(IoScheduler())
            .doOnNext { networkThread?.writeData(it) }
            .subscribeOn(ComputationScheduler())
            .subscribeBy(
                { error ->
                    error.printStackTrace()
                },
                { },
                { item ->
                    {} // maybe save to stats on main thread? or observable...
                })
    }

    override fun onStop() {
        super.onStop()
        fotoapparat.stop()
        networkThread?.interrupt()
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
            frameProcessor = this.frameProcessor!!::onNext
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

    // TODO make this part of RxJava stream...
    // TODO what if processing is too slow?
    // TODO copy frame (meh), then skip frame processor if frame in-processing lock acquired? or wait?
    // private fun frameProcessor(frame: Frame) {
    //     val inputBytes = postprocessor.process(frame)
    //     // classifier.startInference(inputBytes, future_callback)
    //     val outputTransmittedBytes = inputBytes
    //     networkThread?.writeData(outputTransmittedBytes)
    // }

    // TODO Inference
    // TODO Stats
    // TODO Network receiver
    // TODO Video

    // TODO E/BufferQueueProducer: [SurfaceTexture-0-22526-3] cancelBuffer: BufferQueue has been abandoned
    // TODO why does the stream freeze on server-side?
    // TODO run expensive operations in pipeline in parallel
}