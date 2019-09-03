package com.sicariusnoctis.collaborativeintelligence

import android.graphics.ImageFormat
import android.os.Bundle
import android.renderscript.*
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import io.fotoapparat.Fotoapparat
import io.fotoapparat.configuration.CameraConfiguration
import io.fotoapparat.log.logcat
import io.fotoapparat.log.loggers
import io.fotoapparat.parameter.ScaleType
import io.fotoapparat.parameter.camera.CameraParameters
import io.fotoapparat.preview.Frame
import io.fotoapparat.selector.*
import kotlinx.android.synthetic.main.activity_main.*

class MainActivity : AppCompatActivity() {
    private val TAG = MainActivity::class.qualifiedName
    private lateinit var fotoapparat: Fotoapparat
    private lateinit var networkThread: NetworkThread
    private lateinit var rs: RenderScript
    private lateinit var postprocessor: CameraPreviewPostprocessor
    private var cameraParameters: CameraParameters? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        rs = RenderScript.create(this)
        networkThread = NetworkThread() // TODO Don't need lateinit unless specifying IP, port
        initFotoapparat()
    }

    override fun onStart() {
        super.onStart()
        fotoapparat.start()
        fotoapparat.getCurrentParameters().whenAvailable {
            cameraParameters = it
            val previewResolution = cameraParameters?.previewResolution
            if (!::postprocessor.isInitialized) {
                postprocessor = CameraPreviewPostprocessor(
                    rs,
                    previewResolution!!.width,
                    previewResolution!!.height
                )
            }
        }
    }

    override fun onStop() {
        super.onStop()
        fotoapparat.stop()
    }

    override fun onResume() {
        super.onResume()
        networkThread.start()
    }

    override fun onPause() {
        super.onPause()
        networkThread.interrupt()
    }

    private fun bindViews() {
        // ...
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
            frameProcessor = this::frameProcessor
        )

        fotoapparat = Fotoapparat(
            context = this,
            view = cameraView,
            scaleType = ScaleType.CenterCrop,
            lensPosition = back(),
            cameraConfiguration = cameraConfiguration,
            logger = loggers(
                logcat()
                // fileLogger(this)
            ),
            cameraErrorCallback = { error ->
                Log.e(TAG, "$error")
            }
        )
    }

    // TODO what if processing is too slow? copy frame (meh), then skip frame processor if frame in-processing lock acquired? (or wait? nah...)
    private fun frameProcessor(frame: Frame) {
        val rgba = postprocessor.process(frame)
        // classifier.fillOutputByteArray(outputTransmittedBytes)
        val outputTransmittedBytes = rgba
        networkThread.writeData(outputTransmittedBytes)
    }

    // TODO E/BufferQueueProducer: [SurfaceTexture-0-22526-3] cancelBuffer: BufferQueue has been abandoned
    // TODO why does the stream freeze on server-side?
    // TODO run expensive operations in pipeline in parallel
}