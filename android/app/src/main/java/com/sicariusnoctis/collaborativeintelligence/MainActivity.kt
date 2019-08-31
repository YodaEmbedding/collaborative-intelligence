package com.sicariusnoctis.collaborativeintelligence

import android.content.Context
import android.graphics.Bitmap
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
import io.fotoapparat.preview.Frame
import io.fotoapparat.selector.*
import kotlinx.android.synthetic.main.activity_main.*

class MainActivity : AppCompatActivity() {
    private val TAG = MainActivity::class.qualifiedName
    private var isInitRenderScript = false
    private lateinit var fotoapparat: Fotoapparat
    private lateinit var networkThread: NetworkThread
    private lateinit var rs: RenderScript
    private lateinit var yuvToRgbIntrinsic: ScriptIntrinsicYuvToRGB
    private lateinit var resizeIntrinsic: ScriptIntrinsicResize
    private lateinit var inData: Allocation
    private lateinit var rgbaData: Allocation
    private lateinit var resizedData: Allocation


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        initFotoapparat()
        networkThread = NetworkThread() // TODO No real need to lateinit... unless specifying IP, port
        rs = RenderScript.create(this)
    }

    override fun onStart() {
        super.onStart()
        fotoapparat.start()
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
        if (!isInitRenderScript) {
            initRenderScript(frame.size.width, frame.size.height)
        }

        val rgba = rsProcess(frame.image)
        // classifier.fillOutputByteArray(outputTransmittedBytes)
        val outputTransmittedBytes = rgba
        networkThread.writeData(outputTransmittedBytes)
    }

    // TODO ensure camera outputs NV21... or YUVxx? What is default setting?
    // TODO crop?
    // TODO RGBA?
    // TODO reduce rescale aliasing through blur https://medium.com/@petrakeas/alias-free-resize-with-renderscript-5bf15a86ce3
    // TODO correct orientation... shouldn't ALWAYS be rotating, though...
    private fun rsProcess(byteArray: ByteArray): ByteArray {
        inData.copyFrom(byteArray)
        yuvToRgbIntrinsic.setInput(inData)
        yuvToRgbIntrinsic.forEach(rgbaData)
        resizeIntrinsic.setInput(rgbaData)
        resizeIntrinsic.forEach_bicubic(resizedData)

        var outBuffer = ByteArray(resizedData.bytesSize)
        resizedData.copyTo(outBuffer)

        return outBuffer
    }

    private fun initRenderScript(width: Int, height: Int) {
        val yuvType = Type.createX(rs, Element.U8(rs), width * height * 3 / 2)
        val rgbaType = Type.createXY(rs, Element.RGBA_8888(rs), width, height)
        val resizeType = Type.createXY(rs, Element.RGBA_8888(rs), 224, 224)

        inData = Allocation.createTyped(rs, yuvType, Allocation.USAGE_SCRIPT)
        rgbaData = Allocation.createTyped(rs, rgbaType, Allocation.USAGE_SCRIPT)
        resizedData = Allocation.createTyped(rs, resizeType, Allocation.USAGE_SCRIPT)

        yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))
        resizeIntrinsic = ScriptIntrinsicResize.create(rs)

        isInitRenderScript = true
    }

    // TODO E/BufferQueueProducer: [SurfaceTexture-0-22526-3] cancelBuffer: BufferQueue has been abandoned
    // TODO why does the stream freeze on server-side?
    // TODO run expensive operations in pipeline in parallel
}