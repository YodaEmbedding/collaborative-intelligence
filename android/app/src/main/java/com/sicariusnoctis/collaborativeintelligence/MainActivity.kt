package com.sicariusnoctis.collaborativeintelligence

import android.content.Context
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
    lateinit var fotoapparat: Fotoapparat
    lateinit var networkThread: NetworkThread

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        initFotoapparat()
        networkThread = NetworkThread() // TODO No real need to lateinit... unless specifying IP, port
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
        // Log.d(TAG, frame.size.toString())
        val rgba = rsProcess(this, frame.image, frame.size.width, frame.size.height)
        // Log.d(TAG, "${rgba.size}")
        // Log.d(TAG, "${rgba[0]} ${rgba[1]} ${rgba[2]} ${rgba[3]}")
        // classifier.fillOutputByteArray(outputTransmittedBytes)
        val outputTransmittedBytes = rgba
        networkThread.writeData(outputTransmittedBytes)
    }

    // TODO ensure camera outputs NV21... or YUVxx? What is default setting?
    // TODO correct orientation...
    // TODO crop?
    // TODO RGBA?
    // TODO reduce rescale aliasing through blur https://medium.com/@petrakeas/alias-free-resize-with-renderscript-5bf15a86ce3
    private fun rsProcess(context: Context, byteArray: ByteArray, width: Int, height: Int): ByteArray {
        val rs = RenderScript.create(context)

        val yuvType = Type.createX(rs, Element.U8(rs), byteArray.size)
        val rgbaType = Type.createXY(rs, Element.RGBA_8888(rs), width, height)
        val resizeType = Type.createXY(rs, Element.RGBA_8888(rs), 224, 224)

        val inData = Allocation.createTyped(rs, yuvType, Allocation.USAGE_SCRIPT)
        val rgbaData = Allocation.createTyped(rs, rgbaType, Allocation.USAGE_SCRIPT)
        val resizedData = Allocation.createTyped(rs, resizeType, Allocation.USAGE_SCRIPT)

        val yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))
        val resizeIntrinsic = ScriptIntrinsicResize.create(rs)

        inData.copyFrom(byteArray)
        yuvToRgbIntrinsic.setInput(inData)
        yuvToRgbIntrinsic.forEach(rgbaData)
        resizeIntrinsic.setInput(rgbaData)
        resizeIntrinsic.forEach_bicubic(resizedData)

        var outBuffer = ByteArray(resizedData.bytesSize)
        resizedData.copyTo(outBuffer)

        return outBuffer

        //  TODO center crop to square? 224x224
        //
        // frameToCropTransform =
        //     ImageUtils.getTransformationMatrix(
        //         previewWidth,
        //         previewHeight,
        //         classifier.getImageSizeX(),
        //         classifier.getImageSizeY(),
        //         sensorOrientation,
        //         MAINTAIN_ASPECT);
        //
        // val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        // outData.copyTo(bitmap)
    }

    // TODO W/System: A resource failed to call destroy.
    // TODO E/BufferQueueProducer: [SurfaceTexture-0-22526-3] cancelBuffer: BufferQueue has been abandoned
    // TODO why does the stream freeze on server-side?
}