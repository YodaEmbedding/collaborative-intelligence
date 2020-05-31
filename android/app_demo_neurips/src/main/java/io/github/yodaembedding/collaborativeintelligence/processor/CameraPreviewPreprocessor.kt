package io.github.yodaembedding.collaborativeintelligence.processor

import android.content.Context
import android.content.Context.CAMERA_SERVICE
import android.graphics.ImageFormat
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.renderscript.*
import android.view.Surface.*
import android.view.WindowManager
import io.github.yodaembedding.collaborativeintelligence.ScriptC_convert
import io.github.yodaembedding.collaborativeintelligence.ScriptC_crop
import io.github.yodaembedding.collaborativeintelligence.ScriptC_rotate
import io.fotoapparat.preview.Frame
import java.lang.Math.floorMod

class CameraPreviewPreprocessor {
    var dataType: String = ""

    private val rs: RenderScript
    private val width: Int
    private val height: Int
    private val outWidth: Int
    private val outHeight: Int
    private val rotationCompensation: Int

    private val inputAllocation: Allocation
    private val rgbaAllocation: Allocation
    private val cropAllocation: Allocation
    private val resizeAllocation: Allocation
    private val rotateAllocation: Allocation
    private val outputFloatAllocation: Allocation
    private val outputUcharAllocation: Allocation
    private val outputArgbAllocation: Allocation

    private val yuvToRGB: ScriptIntrinsicYuvToRGB
    private val resize: ScriptIntrinsicResize
    private val crop: ScriptC_crop
    private val rotate: ScriptC_rotate
    private val convert: ScriptC_convert

    // TODO Convert to ScriptGroup for optimizations
    // TODO Blur https://medium.com/@petrakeas/alias-free-resize-with-renderscript-5bf15a86ce3
    // TODO Normalize [optional] with mean/std (R, G, B) tuples
    /**
    Process camera preview for model input.

    Convert YUV to RGBA, crop, resize, rotate, convert 32-bit RGBA to 24-bit RGB.
     */
    constructor(
        rs: RenderScript,
        width: Int,
        height: Int,
        outWidth: Int,
        outHeight: Int,
        rotationCompensation: Int
    ) {
        this.rs = rs
        this.width = width
        this.height = height
        this.outWidth = outWidth
        this.outHeight = outHeight
        this.rotationCompensation = rotationCompensation

        val side = minOf(width, height)
        val yuvType = Type.Builder(rs, Element.YUV(rs))
            .setX(width).setY(height)
            .setYuvFormat(ImageFormat.NV21).create()
        val rgbaType = Type.createXY(rs, Element.RGBA_8888(rs), width, height)
        val cropType = Type.createXY(rs, Element.RGBA_8888(rs), side, side)
        val resizeType = Type.createXY(rs, Element.RGBA_8888(rs), outWidth, outHeight)
        val rotateType = Type.createXY(rs, Element.RGBA_8888(rs), outWidth, outHeight)
        val outputFloatType = Type.createX(rs, Element.U8_4(rs), outWidth * outHeight * 3)
        val outputUcharType = Type.createX(rs, Element.U8(rs), outWidth * outHeight * 3)
        val outputArgbType = Type.createX(rs, Element.U8_4(rs), outWidth * outHeight)

        inputAllocation = Allocation.createTyped(rs, yuvType, Allocation.USAGE_SCRIPT)
        rgbaAllocation = Allocation.createTyped(rs, rgbaType, Allocation.USAGE_SCRIPT)
        cropAllocation = Allocation.createTyped(rs, cropType, Allocation.USAGE_SCRIPT)
        resizeAllocation = Allocation.createTyped(rs, resizeType, Allocation.USAGE_SCRIPT)
        rotateAllocation = Allocation.createTyped(rs, rotateType, Allocation.USAGE_SCRIPT)
        outputFloatAllocation = Allocation.createTyped(rs, outputFloatType, Allocation.USAGE_SCRIPT)
        outputUcharAllocation = Allocation.createTyped(rs, outputUcharType, Allocation.USAGE_SCRIPT)
        outputArgbAllocation = Allocation.createTyped(rs, outputArgbType, Allocation.USAGE_SCRIPT)

        yuvToRGB = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))
        crop = ScriptC_crop(rs)
        resize = ScriptIntrinsicResize.create(rs)
        rotate = ScriptC_rotate(rs)
        convert = ScriptC_convert(rs)

        yuvToRGB.setInput(inputAllocation)
        crop._input = rgbaAllocation
        crop._xStart = ((width - side) / 2).toLong()
        crop._yStart = ((height - side) / 2).toLong()
        resize.setInput(cropAllocation)
        rotate._input = resizeAllocation
        rotateUpdateParams()
        convert._width = outWidth.toLong()
    }

    // TODO Slight inefficiency from allocating new ByteArray instead of reusing a direct ByteBuffer
    fun process(frame: Frame): ByteArray {
        inputAllocation.copyFrom(frame.image)
        yuvToRGB.forEach(rgbaAllocation)
        crop.forEach_crop(cropAllocation)
        resize.forEach_bicubic(resizeAllocation)
        applyRotationCompensation(rotateAllocation)
        val outputAllocation = applyConversion(rotateAllocation)

        val byteArray = ByteArray(outputAllocation.bytesSize)
        outputAllocation.copyTo(byteArray)
        return byteArray
    }

    private fun applyConversion(allocation: Allocation): Allocation {
        when (dataType) {
            "float" -> {
                convert._output = outputFloatAllocation
                convert.forEach_rgba2rgbFloat(allocation)
                return outputFloatAllocation
            }
            "uchar" -> {
                convert._output = outputUcharAllocation
                convert.forEach_rgba2rgb(allocation)
                return outputUcharAllocation
            }
            "argb" -> {
                convert._output = outputArgbAllocation
                convert.forEach_rgba2argb(allocation)
                return outputArgbAllocation
            }
            else -> throw Exception()
        }
    }

    private fun applyRotationCompensation(aout: Allocation) {
        when (rotationCompensation) {
            0 -> return
            90 -> rotate.forEach_rotate90(aout)
            180 -> rotate.forEach_rotate180(aout)
            270 -> rotate.forEach_rotate270(aout)
            else -> throw IllegalArgumentException("Rotation required is not a multiple of 90")
        }
    }

    private fun rotateUpdateParams() {
        when (rotationCompensation) {
            0, 180 -> {
                rotate._width = outWidth.toLong()
                rotate._height = outHeight.toLong()
            }
            90, 270 -> {
                rotate._width = outHeight.toLong()
                rotate._height = outWidth.toLong()
            }
            else -> throw IllegalArgumentException("Rotation required is not a multiple of 90")
        }
    }

    companion object {
        private val rotationToDegrees = mapOf(
            ROTATION_0 to 0,
            ROTATION_90 to 90,
            ROTATION_180 to 180,
            ROTATION_270 to 270
        )

        @JvmStatic
        @Throws(CameraAccessException::class)
        fun getRotationCompensation(context: Context, facing: Int): Int {
            val windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
            val deviceRotation = rotationToDegrees.getValue(windowManager.defaultDisplay.rotation)

            val cameraManager = context.getSystemService(CAMERA_SERVICE) as CameraManager
            val characteristics = cameraManager.cameraIdList
                .map { cameraManager.getCameraCharacteristics(it) }
                .first { it.get(CameraCharacteristics.LENS_FACING) == facing }
            val sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION)!!

            val polarity = if (facing == CameraCharacteristics.LENS_FACING_FRONT) 1 else -1
            return floorMod(sensorOrientation + polarity * deviceRotation, 360)
        }
    }
}
