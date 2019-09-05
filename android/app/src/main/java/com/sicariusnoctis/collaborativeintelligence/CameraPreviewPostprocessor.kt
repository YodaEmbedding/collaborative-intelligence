package com.sicariusnoctis.collaborativeintelligence

import android.content.Context
import android.content.Context.CAMERA_SERVICE
import android.graphics.ImageFormat
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.renderscript.*
import android.util.Log
import android.view.Surface.*
import android.view.WindowManager
import io.fotoapparat.characteristic.LensPosition
import io.fotoapparat.preview.Frame
import java.lang.Math.floorMod

class CameraPreviewPostprocessor {
    private val rs: RenderScript
    private val width: Int
    private val height: Int

    private val inputAllocation: Allocation
    private val rgbaAllocation: Allocation
    private val cropAllocation: Allocation
    private val resizeAllocation: Allocation
    private val outputAllocation: Allocation

    private val yuvToRGB: ScriptIntrinsicYuvToRGB
    private val resize: ScriptIntrinsicResize
    private val crop: ScriptC_crop
    private val preprocess: ScriptC_preprocess // TODO rename to more descriptive

    // TODO ensure camera outputs NV21... or YUVxx? What is default setting?
    // TODO... write doc-comment
    // YUV -> RGBA
    // Crop
    // Blur https://medium.com/@petrakeas/alias-free-resize-with-renderscript-5bf15a86ce3
    // Resize
    // RGBA -> ARGB [reorderable]
    // Rotate [optionally, if required]
    // RGBA -> array((224, 224, 3), dtype=float32)
    // normalize [optional] with IMAGE_MEAN, IMAGE_STD
    constructor(rs: RenderScript, width: Int, height: Int) {
        this.rs = rs
        this.width = width
        this.height = height

        val side = minOf(width, height)
        val yuvType = Type.Builder(rs, Element.YUV(rs))
            .setX(width)
            .setY(height)
            .setYuvFormat(ImageFormat.NV21)
            .create()
        val rgbaType = Type.createXY(rs, Element.RGBA_8888(rs), width, height)
        val cropType = Type.createXY(rs, Element.RGBA_8888(rs), side, side)
        val resizeType = Type.createXY(rs, Element.RGBA_8888(rs), 224, 224)
        val outputType = Type.createXY(rs, Element.RGBA_8888(rs), 224, 224)
        // val outputType = Type.createXY(rs, Element.U32(rs), 224, 224)

        inputAllocation = Allocation.createTyped(rs, yuvType, Allocation.USAGE_SCRIPT)
        rgbaAllocation = Allocation.createTyped(rs, rgbaType, Allocation.USAGE_SCRIPT)
        cropAllocation = Allocation.createTyped(rs, cropType, Allocation.USAGE_SCRIPT)
        resizeAllocation = Allocation.createTyped(rs, resizeType, Allocation.USAGE_SCRIPT)
        outputAllocation = Allocation.createTyped(rs, outputType, Allocation.USAGE_SCRIPT)

        yuvToRGB = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))
        crop = ScriptC_crop(rs)
        resize = ScriptIntrinsicResize.create(rs)
        preprocess = ScriptC_preprocess(rs)

        yuvToRGB.setInput(inputAllocation)
        crop._xStart = ((width - side) / 2).toLong()
        crop._yStart = ((height - side) / 2).toLong()
        crop._input = rgbaAllocation
        resize.setInput(cropAllocation)
    }

    fun process(frame: Frame): ByteArray {
        inputAllocation.copyFrom(frame.image)
        yuvToRGB.forEach(rgbaAllocation)
        crop.forEach_crop(cropAllocation)
        resize.forEach_bicubic(resizeAllocation)
        preprocess.forEach_preprocess(resizeAllocation, outputAllocation)

        val byteArray = ByteArray(outputAllocation.bytesSize)
        outputAllocation.copyTo(byteArray)
        return byteArray
    }

    companion object {
        private val TAG = CameraPreviewPostprocessor::class.qualifiedName

        private val rotationToDegrees = mapOf(
            ROTATION_0 to 0,
            ROTATION_90 to 90,
            ROTATION_180 to 180,
            ROTATION_270 to 270
        )

        private val ORIENTATIONS = mapOf(
            ROTATION_0 to 90,
            ROTATION_90 to 0,
            ROTATION_180 to 270,
            ROTATION_270 to 180
        )

        @JvmStatic
        @Throws(CameraAccessException::class)
        fun getRotationCompensation(context: Context, cameraId: String): Int {
            // Get the device's current rotation relative to its "native" orientation.
            // Then, from the ORIENTATIONS table, look up the angle the image must be
            // rotated to compensate for the device's rotation.
            val windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
            val deviceRotationAdjusted = ORIENTATIONS.getValue(windowManager.defaultDisplay.rotation)

            // On most devices, the sensor orientation is 90 degrees, but for some
            // devices it is 270 degrees. For devices with a sensor orientation of
            // 270, rotate the image an additional 180 ((270 + 270) % 360) degrees.
            val cameraManager = context.getSystemService(CAMERA_SERVICE) as CameraManager
            val sensorOrientation = cameraManager
                .getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.SENSOR_ORIENTATION)!!

            return (deviceRotationAdjusted + sensorOrientation + 270) % 360

            // return rotationToDegrees.entries.first { it.value == rotationCompensation }.key
        }

        @JvmStatic
        @Throws(CameraAccessException::class)
        fun getRotation(context: Context, facing: Int): Int {
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

        @JvmStatic
        fun lensPositionToLensFacing(lensPosition: LensPosition): Int {
            return when (lensPosition) {
                LensPosition.Front -> CameraCharacteristics.LENS_FACING_FRONT
                LensPosition.Back -> CameraCharacteristics.LENS_FACING_BACK
                LensPosition.External -> CameraCharacteristics.LENS_FACING_EXTERNAL
            }
        }

        @JvmStatic
        fun getCameraId(context: Context, facing: Int): String {
            val manager = context.getSystemService(CAMERA_SERVICE) as CameraManager

            return manager.cameraIdList.first {
                manager
                    .getCameraCharacteristics(it)
                    .get(CameraCharacteristics.LENS_FACING) == facing
            }
        }
    }
}