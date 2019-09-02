package com.sicariusnoctis.collaborativeintelligence

import android.graphics.ImageFormat
import android.renderscript.*
import android.renderscript.Script.LaunchOptions
import io.fotoapparat.preview.Frame

class CameraPreviewPostprocessor {
    private val TAG = CameraPreviewPostprocessor::class.qualifiedName

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
}