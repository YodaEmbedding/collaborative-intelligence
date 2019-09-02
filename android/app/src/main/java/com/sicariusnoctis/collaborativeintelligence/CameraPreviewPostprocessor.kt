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

    private val cropLaunchOptions: LaunchOptions

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

        val xStart = (width - side) / 2
        val yStart = (height - side) / 2
        cropLaunchOptions = LaunchOptions()
            .setX(xStart, xStart + side)
            .setY(yStart, yStart + side)

        yuvToRGB.setInput(inputAllocation)
        resize.setInput(cropAllocation)
        crop._xStart = xStart.toLong()
        crop._yStart = yStart.toLong()
        crop._output = cropAllocation
    }

    fun process(frame: Frame): ByteArray {
        inputAllocation.copyFrom(frame.image)
        yuvToRGB.forEach(rgbaAllocation)
        crop.forEach_crop(rgbaAllocation, cropLaunchOptions) // TODO the arguments could be reversed...
        resize.forEach_bicubic(resizeAllocation)
        preprocess.forEach_preprocess(resizeAllocation, outputAllocation)
        val byteArray = ByteArray(outputAllocation.bytesSize)
        outputAllocation.copyTo(byteArray)
        return byteArray
    }

    // private fun rsProcess(frame: Frame): ByteArray {
    //     inputAllocation.copyFrom(frame.image)
    //
    //     // TODO do we really need to do this every time...? Maybe if orientation changes, yeah...? idk
    //     var lo = getScriptLaunchOptions(frame.size.width, frame.size.height)
    //
    //     // TODO watch out! the cropped output size is 224 x 224!!
    //
    //     scriptHandle.set_xStart(lo.xStart.toLong())
    //     scriptHandle.set_yStart(lo.yStart.toLong())
    //     scriptHandle.set_outputWidth(frame.size.width.toLong())
    //     scriptHandle.set_outputHeight(frame.size.height.toLong())
    //
    //     scriptHandle.forEach_preprocess(scriptAllocation, outputAllocation, lo)
    //
    //     var outBuffer = ByteArray(outputAllocationRGB.bytesSize)
    //     outputAllocationRGB.copyTo(outBuffer)
    //     return outBuffer
    // }

    // private fun getScriptLaunchOptions(width: Int, height: Int): Script.LaunchOptions {
    //     /*
    //      * These coordinates are the portion of the original image that we want to
    //      * include.  Because we're rotating (in this case) x and y are reversed
    //      * (but still offset from the actual center of each dimension)
    //      */
    //     return Script.LaunchOptions()
    //         .setX(0, width)
    //         .setY(0, height)
    //     // return Script.LaunchOptions()
    //     //     .setX(starty, endy)
    //     //     .setY(startx, endx)
    // }
}