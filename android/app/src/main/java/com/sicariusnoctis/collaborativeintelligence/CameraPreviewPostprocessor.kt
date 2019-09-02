package com.sicariusnoctis.collaborativeintelligence

import android.graphics.ImageFormat
import android.renderscript.*
import io.fotoapparat.preview.Frame
import android.renderscript.Allocation
import android.util.Log

class CameraPreviewPostprocessor {
    private val TAG = CameraPreviewPostprocessor::class.qualifiedName

    private val rs: RenderScript
    private val width: Int
    private val height: Int

    // private val allocations: List<Allocation>
    // private val intrinsics: List<Script>
    // private val scriptGroup: ScriptGroup

    private val inputAllocation: Allocation
    private val rgbaAllocation: Allocation
    private val resizeAllocation: Allocation
    private val outputAllocation: Allocation

    private val yuvToRGB: ScriptIntrinsicYuvToRGB
    private val resize: ScriptIntrinsicResize
    private val preprocess: ScriptC_preprocess

    // private var yuvToRgbIntrinsic: ScriptIntrinsicYuvToRGB
    // private var resizeIntrinsic: ScriptIntrinsicResize
    // private var inData: Allocation
    // private var rgbaData: Allocation
    // private var resizedData: Allocation
    // private var inputAllocation: Allocation
    // private var scriptAllocation: Allocation
    // private var outputAllocation: Allocation
    // private var outputAllocationRGB: Allocation
    // private var scriptHandle: ScriptC_preprocess

    // TODO Process:
    // YUV -> RGBA
    // Crop
    // Blur
    // Resize
    // RGBA -> ARGB [reorderable]
    constructor(rs: RenderScript, width: Int, height: Int) {
        this.rs = rs
        this.width = width
        this.height = height

        val yuvType = Type.Builder(rs, Element.YUV(rs))
            .setX(width)
            .setY(height)
            .setYuvFormat(ImageFormat.NV21)
            .create()
        val rgbaType = Type.createXY(rs, Element.RGBA_8888(rs), width, height)
        val resizedType = Type.createXY(rs, Element.RGBA_8888(rs), 224, 224)
        val outputType = Type.createXY(rs, Element.RGBA_8888(rs), 224, 224)
        // val outputType = Type.createXY(rs, Element.U32(rs), 224, 224)

        inputAllocation = Allocation.createTyped(rs, yuvType, Allocation.USAGE_SCRIPT)
        rgbaAllocation = Allocation.createTyped(rs, rgbaType, Allocation.USAGE_SCRIPT)
        resizeAllocation = Allocation.createTyped(rs, resizedType, Allocation.USAGE_SCRIPT)
        outputAllocation = Allocation.createTyped(rs, outputType, Allocation.USAGE_SCRIPT)
        
        // TODO save these into some object instance list...? Prevent garbage collection?
        yuvToRGB = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))
        resize = ScriptIntrinsicResize.create(rs)
        preprocess = ScriptC_preprocess(rs)

        yuvToRGB.setInput(inputAllocation)
        resize.setInput(rgbaAllocation)

        // yuvToRGB.setInput(inputAllocation) // TODO remove
        //
        // var builder = ScriptGroup.Builder2(rs)
        // val unbound = builder.addInput()
        // val cYuvToRGB = builder.addKernel(
        //     yuvToRGB.kernelID,
        //     rgbaType,
        //     unbound
        // )
        // val cResize = builder.addKernel(
        //     resize.kernelID_bicubic,
        //     resizedType,
        //     cYuvToRGB.`return`
        // )
        // val cPreprocess = builder.addKernel(
        //     preprocess.kernelID_preprocess,
        //     outputType,
        //     cResize.`return`
        // )
        // // builder.addInvoke()
        //
        // scriptGroup =
        //     builder.create("CameraPreviewPostprocessor", cPreprocess.`return`)
    }

    fun process(frame: Frame): ByteArray {
        inputAllocation.copyFrom(frame.image)
        yuvToRGB.forEach(rgbaAllocation)
        resize.forEach_bicubic(resizeAllocation)
        preprocess.forEach_preprocess(resizeAllocation, outputAllocation)
        val byteArray = ByteArray(outputAllocation.bytesSize)
        outputAllocation.copyTo(byteArray)
        return byteArray

        // Log.e(TAG, "Look here")
        // inputAllocation.copyFrom(frame.image)
        // val outObj = scriptGroup.execute(inputAllocation)
        // Log.e(TAG, "${outObj.javaClass.canonicalName}")
        // outputAllocation.copyFrom(outObj)
        // val byteArray = ByteArray(outputAllocation.bytesSize)
        // outputAllocation.copyTo(byteArray)
        // return byteArray
    }

    // TODO ensure camera outputs NV21... or YUVxx? What is default setting?
    // TODO crop?
    // TODO RGBA?
    // TODO reduce rescale aliasing through blur https://medium.com/@petrakeas/alias-free-resize-with-renderscript-5bf15a86ce3
    // TODO correct orientation... shouldn't ALWAYS be rotating, though...
    // private fun rsProcess(frame: Frame): ByteArray {
    //     inData.copyFrom(frame.image)
    //     yuvToRgbIntrinsic.setInput(inData)
    //     yuvToRgbIntrinsic.forEach(rgbaData)
    //     resizeIntrinsic.setInput(rgbaData)
    //     resizeIntrinsic.forEach_bicubic(resizedData)
    //
    //     var outBuffer = ByteArray(resizedData.bytesSize)
    //     resizedData.copyTo(outBuffer)
    //
    //     return outBuffer
    // }

    // private fun initRenderScript(width: Int, height: Int) {
    //     val yuvType = Type.createX(rs, Element.U8(rs), width * height * 3 / 2)
    //     val rgbaType = Type.createXY(rs, Element.RGBA_8888(rs), width, height)
    //     val resizeType = Type.createXY(rs, Element.RGBA_8888(rs), 224, 224)
    //
    //     inData = Allocation.createTyped(rs, yuvType, Allocation.USAGE_SCRIPT)
    //     rgbaData = Allocation.createTyped(rs, rgbaType, Allocation.USAGE_SCRIPT)
    //     resizedData = Allocation.createTyped(rs, resizeType, Allocation.USAGE_SCRIPT)
    //
    //     yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))
    //     resizeIntrinsic = ScriptIntrinsicResize.create(rs)
    //
    //     isInitRenderScript = true
    // }

    // private fun initRenderScript(width: Int, height: Int) {
    //     val yuvTypeBuilder = Type.Builder(rs, Element.YUV(rs))
    //         .setX(width)
    //         .setY(height)
    //         .setYuvFormat(ImageFormat.NV21)
    //         .create()
    //     val rgbType = Type.createXY(rs, Element.RGBA_8888(rs), width, height)
    //     val rgbCroppedType = Type.createXY(rs, Element.RGBA_8888(rs), width, height)
    //
    //     inputAllocation = Allocation.createTyped(rs, yuvTypeBuilder, Allocation.USAGE_SCRIPT)
    //     scriptAllocation = Allocation.createTyped(rs, rgbType, Allocation.USAGE_SCRIPT)
    //     outputAllocation = Allocation.createTyped(rs, rgbType, Allocation.USAGE_SCRIPT)
    //     outputAllocationRGB = Allocation.createTyped(rs, rgbCroppedType, Allocation.USAGE_SCRIPT)
    //
    //     scriptHandle = ScriptC_preprocess(rs)
    //
    //     isInitRenderScript = true
    // }


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