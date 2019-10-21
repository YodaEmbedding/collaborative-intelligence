package com.sicariusnoctis.collaborativeintelligence

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder.nativeOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel.MapMode.READ_ONLY

class Inference : Closeable {
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputBuffer: ByteBuffer
    private lateinit var modelConfig: ModelConfig
    private val gpuDelegate: GpuDelegate = GpuDelegate()
    private val tfliteOptions = Interpreter.Options()
    private var tflite: Interpreter? = null
    private var tfliteModel: MappedByteBuffer? = null

    init {
        // TODO NNAPI only faster than GPU/CPU on some devices/processors with certain models. Add UI switch?
        tfliteOptions
            .setNumThreads(1)
            // .setUseNNAPI(true)
            .addDelegate(gpuDelegate)
    }

    fun run(context: Context, frameRequest: FrameRequest<ByteArray>): FrameRequest<ByteArray> {
        if (!::modelConfig.isInitialized || modelConfig != frameRequest.modelConfig) {
            modelConfig = frameRequest.modelConfig
            setTfliteModel(context)
        }

        return frameRequest.map { run(frameRequest.obj) }
    }

    // TODO Could possibly eliminate copying by exposing buffers? But not "thread-safe"...
    fun run(inputArray: ByteArray): ByteArray {
        if (modelConfig.layer == "server")
            return inputArray.clone()

        inputBuffer.rewind()
        inputBuffer.put(inputArray)

        // TODO flip? read/write buffer modes...
        // inputBuffer.rewind() // TODO needed?
        outputBuffer.rewind()
        tflite!!.run(inputBuffer, outputBuffer)

        val outputArray = ByteArray(outputBuffer.limit())
        outputBuffer.rewind()
        outputBuffer.get(outputArray)
        return outputArray
    }

    override fun close() {
        tflite?.close()
        gpuDelegate.close()
    }

    private fun setTfliteModel(context: Context) {
        // TODO First 65 operations will run on the GPU, and the remaining 3 on the CPU.TfLiteGpuDelegate
        // Invoke: Delegate should run on the same thread where it was initialized.Node number 68
        // (TfLiteGpuDelegate) failed to invoke.
        // TODO gpuDelegate.bindGlBufferToTensor(outputTensor, outputSsboId);

        tflite?.close()

        if (modelConfig.layer == "server") {
            tfliteModel = null
            tflite = null
            inputBuffer = ByteBuffer.allocateDirect(0).order(nativeOrder())
            outputBuffer = ByteBuffer.allocateDirect(0)
            return
        }

        tfliteModel = loadModel(context, "${modelConfig.toPath()}-client.tflite")
        tflite = Interpreter(tfliteModel!!, tfliteOptions)

        val inputCapacity = tflite!!.getInputTensor(0).numBytes()
        val outputCapacity = tflite!!.getOutputTensor(0).numBytes()
        inputBuffer = ByteBuffer.allocateDirect(inputCapacity).order(nativeOrder())
        outputBuffer = ByteBuffer.allocateDirect(outputCapacity)

        // TODO byte order of outputBuffer? shouldn't this be set to ensure consistency across network?
        // outputBuffer = outputBuffer.order(nativeOrder())
        // outputBuffer = outputBuffer.order(LITTLE_ENDIAN)
    }

    @Throws(IOException::class)
    private fun loadModel(context: Context, filename: String): MappedByteBuffer {
        val fd = context.assets.openFd(filename)
        val channel = FileInputStream(fd.fileDescriptor).channel
        return channel.map(READ_ONLY, fd.startOffset, fd.declaredLength)
    }
}

data class FrameRequest<T>(
    val obj: T,
    val frameNumber: Int,
    val modelConfig: ModelConfig
) {
    inline fun <R> map(func: (T) -> R): FrameRequest<R> =
        FrameRequest(func(obj), frameNumber, modelConfig)
}