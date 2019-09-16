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
    private val gpuDelegate: GpuDelegate
    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer
    private val tflite: Interpreter
    private val tfliteModel: MappedByteBuffer
    private val tfliteOptions = Interpreter.Options()

    constructor(context: Context) {
        tfliteModel = loadModel(context, "resnet34-client.tflite")

        // TODO First 65 operations will run on the GPU, and the remaining 3 on the CPU.TfLiteGpuDelegate
        // Invoke: Delegate should run on the same thread where it was initialized.Node number 68
        // (TfLiteGpuDelegate) failed to invoke.

        // TODO NNAPI faster is only better than GPU/CPU on some devices/processors with some models.
        gpuDelegate = GpuDelegate()
        tfliteOptions
            .setNumThreads(1)  // TODO 1 thread?
            // .setUseNNAPI(true)
            // .addDelegate(gpuDelegate)
        tflite = Interpreter(tfliteModel, tfliteOptions)

        inputBuffer = ByteBuffer.allocateDirect(224 * 224 * 3 * 4)
        inputBuffer.order(nativeOrder())
        outputBuffer = ByteBuffer.allocateDirect(14 * 14 * 256 * 1)
        outputBuffer.order(nativeOrder())
            // .order(LITTLE_ENDIAN)
    }

    // TODO Could possibly eliminate copying by exposing buffers? But not "thread-safe"...
    fun run(inputArray: ByteArray): ByteArray {
        inputBuffer.rewind()
        inputBuffer.put(inputArray)

        // TODO flip? read/write buffer modes...
        // inputBuffer.rewind() // TODO needed?
        outputBuffer.rewind()
        tflite.run(inputBuffer, outputBuffer)

        // val outputArray = ByteArray(14 * 14 * 256 * 1)
        val outputArray = ByteArray(outputBuffer.limit())
        outputBuffer.rewind()
        outputBuffer.get(outputArray)
        return outputArray
    }

    override fun close() {
        tflite.close()
        gpuDelegate.close()
    }

    @Throws(IOException::class)
    private fun loadModel(context: Context, filename: String): MappedByteBuffer {
        val fd = context.assets.openFd(filename)
        val channel = FileInputStream(fd.fileDescriptor).channel
        return channel.map(READ_ONLY, fd.startOffset, fd.declaredLength)
    }
}