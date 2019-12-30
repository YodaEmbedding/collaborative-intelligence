package com.sicariusnoctis.collaborativeintelligence

import android.os.Environment.getExternalStorageDirectory
import kotlinx.serialization.Serializable
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder.nativeOrder
import java.nio.file.Paths

class Inference : Closeable {
    private val TAG = Inference::class.qualifiedName

    lateinit var modelConfig: ModelConfig; private set
    private var inputBuffer: ByteBuffer? = null
    private var outputBuffer: ByteBuffer? = null
    private var gpuDelegate: GpuDelegate? = null
    private var tflite: Interpreter? = null

    fun run(frameRequest: FrameRequest<ByteArray>): FrameRequest<ByteArray> {
        if (!::modelConfig.isInitialized || modelConfig != frameRequest.info.modelConfig) {
            throw Exception("Current config $modelConfig differs from requested config ${frameRequest.info.modelConfig}")
        }

        return frameRequest.map { run(frameRequest.obj) }
    }

    private fun run(inputArray: ByteArray): ByteArray {
        if (modelConfig.layer == "server")
            return inputArray.clone()

        inputBuffer!!.rewind()
        inputBuffer!!.put(inputArray)

        outputBuffer!!.rewind()
        tflite!!.run(inputBuffer!!, outputBuffer!!)

        val outputArray = ByteArray(outputBuffer!!.limit())
        outputBuffer!!.rewind()
        outputBuffer!!.get(outputArray)
        return outputArray
    }

    override fun close() {
        tflite?.close()
        tflite = null
        gpuDelegate?.close()
        gpuDelegate = null
        inputBuffer = null
        outputBuffer = null
    }

    fun switchModel(modelConfig: ModelConfig) {
        close()

        this.modelConfig = modelConfig

        if (modelConfig.layer == "server")
            return

        // TODO gpuDelegate.bindGlBufferToTensor(outputTensor, outputSsboId);
        gpuDelegate = GpuDelegate()

        // TODO NNAPI only faster than GPU/CPU on some devices/processors with certain models. Add UI switch?
        val tfliteOptions = Interpreter.Options()
            .setNumThreads(1)
            // .setUseNNAPI(true)
            // .setAllowFp16PrecisionForFp32(true)
            .addDelegate(gpuDelegate)

        val filename = "${modelConfig.toPath()}-client.tflite"
        val file = File(parentDirectory(), filename)
        tflite = Interpreter(file, tfliteOptions)

        val inputCapacity = tflite!!.getInputTensor(0).numBytes()
        val outputCapacity = tflite!!.getOutputTensor(0).numBytes()
        inputBuffer = ByteBuffer.allocateDirect(inputCapacity).order(nativeOrder())
        outputBuffer = ByteBuffer.allocateDirect(outputCapacity)
    }

    private fun parentDirectory(): String {
        val folderRoot = "collaborative-intelligence"
        val sdcard = getExternalStorageDirectory().toString()
        return Paths.get(sdcard, folderRoot).toString()
    }
}

data class FrameRequest<T>(
    val obj: T,
    val info: FrameRequestInfo
) {
    inline fun <R> map(func: (T) -> R): FrameRequest<R> = FrameRequest(func(obj), info)
}

@Serializable
data class FrameRequestInfo(
    val frameNumber: Int,
    val modelConfig: ModelConfig
)