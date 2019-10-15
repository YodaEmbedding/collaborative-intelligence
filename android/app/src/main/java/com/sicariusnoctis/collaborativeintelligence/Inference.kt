package com.sicariusnoctis.collaborativeintelligence

import android.content.Context
import kotlinx.serialization.UnstableDefault
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonObjectSerializer
import kotlinx.serialization.json.content
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
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputBuffer: ByteBuffer
    private lateinit var tflite: Interpreter
    private lateinit var tfliteModel: MappedByteBuffer
    private lateinit var modelConfig: ModelConfig
    private val tfliteOptions = Interpreter.Options()
    private val modelConfigMap: Map<String, List<ModelConfig>>
    private var nextModelConfigField: ModelConfig? = null
    var nextModelConfig: ModelConfig
        @Synchronized get() = nextModelConfigField ?: modelConfig
        @Synchronized set(value) {
            nextModelConfigField = value
        }

    constructor(context: Context) {
        modelConfigMap = loadConfig(context, "models.json").map { (k, v) ->
            k to v.jsonArray.map { x ->
                jsonToModelConfig(x.jsonObject, k)
            }
        }.toMap()

        // TODO NNAPI faster is only better than GPU/CPU on some devices/processors with some models.
        gpuDelegate = GpuDelegate()
        tfliteOptions
            .setNumThreads(1)  // TODO 1 thread?
            // .setUseNNAPI(true)
            .addDelegate(gpuDelegate)

        // TODO DEBUG
        nextModelConfig = modelConfigMap["resnet34"]!!.first {
            it.encoder == "UniformQuantizationU8Encoder"
        }
    }

    // TODO Could possibly eliminate copying by exposing buffers? But not "thread-safe"...
    fun run(inputArray: ByteArray): ByteArray {
        inputBuffer.rewind()
        inputBuffer.put(inputArray)

        // TODO flip? read/write buffer modes...
        // inputBuffer.rewind() // TODO needed?
        outputBuffer.rewind()
        tflite.run(inputBuffer, outputBuffer)

        val outputArray = ByteArray(outputBuffer.limit())
        outputBuffer.rewind()
        outputBuffer.get(outputArray)
        return outputArray
    }

    fun run(context: Context, frameRequest: FrameRequest<ByteArray>): FrameRequest<ByteArray> {
        if (!::modelConfig.isInitialized || modelConfig != frameRequest.modelConfig) {
            modelConfig = frameRequest.modelConfig
            if (::tflite.isInitialized)
                tflite.close()
            setTfliteModel(context)
        }

        return frameRequest.map { run(frameRequest.obj) }
    }

    override fun close() {
        tflite.close()
        gpuDelegate.close()
    }

    private fun setTfliteModel(context: Context) {
        // TODO First 65 operations will run on the GPU, and the remaining 3 on the CPU.TfLiteGpuDelegate
        // Invoke: Delegate should run on the same thread where it was initialized.Node number 68
        // (TfLiteGpuDelegate) failed to invoke.
        // TODO gpuDelegate.bindGlBufferToTensor(outputTensor, outputSsboId);

        tfliteModel = loadModel(context, "${modelConfig.toPath()}-client.tflite")
        tflite = Interpreter(tfliteModel, tfliteOptions)

        val inputCapacity = tflite.getInputTensor(0).numBytes()
        val outputCapacity = tflite.getOutputTensor(0).numBytes()
        inputBuffer = ByteBuffer.allocateDirect(inputCapacity).order(nativeOrder())
        outputBuffer = ByteBuffer.allocateDirect(outputCapacity)

        // TODO byte order of outputBuffer? shouldn't this be set to ensure consistency across network?
        // outputBuffer = outputBuffer.order(nativeOrder())
        // outputBuffer = outputBuffer.order(LITTLE_ENDIAN)
    }

    // TODO Move to separate class?
    private fun jsonToModelConfig(jsonObject: JsonObject, model: String? = null) = ModelConfig(
        model = model ?: jsonObject["model"]!!.content,
        layer = jsonObject["layer"]!!.content,
        encoder = jsonObject["encoder"]!!.content,
        decoder = jsonObject["decoder"]!!.content,
        encoder_args = jsonObject["encoder_args"]?.jsonObject,
        decoder_args = jsonObject["decoder_args"]?.jsonObject
    )

    @UseExperimental(UnstableDefault::class)
    private fun loadConfig(context: Context, filename: String): JsonObject {
        val inputStream = context.assets.open(filename)
        val jsonString = inputStream.bufferedReader().use { it.readText() }
        return Json.parse(JsonObjectSerializer, jsonString)
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
