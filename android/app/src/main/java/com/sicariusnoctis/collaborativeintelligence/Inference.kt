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
    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer
    private val tflite: Interpreter
    private val tfliteModel: MappedByteBuffer
    private val tfliteOptions = Interpreter.Options()
    private val modelConfigMap: Map<String, List<ModelConfig>>
    private val modelConfig: ModelConfig

    constructor(context: Context) {
        modelConfigMap = loadConfig(context, "models.json").map { (k, v) ->
            k to v.jsonArray.map { x ->
                jsonToModelConfig(x.jsonObject, k)
            }
        }.toMap()

        modelConfig = modelConfigMap["resnet34"]!!.first {
            it.encoder == "UniformQuantizationU8Encoder"
        }

        // TODO this kind of stuff can be made into a test...!
        tfliteModel = loadModel(context, "${modelConfig.toPath()}-client.tflite")

        // TODO First 65 operations will run on the GPU, and the remaining 3 on the CPU.TfLiteGpuDelegate
        // Invoke: Delegate should run on the same thread where it was initialized.Node number 68
        // (TfLiteGpuDelegate) failed to invoke.

        // TODO NNAPI faster is only better than GPU/CPU on some devices/processors with some models.
        gpuDelegate = GpuDelegate()
        tfliteOptions
            .setNumThreads(1)  // TODO 1 thread?
            // .setUseNNAPI(true)
            .addDelegate(gpuDelegate)
        tflite = Interpreter(tfliteModel, tfliteOptions)

        // TODO gpuDelegate.bindGlBufferToTensor(outputTensor, outputSsboId);

        val inputCapacity = tflite.getInputTensor(0).numBytes()
        val outputCapacity = tflite.getOutputTensor(0).numBytes()

        inputBuffer = ByteBuffer.allocateDirect(inputCapacity).order(nativeOrder())
        outputBuffer = ByteBuffer.allocateDirect(outputCapacity)

        // TODO byte order of outputBuffer? shouldn't this be set to ensure consistency across network?
        // outputBuffer = outputBuffer.order(nativeOrder())
        // outputBuffer = outputBuffer.order(LITTLE_ENDIAN)
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