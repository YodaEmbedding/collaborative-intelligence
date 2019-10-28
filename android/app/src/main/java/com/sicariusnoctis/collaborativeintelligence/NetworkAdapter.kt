package com.sicariusnoctis.collaborativeintelligence

import android.util.Log
import kotlinx.serialization.Serializable
import kotlinx.serialization.UnstableDefault
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import java.io.BufferedReader
import java.io.DataOutputStream
import java.io.InputStreamReader
import java.net.Socket

class NetworkAdapter {
    private val TAG = NetworkAdapter::class.qualifiedName

    private var inputStream: BufferedReader? = null
    private var outputStream: DataOutputStream? = null
    private var socket: Socket? = null
    private var prevModelConfig: ModelConfig? = null

    fun connect() {
        val HOSTNAME = HOSTNAME
        socket = Socket(HOSTNAME, 5678)

        // TODO BufferedInputStream for byte data?
        // https://stackoverflow.com/questions/15538509/dealing-with-end-of-file-using-bufferedreader-read
        inputStream = BufferedReader(InputStreamReader(socket!!.inputStream))
        outputStream = DataOutputStream(socket!!.outputStream)
    }

    fun close() {
        socket?.close()
        socket = null
        inputStream = null
        outputStream = null
    }

    @UseExperimental(UnstableDefault::class)
    fun readData(): ResultResponse? {
        val msg = inputStream!!.readLine() ?: return null
        return Json.parse(ResultResponse.serializer(), msg)
    }

    fun writeData(frameNumber: Int, data: ByteArray) {
        with(outputStream!!) {
            writeBytes("frame\n")
            writeInt(frameNumber)
            writeInt(data.size)
            write(data)
            flush()
        }
    }

    fun writeFrameRequest(frameRequest: FrameRequest<ByteArray>) {
        Log.i(TAG, "Request: ${frameRequest.info.frameNumber}, ${frameRequest.info.modelConfig}")
        if (prevModelConfig != frameRequest.info.modelConfig) {
            writeModelConfig(frameRequest.info.modelConfig)
            prevModelConfig = frameRequest.info.modelConfig
        }
        writeData(frameRequest.info.frameNumber, frameRequest.obj)
    }

    @UseExperimental(UnstableDefault::class)
    fun writeModelConfig(modelConfig: ModelConfig) {
        val jsonString = Json.stringify(ModelConfig.serializer(), modelConfig)
        with(outputStream!!) {
            writeBytes("json\n")
            writeBytes("$jsonString\n")
            flush()
        }
    }
}

@Serializable
data class Prediction(val name: String, val description: String, val score: Float)

@Serializable
data class ResultResponse(
    val frameNumber: Int,
    // val readTime: Long,
    // val feedTime: Long,
    val inferenceTime: Long,
    val predictions: List<Prediction>
)

@Serializable
data class ModelConfig(
    val model: String,
    val layer: String,
    val encoder: String,
    val decoder: String,
    val encoder_args: JsonObject?,
    val decoder_args: JsonObject?
) {
    fun toPath(): String = listOf(
        model,
        layer,
        dictString(encoder, encoder_args),
        dictString(decoder, encoder_args)
    ).joinToString("&")

    private fun dictString(name: String, args: JsonObject?): String =
        if (args == null) {
            name
        } else {
            "$name(${args.jsonObject.map { (k, v) -> "$k=$v" }.joinToString(", ")})"
        }
}