package com.sicariusnoctis.collaborativeintelligence

import android.util.Log
import kotlinx.serialization.Serializable
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

    fun readData(): ResultResponse? {
        val msg = inputStream!!.readLine() ?: return null
        Log.i(TAG, "Received: $msg")
        return Json.parse(ResultResponse.serializer(), msg)
    }

    fun writeData(frameNumber: Int, data: ByteArray) {
        val eol = ByteArray(1) { 0 }
        Log.i(TAG, "Tensor message size: ${data.size}")

        with(outputStream!!) {
            writeBytes("frame\n")
            // write(eol)
            writeInt(frameNumber)
            // write(eol)
            writeInt(data.size)
            // write(eol)
            write(data)
            // write(eol)
            flush()
        }
    }

    fun writeFrameRequest(frameRequest: FrameRequest<ByteArray>) {
        if (prevModelConfig != frameRequest.modelConfig) {
            writeModelConfig(frameRequest.modelConfig)
            prevModelConfig = frameRequest.modelConfig
        }
        writeData(frameRequest.frameNumber, frameRequest.obj)
    }

    fun writeModelConfig(modelConfig: ModelConfig) {
        val jsonString = Json.stringify(ModelConfig.serializer(), modelConfig)
        with(outputStream!!) {
            writeBytes("json\n")
            writeBytes("$jsonString\n")
            flush()
        }
    }

    // fun writeJson(jsonObject: JsonObject) {
    //     // val x = jsonObject["idk"]?.jsonObject?.get("e")
    //     with(outputStream!!) {
    //         // TODO is this... secure? What about quoted strings?
    //         writeBytes("$jsonObject\n")
    //     }
    // }
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

    // TODO shouldn't args be ", " separated... not "," separated? Hmmm...
    private fun dictString(name: String, args: JsonObject?): String =
        if (args == null) {
            name
        } else {
            "$name(${args.jsonObject.map { (k, v) -> "$k=$v" }.joinToString(", ")})"
        }
}