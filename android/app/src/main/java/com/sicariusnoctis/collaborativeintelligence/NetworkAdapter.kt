package com.sicariusnoctis.collaborativeintelligence

import android.util.Log
import kotlinx.serialization.PolymorphicSerializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.UnstableDefault
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.modules.SerializersModule
import java.io.*
import java.net.Socket
import java.time.Duration
import java.time.Instant
import java.util.*

class NetworkAdapter {
    private val TAG = NetworkAdapter::class.qualifiedName

    private val frameHeaderSize = 6 + 4 + 4
    private var inputStream: BufferedReader? = null
    private var outputStream: DataOutputStream? = null
    // private var outputStream: OutputStream? = null
    private var socket: Socket? = null
    private var prevModelConfig: ModelConfig? = null
    private val jsonSerializer = Json(
        context = SerializersModule {
            polymorphic<Response> {
                ConfirmationResponse::class with ConfirmationResponse.serializer()
                PingResponse::class with PingResponse.serializer()
                ResultResponse::class with ResultResponse.serializer()
            }
        }
    )

    lateinit var uploadStats: UploadStats
    val timeUntilWriteAvailable get() = uploadStats.timeUntilAvailable

    fun connect() {
        val HOSTNAME = HOSTNAME
        socket = Socket(HOSTNAME, 5678)
        // Ensure write+flush turns into a packet by disabling Nagle
        socket!!.tcpNoDelay = true

        // TODO BufferedInputStream for byte data?
        // https://stackoverflow.com/questions/15538509/dealing-with-end-of-file-using-bufferedreader-read
        inputStream = BufferedReader(InputStreamReader(socket!!.inputStream))
        outputStream = DataOutputStream(socket!!.outputStream)
        // outputStream = BufferedOutputStream(socket!!.outputStream)
    }

    fun close() {
        socket?.close()
        socket = null
        inputStream = null
        outputStream = null
    }

    fun readResponse(): Response? {
        while (true) {
            val response = readResponseInner() ?: return null
            if (response is ConfirmationResponse)
                handleConfirmation(response)
            else
                return response
        }
    }

    private fun readResponseInner(): Response? {
        val msg = inputStream!!.readLine() ?: return null
        Log.i(TAG, "Receive: $msg")
        return jsonSerializer.parse(PolymorphicSerializer(Response::class), msg) as Response
    }

    private fun handleConfirmation(response: ConfirmationResponse) {
        uploadStats.confirmBytes(response.frameNumber, frameHeaderSize + response.numBytes)
        Log.i(TAG, "Confirmation: $response\nRemaining bytes: ${uploadStats.remainingBytes}")
    }

    private fun writeData(frameNumber: Int, data: ByteArray) {
        uploadStats.sendBytes(frameNumber, frameHeaderSize + data.size)
        with(outputStream!!) {
            // writer().write("frame\n")
            // writer().write(frameNumber)
            // writer().write(data.size)
            writeBytes("frame\n")
            writeInt(frameNumber)
            writeInt(data.size)
            write(data)
            flush()
        }
    }

    @UseExperimental(UnstableDefault::class)
    private fun writeJson(jsonString: String) {
        // uploadStats.sendBytes(frameNumber, 6 + jsonString.length)
        with(outputStream!!) {
            writeBytes("json\n")
            writeBytes("$jsonString\n")
            // writer().write("json\n")
            // writer().write("$jsonString\n")
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
        // uploadStats.sendBytes(frameNumber, 6 + jsonString.length)
        writeJson(jsonString)
        // TODO wait until all traffic is flushed?
        switchModel()
    }

    fun writePingRequest(pingRequest: PingRequest) {
        with(outputStream!!) {
            writeBytes("ping\n")
            writeInt(pingRequest.id)
            flush()
        }
    }

    @UseExperimental(UnstableDefault::class)
    fun writeSample(frameRequest: FrameRequest<Sample>) {
        val sample = SerializableSample.fromSample(frameRequest.info.frameNumber, frameRequest.obj)
        val jsonString = Json.stringify(SerializableSample.serializer(), sample)
        // uploadStats.sendBytes(frameNumber, 6 + jsonString.length)
        writeJson(jsonString)
    }

    private fun switchModel() {
        uploadStats = UploadStats()
        // TODO maybe send ping here?
    }
}

class UploadStats {
    private var goalBytes: Long = 0
    private var confirmedBytes: Long = 0
    private val samples = LinkedHashMap<Int, Sample>()
    private lateinit var lastSentSample: Sample
    private lateinit var lastConfirmedSample: Sample

    // TODO exponential geometric average may be more accurate
    // TODO though confirmation may not be received after long time, bytesPerSecond remains same...
    // TODO subtract ping? or is that only for first frame? nah for all frames
    val bytesPerSecond
        @Synchronized get() = 1000.0 * lastConfirmedSample.bytes / Duration.between(
            lastConfirmedSample.timeStart,
            lastConfirmedSample.timeEnd
        ).toMillis()
    val remainingBytes @Synchronized get() = goalBytes - confirmedBytes
    val timeUntilAvailable: Duration
        @Synchronized get() {
            val elapsed = Duration.between(lastSentSample.timeStart, Instant.now())
            val expected = Duration.ofMillis((1000 * remainingBytes / bytesPerSecond).toLong())
            return expected - elapsed
        }

    @Synchronized
    fun sendBytes(frameNumber: Int, count: Int) {
        goalBytes += count
        samples[frameNumber] = Sample(Instant.now(), null, count.toLong())
        lastSentSample = samples[frameNumber]!!
    }

    @Synchronized
    fun confirmBytes(frameNumber: Int, count: Int) {
        confirmedBytes += count
        val sample = samples[frameNumber]!!
        sample.timeEnd = Instant.now()
        if (sample.bytes != count.toLong())
            throw Exception("Confirmed bytes ($count) less than were sent (${sample.bytes})")
        lastConfirmedSample = sample
    }

    private data class Sample(var timeStart: Instant, var timeEnd: Instant?, var bytes: Long)
}

@Serializable
data class Prediction(val name: String, val description: String, val score: Float)

@Serializable
data class PingRequest(val id: Int)

@Serializable
open class Response

@Serializable
@SerialName("confirmation")
data class ConfirmationResponse(
    val frameNumber: Int,
    val numBytes: Int
) : Response()

@Serializable
@SerialName("ping")
data class PingResponse(
    val id: Int
) : Response()

@Serializable
@SerialName("result")
data class ResultResponse(
    val frameNumber: Int,
    // val readTime: Long,
    // val feedTime: Long,
    val inferenceTime: Long,
    val predictions: List<Prediction>
) : Response()

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