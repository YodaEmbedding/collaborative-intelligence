package com.sicariusnoctis.collaborativeintelligence

import android.util.Log
import kotlinx.serialization.PolymorphicSerializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.UnstableDefault
import kotlinx.serialization.json.*
import kotlinx.serialization.modules.SerializersModule
import java.io.*
import java.net.Socket
import java.time.Duration
import java.time.Instant
import java.util.*
import kotlin.math.max
import kotlin.math.min

class NetworkAdapter {
    private val TAG = NetworkAdapter::class.qualifiedName

    private val frameHeaderSize = 6 + 4 + 4
    private var inputStream: BufferedReader? = null
    private var outputStream: DataOutputStream? = null
    // private var outputStream: OutputStream? = null
    private var socket: Socket? = null
    private var modelConfig: ModelConfig? = null
    private val jsonSerializer = Json(
        context = SerializersModule {
            polymorphic<Response> {
                ConfirmationResponse::class with ConfirmationResponse.serializer()
                ModelReadyResponse::class with ModelReadyResponse.serializer()
                PingResponse::class with PingResponse.serializer()
                ResultResponse::class with ResultResponse.serializer()
            }
        }
    )
    private val rateLimiter = RateLimiter()

    lateinit var uploadStats: UploadStats
    val timeUntilWriteAvailable get() = uploadStats.timeUntilAvailable
    var uploadLimitRate
        get() = rateLimiter.rate?.div(1024)
        set(value) {
            rateLimiter.rate = value?.times(1024)
        }

    @UnstableDefault
    fun connect() {
        val HOSTNAMES = listOf(HOSTNAME)
        val PORT = 5678

        val networkConfig = loadJsonFromDefaultFolder("network.json")
        val hostnameOverride = listOfNotNull(networkConfig?.get("hostname")?.content)
        val portOverride = networkConfig?.get("port")?.intOrNull
        tryConnect(hostnameOverride + HOSTNAMES, portOverride ?: PORT)

        // Ensure write+flush turns into a packet by disabling Nagle
        socket!!.tcpNoDelay = true

        // TODO BufferedInputStream for byte data?
        // https://stackoverflow.com/questions/15538509/dealing-with-end-of-file-using-bufferedreader-read
        inputStream = BufferedReader(InputStreamReader(socket!!.inputStream))
        outputStream = DataOutputStream(socket!!.outputStream)
        // outputStream = BufferedOutputStream(socket!!.outputStream)
    }

    private fun tryConnect(hostnames: List<String>, port: Int) {
        for (hostname in hostnames) {
            try {
                Log.i(TAG, "Try to connect to $hostname:$port")
                socket = Socket(hostname, port)
                Log.i(TAG, "Connection established on $hostname:$port")
                return
            } catch (e: IOException) {
            }
        }
        throw Exception("Could not establish connection with any of the hosts")
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
            rateLimiter.run(data, ::write)
            flush()
        }
    }

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

    // TODO properly extract model config switching functionality...? switchModel() should call writeModelConfig...
    // This is probably why there's a bug... writeModelConfig is not reliably called!

    @UnstableDefault
    fun writeFrameRequest(frameRequest: FrameRequest<ByteArray>) {
        Log.i(TAG, "Request: ${frameRequest.info.frameNumber}, ${frameRequest.info.modelConfig}")
        if (modelConfig != frameRequest.info.modelConfig) {
            throw Exception("Frame request model config does not match last sent model config")
        }
        writeData(frameRequest.info.frameNumber, frameRequest.obj)
    }

    @UnstableDefault
    fun writeModelConfig(modelConfig: ModelConfig) {
        val jsonString = Json.stringify(ModelConfig.serializer(), modelConfig)
        // uploadStats.sendBytes(frameNumber, 6 + jsonString.length)
        writeJson(jsonString)
        // TODO wait until all traffic is flushed?
        switchModel()
        this.modelConfig = modelConfig
    }

    fun writePingRequest(pingRequest: PingRequest) {
        with(outputStream!!) {
            writeBytes("ping\n")
            writeInt(pingRequest.id)
            flush()
        }
    }

    @UnstableDefault
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

class RateLimiter {
    var rate: Long? = null
    private val timeStep = 100
    private var lastWrite: Instant? = null
    private var lastStep = 0

    fun run(data: ByteArray, write: (ByteArray) -> Unit) {
        if (rate == null) {
            runStep(data, write, null)
            return
        }

        var sent = 0

        while (sent < data.size) {
            val rate = this.rate
            val remain = data.size - sent
            val step = if (rate == null) remain else min(remain, rate.toInt() * timeStep / 1000)
            val slice = data.sliceArray(sent until sent + step)
            runStep(slice, write, rate)
            sent += step
        }
    }

    private fun runStep(data: ByteArray, write: (ByteArray) -> Unit, rate: Long?) {
        if (rate != null && lastWrite != null) {
            val now = Instant.now()
            val dt = Duration.between(lastWrite!!, now).toMillis()
            val wait = 1000 * lastStep / rate
            Thread.sleep(max(0, wait - dt))
        }

        doWrite(data, write)
    }

    private fun doWrite(data: ByteArray, write: (ByteArray) -> Unit) {
        lastStep = data.size
        lastWrite = Instant.now()
        write(data)
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
        if (!samples.containsKey(frameNumber))
            return
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
@SerialName("ready")
data class ModelReadyResponse(
    val modelConfig: ModelConfig
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
    val encoder_args: JsonObject? = null,
    val decoder_args: JsonObject? = null
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