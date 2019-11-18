package com.sicariusnoctis.collaborativeintelligence

import android.net.TrafficStats
import android.util.Log
import com.facebook.network.connectionclass.ConnectionClassManager
import io.reactivex.Observable
import io.reactivex.disposables.CompositeDisposable
import io.reactivex.internal.schedulers.IoScheduler
import kotlinx.serialization.PolymorphicSerializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.UnstableDefault
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.modules.SerializersModule
import java.io.BufferedReader
import java.io.DataOutputStream
import java.io.InputStreamReader
import java.net.Socket
import java.time.Duration
import java.time.Instant
import java.util.concurrent.TimeUnit
import kotlin.reflect.jvm.javaConstructor

class NetworkAdapter {
    private val TAG = NetworkAdapter::class.qualifiedName

    private var inputStream: BufferedReader? = null
    private var outputStream: DataOutputStream? = null
    private var socket: Socket? = null
    private var prevModelConfig: ModelConfig? = null
    lateinit var uploadStats: UploadStats
    private val jsonSerializer = Json(
        context = SerializersModule {
            polymorphic<Response> {
                ResultResponse::class with ResultResponse.serializer()
                ConfirmationResponse::class with ConfirmationResponse.serializer()
            }
        }
    )

    val uploadBytesPerSecond get() = uploadStats.bytesPerSecond
    val uploadRemainingBytes get() = uploadStats.remainingBytes

    fun connect() {
        val HOSTNAME = HOSTNAME
        socket = Socket(HOSTNAME, 5678)
        // Ensure write+flush turns into a packet by disabling Nagle
        socket!!.tcpNoDelay = true

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

    private fun readResponse(): Response? {
        val msg = inputStream!!.readLine() ?: return null
        Log.i(TAG, "Receive: $msg")
        return jsonSerializer.parse(PolymorphicSerializer(Response::class), msg) as Response
    }

    @UseExperimental(UnstableDefault::class)
    fun readResultResponse(): ResultResponse? {
        var response: Response
        // TODO
        do {
            response = readResponse() ?: return null
        }
        while (response !is ResultResponse)
        return response
    }

    private fun writeData(frameNumber: Int, data: ByteArray) {
        with(outputStream!!) {
            writeBytes("frame\n")
            writeInt(frameNumber)
            writeInt(data.size)
            write(data)
            flush()
        }
        uploadStats.addBytes(6 + 4 + 4 + data.size)
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
        // TODO wait until all traffic is flushed?
        switchModel()
    }

    private fun switchModel() {
        if (::uploadStats.isInitialized)
            uploadStats.dispose()
        uploadStats = UploadStats()
    }
}

class UploadStats {
    private val connectionClassManager: ConnectionClassManager
    private val compositeDisposable = CompositeDisposable()
    private var prevSampleTime: Instant? = null
    private var prevSampleBytes: Long = -1
    private var startBytes: Long = -1
    var goalBytes: Long = 0; private set

    val bytesPerSecond @Synchronized get() = 1000 * connectionClassManager.downloadKBitsPerSecond / 8
    val uploadedBytes get() = bytes() - startBytes
    val remainingBytes get() = goalBytes - uploadedBytes

    init {
        val constructor = ConnectionClassManager::class.constructors.first()
        constructor.javaConstructor!!.isAccessible = true
        connectionClassManager = constructor.javaConstructor!!.newInstance()
        start()
    }

    fun dispose() = compositeDisposable.dispose()

    fun addBytes(count: Int) {
        goalBytes += count
    }

    private fun start() {
        val disposable = Observable
            .interval(0, 50, TimeUnit.MILLISECONDS)
            .doOnNext {
                val now = Instant.now()
                val bytes = bytes()

                if (startBytes == -1L) {
                    startBytes = bytes
                }
                else {
                    val byteDiff = bytes - prevSampleBytes
                    val timeDiff = Duration.between(prevSampleTime, now)
                    synchronized(this) {
                        connectionClassManager.addBandwidth(byteDiff, timeDiff.toMillis())
                    }
                }

                prevSampleTime = now
                prevSampleBytes = bytes
            }
            .subscribeOn(IoScheduler())
            .subscribe()
        compositeDisposable.add(disposable)
    }

    private fun bytes() = TrafficStats.getTotalTxBytes()
}

@Serializable
data class Prediction(val name: String, val description: String, val score: Float)

@Serializable
open class Response

@Serializable
@SerialName("confirmation")
data class ConfirmationResponse(
    val frameNumber: Int
    // val status: String
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