package com.sicariusnoctis.collaborativeintelligence

import com.google.common.collect.EvictingQueue
import kotlinx.serialization.Serializable
import java.time.Duration
import java.time.Instant
import java.util.*
import kotlin.collections.HashMap

class Statistics {
    private val modelStats = HashMap<ModelConfig, ModelStatistics>()

    operator fun get(modelConfig: ModelConfig): ModelStatistics =
        modelStats.getOrPut(modelConfig, { ModelStatistics() })

    operator fun get(frameNumber: Int): ModelStatistics = modelStats
        .filterValues { v -> v.containsKey(frameNumber) }
        .values
        .first()

    operator fun set(modelConfig: ModelConfig, modelStatistics: ModelStatistics) {
        modelStats[modelConfig] = modelStatistics
    }
}

class ModelStatistics {
    val isEmpty
        @Synchronized get() = lastNValidSamples.isEmpty()
    val isFirstExist: Boolean
        @Synchronized get() = allSamples.size != 0
    var framesDropped = 0
        @Synchronized get
        private set
    val framesProcessed
        @Synchronized get() = validSamples.size
    val fps
        @Synchronized get() = fps()
    val totalAverage
        @Synchronized get() =
            if (lastNValidSamples.isEmpty()) 0
            else lastNValidSamples.map { it.total!!.toMillis() }.sum() / lastNValidSamples.size
    val uploadAverage
        @Synchronized get() =
            if (lastNValidSamples.isEmpty()) 0
            else lastNValidSamples.map { it.uploadBytes!! }.sum() / lastNValidSamples.size
    val sample: Sample
        @Synchronized get() = lastNValidSamples.last()
    val samples: Map<Int, Sample>
        @Synchronized get() = validSamples
    var currentSample: Sample? = null
        @Synchronized get
        private set

    private val allSamples = LinkedHashMap<Int, Sample>()
    private val validSamples = LinkedHashMap<Int, Sample>()
    private val lastNValidSamples = EvictingQueue.create<Sample>(10)

    @Synchronized
    operator fun get(frameNumber: Int) = allSamples[frameNumber]!!

    @Synchronized
    fun containsKey(frameNumber: Int) = allSamples.containsKey(frameNumber)

    @Synchronized
    fun frameDropped() {
        framesDropped++
    }

    @Synchronized
    fun makeSample(frameNumber: Int) {
        val sample = Sample()
        currentSample = sample
        allSamples[frameNumber] = sample
    }

    @Synchronized
    fun setPreprocess(frameNumber: Int, start: Instant, end: Instant) {

        allSamples[frameNumber]!!.preprocessStart = start
        allSamples[frameNumber]!!.preprocessEnd = end
    }

    @Synchronized
    fun setInference(frameNumber: Int, start: Instant, end: Instant) {
        allSamples[frameNumber]!!.inferenceStart = start
        allSamples[frameNumber]!!.inferenceEnd = end
    }

    @Synchronized
    fun setPostencode(frameNumber: Int, start: Instant, end: Instant) {
        allSamples[frameNumber]!!.postencodeStart = start
        allSamples[frameNumber]!!.postencodeEnd = end
    }

    @Synchronized
    fun setNetworkWrite(frameNumber: Int, start: Instant, end: Instant) {
        allSamples[frameNumber]!!.networkWriteStart = start
        allSamples[frameNumber]!!.networkWriteEnd = end
    }

    @Synchronized
    fun setNetworkRead(frameNumber: Int, start: Instant, end: Instant) {
        allSamples[frameNumber]!!.networkReadStart = start
        allSamples[frameNumber]!!.networkReadEnd = end
    }

    @Synchronized
    fun setUpload(frameNumber: Int, uploadBytes: Int) {
        allSamples[frameNumber]!!.uploadBytes = uploadBytes
    }

    @Synchronized
    fun setResultResponse(frameNumber: Int, resultResponse: ResultResponse) {
        val sample = allSamples[frameNumber]!!
        sample.resultResponse = resultResponse
        validSamples[frameNumber] = sample
        lastNValidSamples.add(sample)
    }

    private fun fps() = 1000.0 * lastNValidSamples.size / Duration.between(
        lastNValidSamples.first().preprocessStart,
        lastNValidSamples.last().networkReadEnd
    ).toMillis().toDouble()
}

@Serializable
data class SerializableSample(
    val frameNumber: Int,
    val preprocess: Long,
    val clientInference: Long,
    val postencode: Long,
    val networkWrite: Long,
    val serverInference: Long,
    val networkRead: Long,
    val networkWait: Long,
    val total: Long
) {
    companion object {
        fun fromSample(frameNumber: Int, sample: Sample) = SerializableSample(
            frameNumber,
            sample.preprocess!!.toMillis(),
            sample.clientInference!!.toMillis(),
            sample.postencode!!.toMillis(),
            sample.networkWrite!!.toMillis(),
            sample.serverInference!!.toMillis(),
            sample.networkRead!!.toMillis(),
            sample.networkWait!!.toMillis(),
            sample.total!!.toMillis()
        )
    }
}

data class Sample(
    var preprocessStart: Instant? = null,
    var preprocessEnd: Instant? = null,
    var inferenceStart: Instant? = null,
    var inferenceEnd: Instant? = null,
    var postencodeStart: Instant? = null,
    var postencodeEnd: Instant? = null,
    var networkWriteStart: Instant? = null,
    var networkWriteEnd: Instant? = null,
    var networkReadStart: Instant? = null,
    var networkReadEnd: Instant? = null,
    var uploadBytes: Int? = null,
    var resultResponse: ResultResponse? = null
) {
    val preprocess get() = durationBetween(preprocessStart, preprocessEnd)
    val clientInference get() = durationBetween(inferenceStart, inferenceEnd)
    val postencode get() = durationBetween(postencodeStart, postencodeEnd)
    val networkWrite get() = durationBetween(networkWriteStart, networkWriteEnd)
    val serverInference get() = resultResponse?.let { Duration.ofMillis(it.inferenceTime) }
    val networkRead get() = durationBetween(networkReadStart, networkReadEnd)
    val networkWait get() = durationBetween(networkWriteStart, networkReadEnd)
    val total get() = durationBetween(preprocessStart, networkReadEnd)
    val frameNumber get() = resultResponse!!.frameNumber

    override fun toString() = durations()
        .zip(durationDescriptions)
        .joinToString(separator = "\n") { (duration, description) ->
            "%.3fs %s".format(duration?.toMillis()?.div(1000.0), description)
        }

    val isValid get() = uploadBytes != null && instants().all { it != null }

    private fun durations() = listOf(
        Pair(preprocessStart, preprocessEnd),
        Pair(inferenceStart, inferenceEnd),
        Pair(postencodeStart, postencodeEnd),
        Pair(networkWriteStart, networkWriteEnd),
        Pair(networkWriteEnd, networkReadEnd)
    )
        .map { durationBetween(it.first, it.second) }

    private fun instants() = listOf(
        preprocessStart, preprocessEnd,
        inferenceStart, inferenceEnd,
        postencodeStart, postencodeEnd,
        networkWriteStart, networkWriteEnd,
        networkReadStart, networkReadEnd
    )

    companion object {
        private val durationDescriptions = listOf(
            "Preprocess",
            "Inference",
            "Postencode",
            "Network Send",
            "Network Wait"
        )

        private fun durationBetween(start: Instant?, end: Instant?): Duration? {
            return if (start == null || end == null) null else Duration.between(start, end)
        }
    }
}
