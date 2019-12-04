package com.sicariusnoctis.collaborativeintelligence

import com.google.common.collect.EvictingQueue
import kotlinx.serialization.Serializable
import java.time.Duration
import java.time.Instant
import java.util.*
import kotlin.collections.HashMap

// TODO reduce spaghetti...

class Statistics {
    private val modelStats = HashMap<ModelConfig, ModelStatistics>()

    operator fun get(modelConfig: ModelConfig): ModelStatistics =
        modelStats.getOrPut(modelConfig, { ModelStatistics() })

    operator fun get(frameNumber: Int): ModelStatistics = modelStats
        .filterValues { v -> v.containsKey(frameNumber) }
        .values
        .first()
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
    val sample: Sample
        @Synchronized get() = lastNValidSamples.last()
    val samples: Map<Int, Sample>
        @Synchronized get() = validSamples
    // val samples: List<Sample>
    //     @Synchronized get() = validSamples.values.toList()

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
    fun setPreprocess(frameNumber: Int, start: Instant, end: Instant) {
        allSamples[frameNumber] = Sample()
        allSamples[frameNumber]!!.preprocessStart = start
        allSamples[frameNumber]!!.preprocessEnd = end
    }

    @Synchronized
    fun setInference(frameNumber: Int, start: Instant, end: Instant) {
        allSamples[frameNumber]!!.inferenceStart = start
        allSamples[frameNumber]!!.inferenceEnd = end
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
    val networkWrite: Long,
    val serverInference: Long,
    val networkRead: Long,
    val networkWait: Long,
    val total: Long
) {
    companion object {
        fun fromSample(frameNumber: Int, sample: Sample) = SerializableSample(
            frameNumber,
            sample.preprocess.toMillis(),
            sample.clientInference.toMillis(),
            sample.networkWrite.toMillis(),
            sample.serverInference.toMillis(),
            sample.networkRead.toMillis(),
            sample.networkWait.toMillis(),
            sample.total.toMillis()
        )
    }
}

data class Sample(
    var preprocessStart: Instant? = null,
    var preprocessEnd: Instant? = null,
    var inferenceStart: Instant? = null,
    var inferenceEnd: Instant? = null,
    var networkWriteStart: Instant? = null,
    var networkWriteEnd: Instant? = null,
    var networkReadStart: Instant? = null,
    var networkReadEnd: Instant? = null,
    var uploadBytes: Int? = null,
    var resultResponse: ResultResponse? = null
) {
    val preprocess: Duration
        get() = Duration.between(preprocessStart, preprocessEnd)
    val clientInference: Duration
        get() = Duration.between(inferenceStart, inferenceEnd)
    // val encoding: Duration
    //     get() = Duration.between(encodingStart, encodingEnd)
    val networkWrite: Duration
        get() = Duration.between(networkWriteStart, networkWriteEnd)
    val serverInference: Duration
        get() = Duration.ofMillis(resultResponse!!.inferenceTime)
    val networkRead: Duration
        get() = Duration.between(networkReadStart, networkReadEnd)
    val networkWait: Duration
        get() = Duration.between(networkWriteStart, networkReadEnd)
    val total: Duration
        get() = Duration.between(preprocessStart, networkReadEnd)
    val frameNumber: Int
        get() = resultResponse!!.frameNumber

    override fun toString() = durations()
        .zip(durationDescriptions)
        .joinToString(separator = "\n") { (duration, description) ->
            "%.3fs %s".format(duration.toMillis() / 1000.0, description)
        }

    val isValid get() = uploadBytes != null && instants().all { it != null }

    private fun durations() = listOf(
        Pair(preprocessStart, preprocessEnd),
        Pair(inferenceStart, inferenceEnd),
        Pair(networkWriteStart, networkWriteEnd),
        Pair(networkWriteEnd, networkReadEnd)
    )
        .map { Duration.between(it.first, it.second) }

    private fun instants() = listOf(
        preprocessStart, preprocessEnd,
        inferenceStart, inferenceEnd,
        networkWriteStart, networkWriteEnd,
        networkReadStart, networkReadEnd
    )

    companion object {
        private val durationDescriptions = listOf(
            "Preprocess",
            "Inference",
            "Network Send",
            "Network Wait"
        )
    }
}
