package com.sicariusnoctis.collaborativeintelligence

import android.util.Log
import java.time.Duration
import java.time.Instant
import java.util.*

// TODO ensure that all are set somehow...

data class Sample(
    // val result: String,
    // val isDropped: Boolean,
    // val startTime: Instant,
    // val preprocessDuration: Duration,
    // val inferenceDuration: Duration,
    // val networkWriteDuration: Duration,
    // val networkWaitDuration: Duration,
    // val networkReadDuration: Duration)
    var preprocessStart: Instant,
    var preprocessEnd: Instant,
    var inferenceStart: Instant,
    var inferenceEnd: Instant,
    var networkWriteStart: Instant,
    var networkWriteEnd: Instant,
    var networkReadStart: Instant,
    var networkReadEnd: Instant
) {
    constructor() : this(
        Instant.MIN,
        Instant.MIN,
        Instant.MIN,
        Instant.MIN,
        Instant.MIN,
        Instant.MIN,
        Instant.MIN,
        Instant.MIN
    )

    override fun toString(): String = durations()
        .zip(durationDescriptions)
        .joinToString(separator = "\n") { (duration, description) ->
            "%.3fs %s".format(duration.toMillis() / 1000.0, description)
        }

    fun durations(): List<Duration> = listOf(
        Pair(preprocessStart, preprocessEnd),
        Pair(inferenceStart, inferenceEnd),
        Pair(networkWriteStart, networkWriteEnd),
        Pair(networkWriteEnd, networkReadStart),
        Pair(networkReadStart, networkReadEnd)
    )
        .map { Duration.between(it.first, it.second) }

    companion object {
        private val durationDescriptions = listOf(
            "Preprocess",
            "Inference",
            "Network Send",
            "Network Wait",
            "Network Receive"
        )
    }
}

class Statistics {

    var dropped = 0
        private set

    private val samples = Vector<Sample>()

    fun display(): String {
        // TODO last finished sample? or averages? or other things?
        return "Previous frame:\n${lastValidSample()}\n\nDropped: $dropped\nFPS: ${fps()}"
    }

    fun fps(): Double {
        val duration = Duration.between(
            samples.firstElement().preprocessStart,
            lastValidSample().networkReadEnd
        )
        Log.e("Statistics", samples.firstElement().toString())
        Log.e("Statistics", lastValidSample().toString())
        return samples.size.toDouble() / duration.toMillis().toDouble() * 1000.0
    }

    fun frameDropped() {
        dropped++
    }

    fun setPreprocess(frameNum: Int, start: Instant, end: Instant) {
        resize(frameNum + 1)
        samples[frameNum].preprocessStart = start
        samples[frameNum].preprocessEnd = end
    }

    fun setInference(frameNum: Int, start: Instant, end: Instant) {
        resize(frameNum + 1)
        samples[frameNum].inferenceStart = start
        samples[frameNum].inferenceEnd = end
    }

    fun setNetworkWrite(frameNum: Int, start: Instant, end: Instant) {
        resize(frameNum + 1)
        samples[frameNum].networkWriteStart = start
        samples[frameNum].networkWriteEnd = end
    }

    fun setNetworkRead(frameNum: Int, start: Instant, end: Instant) {
        resize(frameNum + 1)
        samples[frameNum].networkReadStart = start
        samples[frameNum].networkReadEnd = end
    }

    // TODO bad design... should be counting numValidSamples++?
    private fun lastValidSample(): Sample = samples.findLast { it.networkReadEnd > Instant.MIN }!!

    private fun resize(size: Int) {
        val n = size - samples.size
        if (n < 0) return
        samples.addAll(List(n) { Sample() })
    }
}
