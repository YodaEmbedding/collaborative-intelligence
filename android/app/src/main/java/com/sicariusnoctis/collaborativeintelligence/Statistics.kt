package com.sicariusnoctis.collaborativeintelligence

import java.time.Duration
import java.time.Instant
import java.util.*

// TODO ensure that all are set somehow...
// MAYBE have a SampleBuilder which makes Samples? Or perhaps a boolean flag indicating "isInitialized"

// TODO convert to regular class...
data class Sample(
    var preprocessStart: Instant,
    var preprocessEnd: Instant,
    var inferenceStart: Instant,
    var inferenceEnd: Instant,
    var networkWriteStart: Instant,
    var networkWriteEnd: Instant,
    var networkReadStart: Instant,
    var networkReadEnd: Instant,
    var sampleString: String
) {
    constructor() : this(
        Instant.MIN,
        Instant.MIN,
        Instant.MIN,
        Instant.MIN,
        Instant.MIN,
        Instant.MIN,
        Instant.MIN,
        Instant.MIN,
        ""
    )

    override fun toString(): String = durations()
        .zip(durationDescriptions)
        .joinToString(separator = "\n") { (duration, description) ->
            "%.3fs %s".format(duration.toMillis() / 1000.0, description)
        } + "\n$sampleString"

    fun durations(): List<Duration> = listOf(
        Pair(preprocessStart, preprocessEnd),
        Pair(inferenceStart, inferenceEnd),
        Pair(networkWriteStart, networkWriteEnd),
        Pair(networkWriteEnd, networkReadEnd)
    )
        .map { Duration.between(it.first, it.second) }

    fun isInitialized(): Boolean =
        instants().all { it != Instant.MIN }

    fun instants(): List<Instant> = listOf(
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

class Statistics {

    var dropped = 0
        private set

    private val samples = Vector<Sample>()
    private val validSamples = Vector<Sample>()

    fun display(): String {
        // TODO last finished sample? specified sample number? or averages? or other things?
        return "Previous frame:\n%s\n\nDropped: %d\nFPS: %.3f\nDebug:\n%s\n%s".format(
            validSamples.lastElement(),
            dropped,
            fps(),
            validSamples.lastElement().networkWriteEnd,
            validSamples.lastElement().networkReadStart
        )
    }

    // TODO this doesn't make any sense... using only validSamples?
    // Maybe instantaneous FPS might be a useful metric too
    fun fps(): Double {
        val duration = Duration.between(
            validSamples.firstElement().preprocessStart,
            validSamples.lastElement().networkReadEnd
        )
        return validSamples.size.toDouble() / duration.toMillis().toDouble() * 1000.0
    }

    // TODO
    // fun latencyStats(): String {
    //     // Mean, median, stddev? samples.l
    //
    // }

    fun frameDropped() {
        dropped++
    }

    fun setPreprocess(frameNum: Int, start: Instant, end: Instant) {
        setPropsDecorator(frameNum) {
            samples[frameNum].preprocessStart = start
            samples[frameNum].preprocessEnd = end
        }
    }

    fun setInference(frameNum: Int, start: Instant, end: Instant) {
        setPropsDecorator(frameNum) {
            samples[frameNum].inferenceStart = start
            samples[frameNum].inferenceEnd = end
        }
    }

    fun setNetworkWrite(frameNum: Int, start: Instant, end: Instant) {
        setPropsDecorator(frameNum) {
            samples[frameNum].networkWriteStart = start
            samples[frameNum].networkWriteEnd = end
        }
    }

    fun setNetworkRead(frameNum: Int, start: Instant, end: Instant) {
        setPropsDecorator(frameNum) {
            samples[frameNum].networkReadStart = start
            samples[frameNum].networkReadEnd = end
        }
    }

    fun appendSampleString(frameNum: Int, sampleString: String) {
        setPropsDecorator(frameNum) {
            samples[frameNum].sampleString += sampleString
        }
    }

    private inline fun setPropsDecorator(frameNum: Int, func: () -> Unit) {
        resize(frameNum + 1)
        func()
        if (samples[frameNum].isInitialized())
            validSamples.add(samples[frameNum])
    }

    private fun resize(size: Int) {
        val n = size - samples.size
        if (n < 0) return
        samples.addAll(List(n) { Sample() })
    }
}
