package com.sicariusnoctis.collaborativeintelligence

import java.time.Duration
import java.time.Instant
import java.util.*

class Statistics {
    var framesDropped = 0
        @Synchronized get
        private set
    val framesProcessed
        @Synchronized get() = validSamples.size
    val fps
        @Synchronized get() = fps()
    val uploadBytes
        @Synchronized get() = last.uploadBytes!!
    val preprocess: Duration
        @Synchronized get() = Duration.between(last.preprocessStart, last.preprocessEnd)
    val clientInference: Duration
        @Synchronized get() = Duration.between(last.inferenceStart, last.inferenceEnd)
    // val encoding: Duration
    //     @Synchronized get() = Duration.between(last.encodingStart, last.encodingEnd)
    val networkWait: Duration
        @Synchronized get() = Duration.between(last.networkWriteStart, last.networkReadEnd)
    val total: Duration
        @Synchronized get() = Duration.between(last.preprocessStart, last.networkReadEnd)

    private val samples = Vector<Sample>()
    private val validSamples = Vector<Sample>()
    private val first get() = validSamples.firstElement()
    private val last get() = validSamples.lastElement()

    @Synchronized
    fun frameDropped() {
        framesDropped++
    }

    fun setPreprocess(frameNum: Int, start: Instant, end: Instant) {
        setPropsDecorator(frameNum) {
            it.preprocessStart = start
            it.preprocessEnd = end
        }
    }

    fun setInference(frameNum: Int, start: Instant, end: Instant) {
        setPropsDecorator(frameNum) {
            it.inferenceStart = start
            it.inferenceEnd = end
        }
    }

    fun setNetworkWrite(frameNum: Int, start: Instant, end: Instant) {
        setPropsDecorator(frameNum) {
            it.networkWriteStart = start
            it.networkWriteEnd = end
        }
    }

    fun setNetworkRead(frameNum: Int, start: Instant, end: Instant) {
        setPropsDecorator(frameNum) {
            it.networkReadStart = start
            it.networkReadEnd = end
        }
    }

    fun setUpload(frameNum: Int, uploadBytes: Int) {
        setPropsDecorator(frameNum) {
            it.uploadBytes = uploadBytes
        }
    }

    fun appendSampleString(frameNum: Int, sampleString: String) {
        setPropsDecorator(frameNum) {
            it.sampleString += sampleString
        }
    }

    // TODO reset validSamples? or something like that
    // TODO this doesn't make any sense... using only validSamples?
    private fun fps() = 1000.0 * minOf(validSamples.size, 10) / Duration.between(
        validSamples.elementAt(maxOf(validSamples.size - 10, 0)).preprocessStart,
        last.networkReadEnd
    ).toMillis().toDouble()

    private fun resize(size: Int) {
        val n = size - samples.size
        if (n < 0) return
        samples.addAll(List(n) { Sample() })
    }

    @Synchronized
    private fun setPropsDecorator(frameNum: Int, func: (Sample) -> Unit) {
        resize(frameNum + 1)
        func(samples[frameNum])
        if (samples[frameNum].isValid)
            validSamples.add(samples[frameNum])
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
    var sampleString: String = ""
) {
    override fun toString() = durations()
        .zip(durationDescriptions)
        .joinToString(separator = "\n") { (duration, description) ->
            "%.3fs %s".format(duration.toMillis() / 1000.0, description)
        } + "\n$sampleString"

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
