package com.sicariusnoctis.collaborativeintelligence.processor

import com.sicariusnoctis.collaborativeintelligence.PostencoderConfig
import com.sicariusnoctis.collaborativeintelligence.ProcessorConfig
import com.sicariusnoctis.collaborativeintelligence.TensorLayout
import com.sicariusnoctis.collaborativeintelligence.processor.postencoders.JpegPostencoder
import com.sicariusnoctis.collaborativeintelligence.processor.postencoders.JpegRgbPostencoder
import com.sicariusnoctis.collaborativeintelligence.processor.postencoders.Postencoder

class PostencoderManager {
    lateinit var postencoderConfig: PostencoderConfig; private set

    private var postencoder: Postencoder? = null

    fun run(frameRequest: FrameRequest<ByteArray>): FrameRequest<ByteArray> {
        return frameRequest.map { run(it) }
    }

    private fun run(inputArray: ByteArray): ByteArray {
        return postencoder?.run(inputArray) ?: inputArray
    }

    fun switch(processorConfig: ProcessorConfig, inLayout: TensorLayout?) {
        this.postencoderConfig = processorConfig.postencoderConfig
        postencoder = makePostencoder(processorConfig, inLayout)
        // TODO This should be delegated via polymorphism, not runtime casting...
        when (postencoder) {
            is JpegPostencoder -> {
                (postencoder as JpegPostencoder).quality = postencoderConfig.quality
            }
            is JpegRgbPostencoder -> {
                (postencoder as JpegRgbPostencoder).quality = postencoderConfig.quality
            }
        }
    }

    private fun makePostencoder(processorConfig: ProcessorConfig, inLayout: TensorLayout?) =
        when (processorConfig.postencoderConfig.type) {
            "None" -> null
            "jpeg" -> when (processorConfig.modelConfig.layer) {
                "client" -> throw IllegalArgumentException()
                "server" -> JpegRgbPostencoder(inLayout!!)
                else -> JpegPostencoder(inLayout!!)
            }
            "h264" -> throw NotImplementedError()
            else -> throw NotImplementedError()
        }
}
