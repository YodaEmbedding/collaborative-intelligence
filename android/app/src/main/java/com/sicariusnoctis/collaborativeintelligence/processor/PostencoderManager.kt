package com.sicariusnoctis.collaborativeintelligence.processor

import com.sicariusnoctis.collaborativeintelligence.PostencoderConfig
import com.sicariusnoctis.collaborativeintelligence.ProcessorConfig
import com.sicariusnoctis.collaborativeintelligence.TensorLayout
import com.sicariusnoctis.collaborativeintelligence.processor.postencoders.JpegPostencoder
import com.sicariusnoctis.collaborativeintelligence.processor.postencoders.JpegRgbPostencoder
import com.sicariusnoctis.collaborativeintelligence.processor.postencoders.Postencoder
import java.lang.IllegalArgumentException

// TODO factor out preprocessor? or maybe rename CameraPreviewPreprocessor to something else...
class PostencoderManager {
    lateinit var postencoderConfig: PostencoderConfig; private set

    private var postencoder: Postencoder? = null

    fun run(frameRequest: FrameRequest<ByteArray>): FrameRequest<ByteArray> {
        return frameRequest.map { run(it) }
    }

    private fun run(inputArray: ByteArray): ByteArray {
        return postencoder?.run(inputArray) ?: inputArray
    }

    // TODO If preprocessor runs on multiple threads, uh oh
    fun switch(processorConfig: ProcessorConfig, inLayout: TensorLayout?) {
        this.postencoderConfig = processorConfig.postencoderConfig
        postencoder = when (processorConfig.postencoderConfig.type) {
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

    // TODO switch? or how to utilize models.json?

    // TODO extract models.json reader?  from the ui? but... UI provides info anyways
}
