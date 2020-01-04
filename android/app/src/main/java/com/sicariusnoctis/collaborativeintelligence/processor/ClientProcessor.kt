package com.sicariusnoctis.collaborativeintelligence.processor

import android.renderscript.RenderScript
import android.util.Log
import com.sicariusnoctis.collaborativeintelligence.*
import io.fotoapparat.preview.Frame
import io.reactivex.Completable
import io.reactivex.Flowable
import io.reactivex.disposables.CompositeDisposable
import io.reactivex.internal.schedulers.ComputationScheduler
import io.reactivex.schedulers.Schedulers
import java.io.Closeable
import java.util.concurrent.Executors

class ClientProcessor(private val rs: RenderScript, private val statistics: Statistics) :
    Closeable {
    private val TAG = ClientProcessor::class.qualifiedName

    private var inference: Inference? = null
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val inferenceScheduler = Schedulers.from(inferenceExecutor)
    private var preprocessor: CameraPreviewPreprocessor? = null
    private val subscriptions = CompositeDisposable()
    private var postencoderManager = PostencoderManager()
    // private val tiler: Tiler? = null

    val modelConfig get() = inference!!.modelConfig
    val postencoderConfig get() = postencoderManager.postencoderConfig

    private var _state: ClientProcessorState = ClientProcessorState.Stopped
    val state @Synchronized get() = _state

    override fun close() {
        _state = ClientProcessorState.Stopped
        inferenceExecutor.submit { inference?.close() }
        inferenceExecutor.shutdown()
        // preprocessor.close() // TODO
        subscriptions.dispose()
    }

    fun initPreprocesor(width: Int, height: Int, rotationCompensation: Int) {
        preprocessor =
            CameraPreviewPreprocessor(rs, width, height, 224, 224, rotationCompensation)
        setStateIfReady()
    }

    fun initInference() = Completable.fromRunnable {
        inference = Inference()
        setStateIfReady()
    }.subscribeOn(inferenceScheduler)

    // TODO replace with directly creating new inference? and trigger postencoder = Postencoder()?
    fun switchModelInference(
        processorConfig: ProcessorConfig
        // modelConfig: ModelConfig,
        // postencoderConfig: PostencoderConfig
    ) = Completable.fromRunnable {
        _state = ClientProcessorState.BusyReconfigure
        preprocessor!!.dataType = when (processorConfig.modelConfig.layer) {
            "server" -> when (processorConfig.postencoderConfig.type) {
                "None" -> "uchar"
                "jpeg" -> "argb"
                else -> throw NotImplementedError()
            }
            else -> "float"
        }
        inference!!.switchModel(processorConfig.modelConfig)
        switchPostencoder(processorConfig).blockingAwait()
        setStateIfReady()
    }.subscribeOn(inferenceScheduler)

    fun switchPostencoder(processorConfig: ProcessorConfig) = Completable.fromRunnable {
        _state = ClientProcessorState.BusyReconfigure
        val layout = when (processorConfig.modelConfig.layer) {
            "client" -> null
            "server" -> TensorLayout(3, 224, 224, "hwc")
            else -> inference!!.outputTensorLayout
        }
        postencoderManager.switch(processorConfig, layout)
        setStateIfReady()
    }

    fun mapFrameRequests(frameRequests: Flowable<FrameRequest<Frame>>):
            Flowable<FrameRequest<ByteArray>> = frameRequests
        .doOnNext {
            if (_state != ClientProcessorState.Ready)
                throw Exception("Client processor is not ready")
            _state = ClientProcessorState.BusyInference // TODO
            Log.i(TAG, "Started processing frame ${it.info.frameNumber}")
        }
        .mapTimed(statistics, ModelStatistics::setPreprocess) {
            it.map(preprocessor!!::process)
        }
        .observeOn(inferenceScheduler, false, 1)
        .mapTimed(statistics, ModelStatistics::setInference) {
            inference!!.run(it)
        }
        .observeOn(ComputationScheduler(), false, 1)
        .mapTimed(statistics, ModelStatistics::setPostencode) {

            // TODO skip if client only

            // TODO use different layout/encoding if server only

            // TODO extract tiling directly to here, so we have executive control... let the encoder provide "recommend_tiling()" if it wants...

            // TODO also, shouldn't we tell the server what the tiling layout is? or can it figure it out? using the decoder's attributes...

            // tiler = Tiler(inLayout, inLayout.squarishTiling(), mbuSize)

            postencoderManager.run(it)
            // frameRequest.map {
            // JpegPostencoder(inference!!.outputTensorLayout).run(it)
            // postencoderManager.run(it)
            // TODO encoder should change... do we need a "State" monad? instead of passing everything in obj...
            // "State" holds all references to what the current inference request needs?
            // } // TODO hardcoded parameters, jpegEncoder reconstructed each time
        }
        .doOnNext {
            _state = ClientProcessorState.Ready
        }

    private fun setStateIfReady() {
        if (inference == null || preprocessor == null)
            return
        _state = ClientProcessorState.Ready
    }
}

enum class ClientProcessorState {
    Stopped,
    Ready,
    BusyInference,
    BusyReconfigure,
}

// It might make sense to have "contexts" to manage the time-variant resources

// usage:
// clientProcessor.switchModel()
// clientProcessor.subscribe(frames)
// onStop { clientProcessor.close() }
// onStart { clientProcessor = ClientProcessor() }
