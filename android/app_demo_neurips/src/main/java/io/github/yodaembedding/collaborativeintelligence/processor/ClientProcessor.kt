package io.github.yodaembedding.collaborativeintelligence.processor

import android.renderscript.RenderScript
import android.util.Log
import io.github.yodaembedding.collaborativeintelligence.*
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

    val modelConfig get() = inference!!.modelConfig
    val postencoderConfig get() = postencoderManager.postencoderConfig

    private var _state: ClientProcessorState = ClientProcessorState.Stopped
    val state @Synchronized get() = _state

    override fun close() {
        _state = ClientProcessorState.Stopped
        inferenceExecutor.submit { inference?.close() }
        inferenceExecutor.shutdown()
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

    fun switchModelInference(
        processorConfig: ProcessorConfig
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
            _state = ClientProcessorState.BusyInference
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
            postencoderManager.run(it)
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
