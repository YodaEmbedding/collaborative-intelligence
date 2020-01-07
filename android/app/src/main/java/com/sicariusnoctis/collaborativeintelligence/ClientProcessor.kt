package com.sicariusnoctis.collaborativeintelligence

import android.renderscript.RenderScript
import android.util.Log
import io.fotoapparat.preview.Frame
import io.reactivex.Completable
import io.reactivex.Flowable
import io.reactivex.disposables.CompositeDisposable
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

    val modelConfig get() = inference!!.modelConfig

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
    fun switchModelInference(modelConfig: ModelConfig) = Completable.fromRunnable {
        _state = ClientProcessorState.BusyReconfigure
        inference!!.switchModel(modelConfig)
        setStateIfReady()
    }.subscribeOn(inferenceScheduler)

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
