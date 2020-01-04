package com.sicariusnoctis.collaborativeintelligence

import android.renderscript.RenderScript
import io.reactivex.Completable
import io.reactivex.disposables.CompositeDisposable
import io.reactivex.schedulers.Schedulers
import java.io.Closeable
import java.util.concurrent.Executors

class ClientProcessor(private val rs: RenderScript) : Closeable {
    private val TAG = ClientProcessor::class.qualifiedName

    var inference: Inference? = null
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    val inferenceScheduler = Schedulers.from(inferenceExecutor)
    lateinit var postprocessor: CameraPreviewPostprocessor
    private val subscriptions = CompositeDisposable()

    val modelConfig get() = inference!!.modelConfig

    override fun close() {
        inferenceExecutor.submit { inference?.close() }
        inferenceExecutor.shutdown()
        // postprocessor.close() // TODO
        subscriptions.dispose()
    }

    fun initPostprocessor(width: Int, height: Int, rotationCompensation: Int) {
        postprocessor =
            CameraPreviewPostprocessor(rs, width, height, 224, 224, rotationCompensation)
    }

    fun initInference() = Completable.fromRunnable {
        inference = Inference()
    }
        .subscribeOn(inferenceScheduler)

    // TODO replace with directly creating new inference? and trigger postencoder = Postencoder()?
    fun switchModelInference(modelConfig: ModelConfig) = Completable.fromRunnable {
        inference!!.switchModel(modelConfig)
    }.subscribeOn(inferenceScheduler)
}

// It might make sense to have "contexts" to manage the time-variant resources

// usage:
// clientProcessor.switchModel()
// clientProcessor.subscribe(frames)
// onStop { clientProcessor.close() }
// onStart { clientProcessor = ClientProcessor() }