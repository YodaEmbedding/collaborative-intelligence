package com.sicariusnoctis.collaborativeintelligence

import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.renderscript.RenderScript
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.sicariusnoctis.collaborativeintelligence.processor.*
import com.sicariusnoctis.collaborativeintelligence.ui.ModelUiController
import com.sicariusnoctis.collaborativeintelligence.ui.OptionsUiController
import com.sicariusnoctis.collaborativeintelligence.ui.PostencoderUiController
import com.sicariusnoctis.collaborativeintelligence.ui.StatisticsUiController
import io.fotoapparat.parameter.Resolution
import io.fotoapparat.preview.Frame
import io.reactivex.Completable
import io.reactivex.Flowable
import io.reactivex.disposables.CompositeDisposable
import io.reactivex.internal.schedulers.IoScheduler
import io.reactivex.processors.PublishProcessor
import io.reactivex.rxkotlin.subscribeBy
import io.reactivex.rxkotlin.zipWith
import kotlinx.android.synthetic.main.activity_main.*
import java.time.Duration
import java.time.Instant
import java.util.concurrent.CompletableFuture

class MainActivity : AppCompatActivity() {
    private val TAG = MainActivity::class.qualifiedName

    private lateinit var camera: Camera
    private lateinit var clientProcessor: ClientProcessor
    private lateinit var networkManager: NetworkManager
    private lateinit var previewResolution: Resolution
    private lateinit var rs: RenderScript
    private lateinit var modelUiController: ModelUiController
    private lateinit var optionsUiController: OptionsUiController
    private lateinit var postencoderUiController: PostencoderUiController
    private lateinit var statisticsUiController: StatisticsUiController

    private var prevFrameTime: Instant? = null
    private val frameProcessor: PublishProcessor<Frame> = PublishProcessor.create()
    private val statistics = Statistics()
    private val subscriptions = CompositeDisposable()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        rs = RenderScript.create(this)
        camera = Camera(this, cameraView) { frame -> this.frameProcessor.onNext(frame) }
        modelUiController = ModelUiController(
            modelSpinner,
            layerSeekBar,
            compressionSpinner
        )
        optionsUiController = OptionsUiController(
            uploadRateLimitSeekBar
        )
        postencoderUiController = PostencoderUiController(
            modelUiController.modelConfig,
            modelUiController.modelConfigEvents,
            // modelUiController.modelConfigEvents.startWith(modelUiController.modelConfig)
            postencoderSpinner,
            postencoderQualitySeekBar
        )
        statisticsUiController = StatisticsUiController(
            statistics,
            predictionsText,
            fpsText,
            uploadText,
            uploadAvgText,
            preprocessText,
            clientInferenceText,
            encodingText,
            // networkReadText,
            serverInferenceText,
            // networkWriteText,
            totalText,
            totalAvgText,
            framesProcessedText,
            lineChart
        )
    }

    override fun onStart() {
        super.onStart()

        clientProcessor = ClientProcessor(rs, statistics)
        networkManager = NetworkManager(statistics, statisticsUiController)

        val cameraReady = CompletableFuture<Unit>()
        camera.start()
        camera.getCurrentParameters().whenAvailable {
            this.previewResolution = it!!.previewResolution
            cameraReady.complete(Unit)
        }

        Completable
            .mergeArray(
                Completable.fromFuture(cameraReady),
                clientProcessor.initInference(),
                networkManager.connectNetworkAdapter()
            )
            .subscribeOn(IoScheduler())
            .andThen(initPreprocessor())
            .andThen(networkManager.subscribeNetworkIo())
            .andThen(networkManager.subscribePingGenerator())
            .andThen(
                switchModel(
                    modelUiController.modelConfig,
                    postencoderUiController.postencoderConfig
                )
            )
            .andThen(subscribeFrameProcessor())
            .subscribeBy({ it.printStackTrace() })
    }

    override fun onStop() {
        super.onStop()
        camera.stop()
        clientProcessor.close()
        networkManager.close()
        networkManager.dispose()
    }

    override fun onDestroy() {
        subscriptions.dispose()
        super.onDestroy()
    }

    private fun initPreprocessor() = Completable.fromRunnable {
        val rotationCompensation = CameraPreviewPreprocessor.getRotationCompensation(
            this, CameraCharacteristics.LENS_FACING_BACK
        )

        clientProcessor.initPreprocesor(
            previewResolution.width,
            previewResolution.height,
            rotationCompensation
        )
    }

    // TODO this is kind of using references to things that stop existing after onStop
    private fun subscribeFrameProcessor() = Completable.fromRunnable {
        // clientProcessor.subscribeFrameProcessor(frameProcessor)

        // TODO why does FrameRequestInfo NEED modelConfig? does it need other options too?
        // TODO are we going to handle post processor switching somewhere? where?

        val frameRequests = frameProcessor
            .subscribeOn(IoScheduler())
            .onBackpressureDrop()
            .observeOn(IoScheduler(), false, 1)
            .map {
                val info = FrameRequestInfo(
                    -1,
                    modelUiController.modelConfig,
                    postencoderUiController.postencoderConfig
                )
                FrameRequest(it, info)
            }
            .onBackpressureLimitRate(
                onDrop = { statistics[it.info.modelConfig].frameDropped() },
                limit = { shouldProcessFrame(it.info.modelConfig, it.info.postencoderConfig) }
            )
            .zipWith(Flowable.range(0, Int.MAX_VALUE)) { (frame, info), frameNumber ->
                prevFrameTime = Instant.now()
                statistics[info.modelConfig].makeSample(frameNumber)
                FrameRequest(frame, info.replace(frameNumber))
            }

        val clientRequests = clientProcessor.mapFrameRequests(frameRequests)

        val clientRequestsSubscription = clientRequests
            .observeOn(networkManager.networkWriteScheduler, false, 1)
            .doOnNext {
                networkManager.uploadLimitRate = optionsUiController.uploadRateLimit
            }
            .doOnNextTimed(statistics, ModelStatistics::setNetworkWrite) {
                networkManager.writeFrameRequest(it)
            }
            .doOnNext {
                statistics[it.info.modelConfig].setUpload(it.info.frameNumber, it.obj.size)
            }
            .subscribeBy({ it.printStackTrace() })

        subscriptions.add(clientRequestsSubscription)
    }

    // TODO deal with unnecessary ProcessorConfig constructions...
    // TODO rename to switchProcessor? meh...
    // TODO clean up using .log() extension method
    private fun switchModel(modelConfig: ModelConfig, postencoderConfig: PostencoderConfig) =
        Completable
            .fromRunnable { Log.i(TAG, "Switching model begin") }
            .andThen(
                Completable.mergeArray(
                    clientProcessor.switchModelInference(
                        ProcessorConfig(
                            modelConfig,
                            postencoderConfig
                        )
                    ),
                    networkManager.switchModelServer(
                        ProcessorConfig(
                            modelConfig,
                            postencoderConfig
                        )
                    )
                )
            )
            .doOnComplete { Log.i(TAG, "Switching model end") }

    // private fun switchPostencoder(postencoderConfig: PostencoderConfig) =
    //     Completable
    //         .fromRunnable { Log.i(TAG, "Switching postencoder begin") }
    //         .andThen(
    //             Completable.mergeArray(
    //                 clientProcessor.switchPostencoder(postencoderConfig),
    //                 networkManager.switchPostencoderServer(postencoderConfig)
    //             )
    //         )
    //         .doOnComplete { Log.i(TAG, "Switching postencoder end") }

    // TODO extract gate... maybe split into two filters (second half handled by stats only)
    private fun shouldProcessFrame(
        modelConfig: ModelConfig,
        postencoderConfig: PostencoderConfig
    ): Boolean {
        val stats = statistics[modelConfig]

        // Load new model if config changed
        if (modelConfig != clientProcessor.modelConfig) {
            // Wait till latest sample has been processed client-side first
            if (clientProcessor.state == ClientProcessorState.BusyReconfigure) {
                Log.i(TAG, "Dropped frame because waiting to switch models")
                return false
            }

            // TODO remove blockingAwait by setting a mutex?
            switchModel(modelConfig, postencoderConfig).blockingAwait()
            statistics[modelConfig] = ModelStatistics()
            Log.i(TAG, "Dropped frame after switching model")
            return false
        }

        // Load new postencoder if config changed
        if (postencoderConfig != clientProcessor.postencoderConfig) {
            // Wait till latest sample has been processed client-side first
            if (clientProcessor.state == ClientProcessorState.BusyReconfigure) {
                Log.i(TAG, "Dropped frame because waiting to switch postencoders")
                return false
            }

            // switchPostencoder(postencoderConfig).blockingAwait()
            switchModel(modelConfig, postencoderConfig).blockingAwait()
            statistics[modelConfig] = ModelStatistics()
            Log.i(TAG, "Dropped frame after switching postencoder")
            return false
        }

        if (!stats.isFirstExist)
            return true

        // If first server response is not yet received, drop frame
        if (stats.isEmpty) {
            Log.i(TAG, "Dropped frame because waiting for first server response")
            return false
        }

        // TODO watch out for encoding time...
        if (stats.currentSample?.inferenceEnd == null) {
            Log.i(TAG, "Dropped frame because frame is currently being processed")
            return false
        }

        if (stats.currentSample?.networkWriteEnd == null) {
            Log.i(TAG, "Dropped frame because frame is currently being written over network")
            return false
        }

        if (stats.currentSample?.networkReadEnd == null) {
            // Log.i(TAG, "Dropped frame because frame is currently being processed by server")
            return false
        }

        val sample = stats.sample

        val enoughTime = Duration.ofMillis(0)
        if (networkManager.timeUntilWriteAvailable > enoughTime) {
            Log.i(TAG, "Dropped frame because of slow upload speed!")
            return false
        }

        val timeSinceLastFrameRequest = Duration.between(prevFrameTime, Instant.now())

        if (timeSinceLastFrameRequest < sample.clientInference) {
            Log.i(TAG, "Dropped frame because of slow client inference time!")
            return false
        }

        if (timeSinceLastFrameRequest < sample.serverInference) {
            Log.i(TAG, "Dropped frame because of slow server inference time!")
            return false
        }

        return true
    }
}
