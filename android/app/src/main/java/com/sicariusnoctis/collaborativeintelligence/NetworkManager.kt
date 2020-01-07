package com.sicariusnoctis.collaborativeintelligence

import android.util.Log
import com.sicariusnoctis.collaborativeintelligence.processor.FrameRequest
import com.sicariusnoctis.collaborativeintelligence.ui.StatisticsUiController
import io.reactivex.Completable
import io.reactivex.Observable
import io.reactivex.Single
import io.reactivex.android.schedulers.AndroidSchedulers
import io.reactivex.disposables.CompositeDisposable
import io.reactivex.internal.schedulers.IoScheduler
import io.reactivex.observables.ConnectableObservable
import io.reactivex.processors.PublishProcessor
import io.reactivex.rxkotlin.subscribeBy
import io.reactivex.schedulers.Schedulers
import java.io.Closeable
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class NetworkManager(
    private val statistics: Statistics,
    private val statisticsUiController: StatisticsUiController
) : Closeable {
    private val TAG = NetworkManager::class.qualifiedName

    private var networkWriteExecutor = Executors.newSingleThreadExecutor()
    var networkWriteScheduler = Schedulers.from(networkWriteExecutor)
    private var networkAdapter: NetworkAdapter = NetworkAdapter()
    private var networkRead: Observable<Response>? = null
    private var networkWrite: PublishProcessor<Any>? = null
    private val subscriptions = CompositeDisposable()

    val timeUntilWriteAvailable get() = networkAdapter.timeUntilWriteAvailable

    var uploadLimitRate
        get() = networkAdapter.uploadLimitRate
        set(value) {
            networkAdapter.uploadLimitRate = value
        }

    override fun close() {
        networkAdapter.close()
    }

    fun dispose() {
        subscriptions.dispose()
    }

    fun connectNetworkAdapter() = Completable.fromRunnable {
        networkAdapter.connect()
    }
        .subscribeOn(IoScheduler())

    fun subscribeNetworkIo() = Completable.fromRunnable {
        // TODO don't really NEED a networkWrite observable... if we make a single executor thread
        networkWrite = PublishProcessor.create()
        val networkWriteRequests = networkWrite!!
            .subscribeOn(networkWriteScheduler)
            .onBackpressureBuffer()
            .share()

        // TODO collapse all these into single function...

        val networkWriteModelConfigSubscription = networkWriteRequests
            .filter { it is ModelConfig }
            .doOnNext { networkAdapter.writeModelConfig(it as ModelConfig) }
            .subscribeBy({ it.printStackTrace() })
        subscriptions.add(networkWriteModelConfigSubscription)

        val networkWriteFrameRequestSubscription = networkWriteRequests
            .filter { it is FrameRequest<*> && it.obj is ByteArray }
            .doOnNext { networkAdapter.writeFrameRequest(it as FrameRequest<ByteArray>) }
            .subscribeBy({ it.printStackTrace() })
        subscriptions.add(networkWriteFrameRequestSubscription)

        val networkWriteSampleSubscription = networkWriteRequests
            .filter { it is FrameRequest<*> && it.obj is Sample }
            .doOnNext { networkAdapter.writeSample(it as FrameRequest<Sample>) }
            .subscribeBy({ it.printStackTrace() })
        subscriptions.add(networkWriteSampleSubscription)

        val networkWritePingSubscription = networkWriteRequests
            .filter { it is PingRequest }
            .doOnNext { networkAdapter.writePingRequest(it as PingRequest) }
            .subscribeBy({ it.printStackTrace() })
        subscriptions.add(networkWritePingSubscription)

        networkRead = Observable.fromIterable(Iterable {
            iterator {
                while (true) {
                    val (result, start, end) = timed { networkAdapter.readResponse() }
                    if (result == null) break
                    val response = result!!
                    if (response is ResultResponse) {
                        val stats = statistics[response.frameNumber]
                        stats.setNetworkRead(response.frameNumber, start, end)
                        stats.setResultResponse(response.frameNumber, response)
                    }
                    yield(response)
                }
            }
        })
            .subscribeOn(IoScheduler())
            .publish()

        val networkModelReadySubscription = networkRead!!
            .filter { it is ModelReadyResponse }
            .map { it as ModelReadyResponse }
            .doOnNext { Log.i(TAG, "Model loaded on server: $it") }
            .subscribeBy({ it.printStackTrace() })
        subscriptions.add(networkModelReadySubscription)

        val networkPingResponsesSubscription = networkRead!!
            .filter { it is PingResponse }
            .map { it as PingResponse }
            .doOnNext { Log.i(TAG, "Ping received: ${it.id}") }
            .subscribeBy({ it.printStackTrace() })
        subscriptions.add(networkPingResponsesSubscription)

        // val networkResultResponses = networkRead!!
        val networkResultResponsesSubscription = networkRead!!
            .filter { it is ResultResponse }
            .map { it as ResultResponse }
            .map { statistics[it.frameNumber].sample }
            .observeOn(AndroidSchedulers.mainThread())
            .doOnNext { Log.i(TAG, "Finished processing frame ${it.frameNumber}") }
            .doOnNext { statisticsUiController.addSample(it) }
            .subscribeBy({ it.printStackTrace() })
        subscriptions.add(networkResultResponsesSubscription)

        (networkRead!! as ConnectableObservable).connect()
    }

    fun subscribePingGenerator() = Completable.fromRunnable {
        subscriptions.add(Observable
            .interval(10, TimeUnit.SECONDS)
            .observeOn(networkWriteScheduler, false, 1)
            .doOnNext { networkAdapter.writePingRequest(PingRequest(it.toInt())) }
            .doOnNext { Log.i(TAG, "Ping sent: $it") }
            .subscribeBy({ it.printStackTrace() })
        )
    }

    fun switchModelServer(modelConfig: ModelConfig): Completable = Single
        .just(modelConfig)
        .observeOn(networkWriteScheduler)
        .doOnSuccess { networkAdapter.writeModelConfig(it) }
        .ignoreElement()
        .andThen(switchModelServerObtainResponse(modelConfig))

    private fun switchModelServerObtainResponse(modelConfig: ModelConfig) = Completable.defer {
        networkRead!!
            .filter { it is ModelReadyResponse }
            .map { it as ModelReadyResponse }
            .firstOrError()
            .doOnSuccess {
                if (it.modelConfig != modelConfig)
                    throw Exception("Model config on server ${it.modelConfig} different from expected $modelConfig ")
            }
            .ignoreElement()
    }

    fun writeFrameRequest(frameRequest: FrameRequest<ByteArray>) {
        networkAdapter.writeFrameRequest(frameRequest)
    }
}
