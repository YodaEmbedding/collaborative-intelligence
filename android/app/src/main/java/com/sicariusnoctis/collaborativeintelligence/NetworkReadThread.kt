package com.sicariusnoctis.collaborativeintelligence

import io.reactivex.BackpressureStrategy
import io.reactivex.Flowable
import java.util.*
import java.util.concurrent.LinkedBlockingQueue

class NetworkReadThread : Thread {
    val outputStream: Flowable<String>

    private val networkAdapter: NetworkAdapter
    private val queue = LinkedBlockingQueue<Optional<String>>()

    constructor(networkAdapter: NetworkAdapter) : super() {
        this.networkAdapter = networkAdapter
        this.outputStream = Flowable.create({ subscriber ->
            // TODO Wait until thread is alive using CountDownLatch?
            // TODO thread.isAlive()? socket.isOpen? volatile boolean flag?
            while (true) {
                val item = queue.take()
                if (!item.isPresent)
                    break
                subscriber.onNext(item.get())
            }
            subscriber.onComplete()
        }, BackpressureStrategy.MISSING)
        // TODO BackpressureStrategy
    }

    override fun run() {
        try {
            while (true) {
                val data = networkAdapter.readData() ?: break
                queue.put(Optional.of(data))
            }
        } catch (e: InterruptedException) {

        } finally {
            queue.put(Optional.empty())
        }
    }
}