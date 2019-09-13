package com.sicariusnoctis.collaborativeintelligence

import io.reactivex.*
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.lang.Thread
import java.util.Optional
import java.util.concurrent.LinkedBlockingQueue

class NetworkReadThread : Thread {
    val outputStream: Flowable<String>

    private val inputStream: BufferedReader
    private val queue = LinkedBlockingQueue<Optional<String>>()

    constructor(inputStream: InputStream) : super() {
        // TODO BufferedInputStream for byte data?
        // https://stackoverflow.com/questions/15538509/dealing-with-end-of-file-using-bufferedreader-read
        this.inputStream = BufferedReader(InputStreamReader(inputStream))
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
    }

    override fun run() {
        try {
            while (true) {
                val data = inputStream.readLine() ?: break
                println("Found data! $data")
                queue.put(Optional.of(data))
            }
        } catch (e: InterruptedException) {

        } finally {
            queue.put(Optional.empty())
        }
    }
}