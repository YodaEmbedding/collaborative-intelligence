package com.sicariusnoctis.collaborativeintelligence

import android.graphics.Bitmap
import java.io.ByteArrayOutputStream
import java.util.concurrent.LinkedBlockingQueue

class NetworkThread : Thread() {
    private var sendQueue = LinkedBlockingQueue<ByteArray>()
    private var networkAdapter = NetworkAdapter()

    override fun run() {
        try {
            networkAdapter.connect()
            while (true) {
                // TODO pop away extra frames if "REALTIME" mode set... also count # dropped
                val data = sendQueue.take()
                networkAdapter.writeData(data)
            }
        } catch (e: InterruptedException) {

        } finally {
            networkAdapter.close()
        }
    }

    fun writeData(data: ByteArray) {
        sendQueue.add(data)
    }

    fun writeData(data: Bitmap) {
        val stream = ByteArrayOutputStream()
        data.compress(Bitmap.CompressFormat.JPEG, 90, stream)
        sendQueue.add(stream.toByteArray())
    }
}