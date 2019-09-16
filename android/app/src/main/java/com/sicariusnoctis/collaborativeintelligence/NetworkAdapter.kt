package com.sicariusnoctis.collaborativeintelligence

import android.util.Log
import java.io.BufferedReader
import java.io.DataOutputStream
import java.io.InputStreamReader
import java.net.Socket
import java.nio.ByteBuffer

class NetworkAdapter {
    private val TAG = NetworkAdapter::class.qualifiedName

    private var inputStream: BufferedReader? = null
    private var outputStream: DataOutputStream? = null
    private var socket: Socket? = null

    fun connect() {
        socket = Socket(HOSTNAME, 5678)

        // TODO BufferedInputStream for byte data?
        // https://stackoverflow.com/questions/15538509/dealing-with-end-of-file-using-bufferedreader-read
        inputStream = BufferedReader(InputStreamReader(socket!!.inputStream))
        outputStream = DataOutputStream(socket!!.outputStream)
    }

    fun close() {
        socket?.close()
        socket = null
        inputStream = null
        outputStream = null
    }

    fun readData(): String? {
        val data = inputStream!!.readLine()
        Log.i(TAG, "Received: $data")
        return data
    }

    fun writeData(msg: ByteArray) {
        val msgLen = ByteBuffer.allocate(4).putInt(msg.size).array()
        val eol = ByteArray(1) { 0 }
        Log.i(TAG, "Message size: ${msg.size}")
        outputStream!!.write(msgLen)
        outputStream!!.write(eol)
        outputStream!!.write(msg)
        outputStream!!.write(eol)
    }
}