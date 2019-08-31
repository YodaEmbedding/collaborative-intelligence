package com.sicariusnoctis.collaborativeintelligence

import android.util.Log
import java.net.Socket
import java.io.BufferedReader
import java.io.DataOutputStream
import java.io.InputStreamReader
import java.lang.Exception
import java.nio.ByteBuffer

class NetworkAdapter {
    private var socket: Socket? = null
    private var outputStream: DataOutputStream? = null
    private var inputStream: BufferedReader? = null

    fun connect() {
        try {
            socket = Socket(HOSTNAME, 5678)
            outputStream = DataOutputStream(socket?.getOutputStream())
            inputStream =
                BufferedReader(InputStreamReader(socket?.getInputStream()))
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun close() {
        try {
            socket?.close()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun writeData(msg: ByteArray) {
        val msg_len = ByteBuffer.allocate(4).putInt(msg.size).array()
        val eol = ByteArray(1) {0}
        Log.i("Message size:", msg.size.toString())
        outputStream?.write(msg_len)
        outputStream?.write(eol)
        outputStream?.write(msg)
        outputStream?.write(eol)
    }
}