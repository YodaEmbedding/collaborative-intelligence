package com.sicariusnoctis.collaborativeintelligence

import androidx.test.runner.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.DataOutputStream
import java.net.Socket
import java.time.Duration
import java.time.Instant

@RunWith(AndroidJUnit4::class)
class NetworkAdapterTest {
    @Test
    fun uploadSimple() {
        val frame = ByteArray(224 * 224 * 3 * 4) { 0 }

        val HOSTNAME = HOSTNAME
        val PORT = 5678
        val socket = Socket(HOSTNAME, PORT)
        socket.sendBufferSize = 1024 * 64
        val dataOutputStream = DataOutputStream(socket.getOutputStream())

        for (i in 0..1000) {
            dataOutputStream.writeInt(frame.size)
            dataOutputStream.write(frame)
        }
    }

    @Test
    fun uploadFrames() {
        val modelConfig = ModelConfig(
            model = "resnet18",
            layer = "server",
            encoder = "None",
            decoder = "None"
        )
        val frame = ByteArray(224 * 224 * 3 * 4) { 0 }
        val frameRequestInfo = FrameRequestInfo(0, modelConfig)
        val frameRequest = FrameRequest(frame, frameRequestInfo)

        val networkAdapter = NetworkAdapter()
        networkAdapter.connect()
        networkAdapter.writeModelConfig(modelConfig)

        for (i in 0..1000) {
            val t1 = Instant.now()
            networkAdapter.writeFrameRequest(frameRequest)

            val confirmationResponse = networkAdapter.readResponse()
            assert(confirmationResponse is ResultResponse)

            val resultResponse = networkAdapter.readResponse()
            assert(resultResponse is ResultResponse)
            assertEquals(0, (resultResponse as ResultResponse).frameNumber)

            val t2 = Instant.now()
            val dt = Duration.between(t1, t2)
            assertTrue("$dt exceeds time limit", dt < Duration.ofMillis(1000))
            println("$i: ${dt.toMillis()} ms")
        }
    }
}