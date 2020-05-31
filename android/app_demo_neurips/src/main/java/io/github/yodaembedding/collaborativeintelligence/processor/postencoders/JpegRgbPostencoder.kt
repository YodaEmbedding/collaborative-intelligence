package io.github.yodaembedding.collaborativeintelligence.processor.postencoders

import android.graphics.Bitmap
import io.github.yodaembedding.collaborativeintelligence.TensorLayout
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.IntBuffer


class JpegRgbPostencoder(private val inLayout: TensorLayout) : Postencoder {
    private val TAG = JpegRgbPostencoder::class.qualifiedName

    var quality: Int = 70

    override fun run(argb: ByteArray): ByteArray {
        val byteBuffer = ByteBuffer.wrap(argb)
        val intBuffer = byteBuffer.asIntBuffer()

        val argbInt = IntArray(intBuffer.remaining())
        intBuffer.get(argbInt)

        val bitmap = Bitmap.createBitmap(inLayout.w, inLayout.h, Bitmap.Config.ARGB_8888, false)
        bitmap.copyPixelsFromBuffer(IntBuffer.wrap(argbInt))

        val outStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, outStream)
        return outStream.toByteArray()
    }
}
