package com.sicariusnoctis.collaborativeintelligence.processor.postencoders

import android.graphics.Bitmap
import com.sicariusnoctis.collaborativeintelligence.TensorLayout
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.IntBuffer


class JpegRgbPostencoder(private val inLayout: TensorLayout) : Postencoder {
    private val TAG = JpegRgbPostencoder::class.qualifiedName

    override fun run(argb: ByteArray): ByteArray {
        // val argb = preprocessor.convertRgbToArgb(rgb)

        val byteBuffer = ByteBuffer.wrap(argb)
        val intBuffer = byteBuffer.asIntBuffer()

        val argbInt = IntArray(intBuffer.remaining())
        intBuffer.get(argbInt)

        val bitmap = Bitmap.createBitmap(inLayout.w, inLayout.h, Bitmap.Config.ARGB_8888, false)
        bitmap.copyPixelsFromBuffer(IntBuffer.wrap(argbInt))

        val outStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 70, outStream)
        return outStream.toByteArray()
    }
}
