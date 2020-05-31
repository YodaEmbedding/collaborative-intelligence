package com.sicariusnoctis.collaborativeintelligence.processor.postencoders

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import com.sicariusnoctis.collaborativeintelligence.TensorLayout
import com.sicariusnoctis.collaborativeintelligence.Tiler
import java.io.ByteArrayOutputStream

class JpegPostencoder(inLayout: TensorLayout) : Postencoder {
    private val TAG = JpegPostencoder::class.qualifiedName

    var quality: Int = 20

    private val mbuSize = 16
    private val tiler = Tiler(inLayout, inLayout.squarishTiling(), mbuSize)
    private val uvData = ByteArray(tiler.height * tiler.width / 2) { -128 }

    override fun run(tensor: ByteArray): ByteArray {
        // val data = tiler.run(tensor)
        return yuvEncode(tensor)
    }

    private fun yuvEncode(tensor: ByteArray): ByteArray {
        val yData = tiler.run(tensor)
        val yuvData = yData + uvData
        val height = tiler.height
        val width = tiler.width
        val outStream = ByteArrayOutputStream()

        // TODO quality level?
        // TODO NV21? quality? tile with non-zero uvData?
        val yuvImage = YuvImage(yuvData, ImageFormat.NV21, width, height, null)
        yuvImage.compressToJpeg(Rect(0, 0, width, height), quality, outStream)

        return outStream.toByteArray()
    }
}
