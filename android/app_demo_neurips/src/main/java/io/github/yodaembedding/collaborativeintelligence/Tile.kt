package io.github.yodaembedding.collaborativeintelligence

import java.lang.Exception
import kotlin.math.ceil
import kotlin.math.sqrt

/**
 * Tile a tensor's channels in specified layout
 *
 * @param blockSize output array dimensions resized to be multiple of this value.
 */
class Tiler(val inLayout: TensorLayout, val outLayout: TiledLayout, val blockSize: Int) {
    val height = ceil(outLayout.tiledHeight.toDouble() / blockSize).toInt() * blockSize
    val width = ceil(outLayout.tiledWidth.toDouble() / blockSize).toInt() * blockSize

    init {
        assert(inLayout.h == outLayout.h)
        assert(inLayout.w == outLayout.w)
        assert(inLayout.c == outLayout.c)
    }

    // TODO replace with RenderScript or JNI
    fun run(tensor: ByteArray): ByteArray {
        assert(tensor.size == inLayout.size)

        val inArray = when (inLayout.order) {
            "chw" -> tensor
            "hwc" -> hwcToChw(tensor)
            else -> throw Exception("Unknown order")
        }

        val outArray = ByteArray(height * width)

        outer@
        for (i in 0 until outLayout.nrows) {
            val outRowOffset = width * outLayout.h * i
            for (j in 0 until outLayout.ncols) {
                if (i * outLayout.ncols + j >= outLayout.c)
                    break@outer
                val outChannelOffset = outRowOffset + outLayout.w * j
                for (k in 0 until outLayout.h) {
                    val inPos = outLayout.channelSize * (i * outLayout.ncols + j) + outLayout.w * k
                    val outPos = outChannelOffset + width * k
                    System.arraycopy(inArray, inPos, outArray, outPos, outLayout.w)
                }
            }
        }

        return outArray
    }

    // TODO replace with RenderScript or JNI
    /**
     * Returns tensor.reshape(h * w, c).T.reshape(-1)
     */
    private fun hwcToChw(tensor: ByteArray): ByteArray {
        return transpose(
            tensor,
            Pair(inLayout.channelSize, inLayout.c)
        )
    }

    companion object {
        /**
         * Transpose given tensor of given shape
         */
        private fun transpose(tensor: ByteArray, shape: Pair<Int, Int>): ByteArray {
            val outArray = ByteArray(shape.first * shape.second)

            for (j in 0 until shape.first) {
                for (i in 0 until shape.second) {
                    outArray[i * shape.first + j] = tensor[j * shape.second + i]
                }
            }

            return outArray
        }
    }
}

data class TensorLayout(val c: Int, val h: Int, val w: Int, val order: String) {
    val size = h * w * c
    val channelSize = h * w

    fun squarishTiling(): TiledLayout {
        val chans = c.toDouble()
        val rows = ceil(sqrt(chans)).toInt()
        val cols = ceil(chans / rows).toInt()
        return TiledLayout.create(
            this,
            rows,
            cols
        )
    }
}

data class TiledLayout(val c: Int, val h: Int, val w: Int, val nrows: Int, val ncols: Int) {
    val tiledHeight = h * nrows
    val tiledWidth = w * ncols
    val size = tiledHeight * tiledWidth
    val channelSize = h * w
    val numValidBytes = channelSize * c

    companion object {
        fun create(layout: TensorLayout, rows: Int, cols: Int) =
            TiledLayout(
                layout.c,
                layout.h,
                layout.w,
                rows,
                cols
            )
    }
}
