package com.sicariusnoctis.collaborativeintelligence.processor.postencoders

import com.sicariusnoctis.collaborativeintelligence.TensorLayout

interface Postencoder {
    fun run(tensor: ByteArray): ByteArray
    // fun setArgs
}
