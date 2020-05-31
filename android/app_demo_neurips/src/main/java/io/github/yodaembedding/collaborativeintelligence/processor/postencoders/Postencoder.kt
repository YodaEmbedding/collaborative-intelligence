package io.github.yodaembedding.collaborativeintelligence.processor.postencoders

interface Postencoder {
    fun run(tensor: ByteArray): ByteArray
    // fun setArgs
}
