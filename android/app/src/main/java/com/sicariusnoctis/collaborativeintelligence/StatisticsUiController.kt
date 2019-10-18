package com.sicariusnoctis.collaborativeintelligence

import android.annotation.SuppressLint
import android.widget.TextView

class StatisticsUiController(
    private val statistics: Statistics,
    private val predictionsText: TextView,
    private val fpsText: TextView,
    private val uploadText: TextView,
    private val preprocessText: TextView,
    private val clientInferenceText: TextView,
    private val encodingText: TextView,
    private val networkWaitText: TextView,
    private val totalText: TextView,
    private val framesProcessedText: TextView
    // private val framesDroppedText: TextView
) {
    @SuppressLint("SetTextI18n")
    fun addResponse(response: ResultResponse) {
        // TODO shouldn't these be synchronized? Or rather, statistics...
        // response.inferenceTime // TODO?
        predictionsText.text =
            response.predictions.joinToString("\n") { "${it.description} ${(it.score * 100).toInt()}%" }
        fpsText.text = "FPS: ${String.format("%.1f", statistics.fps)}"
        uploadText.text = "Upload: ${statistics.uploadBytes / 1024} KB"
        preprocessText.text = "Preprocess: ${statistics.preprocess.toMillis()} ms"
        clientInferenceText.text = "Client inference: ${statistics.clientInference.toMillis()} ms"
        encodingText.text = "Encoding: N/A"
        // encodingText.text = "Encoding: ${statistics.encoding.toMillis()} ms" // TODO
        networkWaitText.text = "Network wait: ${statistics.networkWait.toMillis()} ms"
        totalText.text = "Total: ${statistics.total.toMillis()} ms"
        framesProcessedText.text = "Processed: ${statistics.framesProcessed}" // TODO
        // framesDroppedText.text = "Dropped: ${statistics.framesDropped}"
    }
}