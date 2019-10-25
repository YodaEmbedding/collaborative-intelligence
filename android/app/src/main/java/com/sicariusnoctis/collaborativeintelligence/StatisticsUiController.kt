package com.sicariusnoctis.collaborativeintelligence

import android.annotation.SuppressLint
import android.graphics.Color
import android.util.Log
import android.widget.TextView
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import java.time.Duration
import java.time.Instant

class StatisticsUiController(
    private val statistics: Statistics,
    private val predictionsText: TextView,
    private val fpsText: TextView,
    private val uploadText: TextView,
    private val preprocessText: TextView,
    private val clientInferenceText: TextView,
    private val encodingText: TextView,
    private val networkReadText: TextView,
    private val serverInferenceText: TextView,
    private val networkWriteText: TextView,
    private val totalText: TextView,
    private val framesProcessedText: TextView,
    private val lineChart: LineChart
) {
    private val TAG = StatisticsUiController::class.qualifiedName

    private lateinit var totalLineDataset: LineDataSet
    private lateinit var uploadLineDataset: LineDataSet
    private lateinit var lineData: LineData
    private lateinit var epoch: Instant
    private lateinit var prevChartRefresh: Instant

    init {
        initChart()
    }

    fun addSample(sample: Sample) {
        updateTextViews(sample)
        updateChart(sample)
    }

    private fun initChart() {
        uploadLineDataset = LineDataSet(RealtimeSortedEntryList(), "Upload (KB/frame)")
        uploadLineDataset.setDrawCircles(false)
        uploadLineDataset.setDrawValues(false)
        uploadLineDataset.color = Color.rgb(0, 255, 255)
        uploadLineDataset.lineWidth = 2f

        totalLineDataset = LineDataSet(RealtimeSortedEntryList(), "Total inference time (ms)")
        totalLineDataset.setDrawCircles(false)
        totalLineDataset.setDrawValues(false)
        totalLineDataset.color = Color.rgb(255, 128, 0)
        totalLineDataset.lineWidth = 2f

        lineData = LineData()
        lineData.addDataSet(uploadLineDataset)
        lineData.addDataSet(totalLineDataset)

        lineChart.data = lineData
        lineChart.description = null
        lineChart.invalidate()
    }

    @SuppressLint("SetTextI18n")
    private fun updateTextViews(sample: Sample) {
        predictionsText.text = sample.resultResponse!!.predictions.joinToString("\n") {
            "${it.description} ${(it.score * 100).toInt()}%"
        }
        fpsText.text = "FPS: ${String.format("%.1f", statistics.fps)}"
        uploadText.text = "Upload: ${sample.uploadBytes!! / 1024} KB/frame"
        preprocessText.text = "1. Preprocess: ${sample.preprocess.toMillis()} ms"
        clientInferenceText.text =
            "2. Client infer: ${sample.clientInference.toMillis()} ms"
        encodingText.text = "3. Encoding: N/A"
        networkWriteText.text = "4. Network send: ${sample.networkWrite.toMillis()} ms"
        serverInferenceText.text = "5. Server infer: ${sample.serverInference.toMillis()} ms"
        networkReadText.text = "6. Network read: ${sample.networkRead.toMillis()} ms"
        totalText.text = "Total: ${sample.total.toMillis()} ms"
        framesProcessedText.text = "Processed: ${statistics.framesProcessed}"
    }

    private fun updateChart(sample: Sample) {
        if (!::epoch.isInitialized)
            epoch = sample.preprocessStart!!

        val t = Duration.between(epoch, sample.preprocessStart).toMillis() / 1000f
        val y1 = sample.total.toMillis().toFloat()
        val y2 = sample.uploadBytes!!.toFloat() / 1024
        totalLineDataset.addEntry(Entry(t, y1))
        uploadLineDataset.addEntry(Entry(t, y2))

        if (::prevChartRefresh.isInitialized &&
            Duration.between(prevChartRefresh, Instant.now()) < Duration.ofMillis(200)
        )
            return

        prevChartRefresh = Instant.now()

        lineData.notifyDataChanged()
        lineChart.notifyDataSetChanged()
        lineChart.xAxis.axisMinimum = totalLineDataset.xMax - 10f
        lineChart.xAxis.axisMaximum = totalLineDataset.xMax
        lineChart.invalidate()

        // TODO skip first sample whenever modelChange happens?
    }
}