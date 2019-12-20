package com.sicariusnoctis.collaborativeintelligence.ui

import android.annotation.SuppressLint
import android.graphics.Color
import android.widget.TextView
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.sicariusnoctis.collaborativeintelligence.RealtimeSortedEntryList
import com.sicariusnoctis.collaborativeintelligence.Sample
import com.sicariusnoctis.collaborativeintelligence.Statistics
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
        val orange = Color.rgb(255, 128, 0)
        val orangeH = Color.rgb(255, 160, 32)
        val blue = Color.rgb(0, 255, 255)

        uploadLineDataset = LineDataSet(RealtimeSortedEntryList(), "Upload (KB/frame)")
        uploadLineDataset.setDrawCircleHole(false)
        uploadLineDataset.setDrawCircles(false)
        uploadLineDataset.setDrawValues(false)
        uploadLineDataset.color = blue
        uploadLineDataset.lineWidth = 4f

        totalLineDataset = LineDataSet(RealtimeSortedEntryList(), "Total inference time (ms)")
        totalLineDataset.setDrawCircleHole(false)
        totalLineDataset.setDrawCircles(true)
        totalLineDataset.setDrawValues(false)
        totalLineDataset.circleColors = listOf(orangeH)
        totalLineDataset.circleRadius = 2f
        totalLineDataset.color = orange
        totalLineDataset.lineWidth = 4f

        lineData = LineData()
        lineData.addDataSet(uploadLineDataset)
        lineData.addDataSet(totalLineDataset)

        lineChart.axisLeft.axisMinimum = 0.0f
        lineChart.axisLeft.axisMaximum = 600.0f
        lineChart.axisRight.axisMinimum = 0.0f
        lineChart.axisRight.axisMaximum = 600.0f
        lineChart.data = lineData
        lineChart.description = null
        lineChart.legend.textSize = 12.0f
        lineChart.legend.xEntrySpace = 18.0f
        lineChart.invalidate()
    }

    @SuppressLint("SetTextI18n")
    private fun updateTextViews(sample: Sample) {
        val stats = statistics[sample.frameNumber]
        predictionsText.text = sample.resultResponse!!.predictions.joinToString("\n") {
            "${formatPercentage(it.score)} ${it.description}"
        }
        fpsText.text = "FPS: ${String.format("%.1f", stats.fps)}"
        uploadText.text = "Upload: ${sample.uploadBytes!! / 1024} KB/frame"
        preprocessText.text = "1. Preprocess: ${sample.preprocess.toMillis()} ms"
        clientInferenceText.text =
            "2. Client infer: ${sample.clientInference.toMillis()} ms"
        encodingText.text = "3. Encoding: N/A"
        networkWriteText.text = "4. Network send: ${sample.networkWrite.toMillis()} ms"
        serverInferenceText.text = "5. Server infer: ${sample.serverInference.toMillis()} ms"
        networkReadText.text = "6. Network read: ${sample.networkRead.toMillis()} ms"
        totalText.text = "Total: ${sample.total.toMillis()} ms"
        framesProcessedText.text = "Processed: ${stats.framesProcessed}"
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

    companion object {
        fun formatPercentage(score: Float): String {
            val s = (score * 100).toInt().toString()
            return if (s.length == 1) " $s%" else "$s%"
        }
    }
}