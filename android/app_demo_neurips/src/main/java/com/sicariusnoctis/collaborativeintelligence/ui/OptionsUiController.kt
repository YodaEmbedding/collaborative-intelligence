package com.sicariusnoctis.collaborativeintelligence.ui

import com.warkiz.widget.IndicatorSeekBar
import com.warkiz.widget.OnSeekChangeListener
import com.warkiz.widget.SeekParams

class OptionsUiController(private val uploadRateLimitSeekBar: IndicatorSeekBar) {
    var uploadRateLimit: Long? = null; private set

    init {
        uploadRateLimitSeekBar.onSeekChangeListener = object : OnSeekChangeListener {
            override fun onSeeking(seekParams: SeekParams) {
                updateFromLabel(seekParams.tickText)
            }

            override fun onStartTrackingTouch(seekBar: IndicatorSeekBar) {}

            override fun onStopTrackingTouch(seekBar: IndicatorSeekBar) {}
        }

        uploadRateLimitSeekBar.setIndicatorTextFormat("\${TICK_TEXT} KB/s")
    }

    private fun updateFromLabel(label: String) {
        uploadRateLimit = if (label == "∞") null else label.toLong()
    }
}
