package com.sicariusnoctis.collaborativeintelligence.ui

import android.view.View
import android.widget.AdapterView
import android.widget.Spinner
import com.sicariusnoctis.collaborativeintelligence.ModelConfig
import com.sicariusnoctis.collaborativeintelligence.loadModelConfigMap
import com.sicariusnoctis.collaborativeintelligence.updateSpinner
import com.warkiz.widget.IndicatorSeekBar
import com.warkiz.widget.OnSeekChangeListener
import com.warkiz.widget.SeekParams
import io.reactivex.subjects.PublishSubject

class ModelUiController(
    private val modelSpinner: Spinner,
    private val layerSeekBar: IndicatorSeekBar,
    private val compressionSpinner: Spinner
) {
    var modelConfig: ModelConfig
        @Synchronized get() = _modelConfig
        private set(value) {
            _modelConfig = value
            modelConfigEvents.onNext(value)
        }

    val modelConfigEvents = PublishSubject.create<ModelConfig>()

    private lateinit var _modelConfig: ModelConfig
    private val modelConfigMap = loadModelConfigMap("models.json")
    private val model
        get() = modelSpinner.getItemAtPosition(modelSpinner.selectedItemPosition).toString()
    private val layer
        get() = when (layerChoices.size) {
            1 -> layerChoices[0]
            else -> layerChoices[layerSeekBar.progress]
        }
    private val compression
        get() = compressionSpinner.getItemAtPosition(compressionSpinner.selectedItemPosition).toString()

    private val modelConfigs get() = modelConfigMap.filter { it.key == model }.flatMap { it.value }
    private val modelChoices get() = modelConfigMap.keys.toList()
    private val layerChoices get() = LinkedHashSet<String>(modelConfigs.map { it.layer }).toList()
    private val compressionChoices get() = modelConfigs.filter { it.layer == layer }.map { it.encoder }
    private val choiceHistory = ChoiceHistory()

    init {
        initUiChoices()
        initUiHandlers()
    }

    private fun initUiChoices() {
        updateChoices(updateModel = true, updateLayer = true, updateCompression = true)
    }

    private fun initUiHandlers() {
        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?, view: View?, position: Int, id: Long
            ) {
                choiceHistory.set(model)
                updateChoices(updateModel = false, updateLayer = true, updateCompression = true)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        layerSeekBar.onSeekChangeListener = object : OnSeekChangeListener {
            override fun onSeeking(seekParams: SeekParams) {}

            override fun onStartTrackingTouch(seekBar: IndicatorSeekBar) {}

            override fun onStopTrackingTouch(seekBar: IndicatorSeekBar) {
                choiceHistory.set(model, layer)
                updateChoices(updateModel = false, updateLayer = false, updateCompression = true)
            }
        }

        compressionSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?, view: View?, position: Int, id: Long
            ) {
                choiceHistory.set(model, layer, compression)
                updateModelConfig()
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun updateChoices(
        updateModel: Boolean,
        updateLayer: Boolean,
        updateCompression: Boolean
    ) {
        if (updateModel) {
            val choices = modelChoices
            updateSpinner(modelSpinner, choices)

            val default = choiceHistory.get()
            modelSpinner.setSelection(choicePosition(choices, default))
        }

        if (updateLayer) {
            val choices = layerChoices
            layerSeekBar.tickCount = choices.count()
            layerSeekBar.customTickTexts(choices.toTypedArray())
            layerSeekBar.min = 0f
            layerSeekBar.max = choices.count() - 1f

            val default = choiceHistory.get(model)
            layerSeekBar.setProgress(choicePosition(choices, default).toFloat())
        }

        if (updateCompression) {
            val choices = compressionChoices
            updateSpinner(compressionSpinner, choices)

            val default = choiceHistory.get(model, layer)
            compressionSpinner.setSelection(choicePosition(choices, default))
        }

        updateModelConfig()
    }

    private fun updateModelConfig() {
        modelConfig = modelConfigMap.filter { it.key == model }.flatMap { it.value }
            .first { it.layer == layer && it.encoder == compression }
    }

    companion object {
        private fun <T> choicePosition(choices: List<T>, value: T?) =
            if (value == null) 0 else maxOf(choices.indexOf(value), 0)
    }
}

class ChoiceHistory {
    private val choiceHistory = lruMap<LinkedHashMap<String, String?>>()

    fun get() = choiceHistory.entries.lastOrNull()?.key

    fun get(model: String) = choiceHistory[model]?.entries?.lastOrNull()?.key

    fun get(model: String, layer: String) = choiceHistory[model]?.get(layer)

    fun set(model: String) {
        if (!choiceHistory.containsKey(model))
            choiceHistory[model] = lruMap()
    }

    fun set(model: String, layer: String) {
        set(model)
        if (!choiceHistory[model]!!.containsKey(layer))
            choiceHistory[model]!![layer] = null
    }

    fun set(model: String, layer: String, compression: String) {
        set(model, layer)
        choiceHistory[model]!![layer] = compression
    }

    companion object {
        private fun <T> lruMap() =
            LinkedHashMap<String, T>(0, 0.75f, true)
    }
}