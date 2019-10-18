package com.sicariusnoctis.collaborativeintelligence

import android.content.Context
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Spinner
import com.warkiz.widget.IndicatorSeekBar
import com.warkiz.widget.OnSeekChangeListener
import com.warkiz.widget.SeekParams
import kotlinx.serialization.UnstableDefault
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonObjectSerializer
import kotlinx.serialization.json.content

class UiController(
    context: Context,
    private val modelSpinner: Spinner,
    private val layerSeekBar: IndicatorSeekBar,
    private val compressionSpinner: Spinner
) {
    val modelConfig: ModelConfig
        @Synchronized get() = _modelConfig

    private lateinit var _modelConfig: ModelConfig
    private val modelConfigMap = loadModelConfigs(context, "models.json")
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
        updateModelConfig()
    }

    private fun initUiHandlers() {
        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                choiceHistory.set(model)
                updateChoices(updateModel = false, updateLayer = true, updateCompression = true)
                updateModelConfig()
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        layerSeekBar.onSeekChangeListener = object : OnSeekChangeListener {
            override fun onSeeking(seekParams: SeekParams) {}

            override fun onStartTrackingTouch(seekBar: IndicatorSeekBar) {}

            override fun onStopTrackingTouch(seekBar: IndicatorSeekBar) {
                choiceHistory.set(model, layer)
                updateChoices(updateModel = false, updateLayer = false, updateCompression = true)
                updateModelConfig()
            }
        }

        compressionSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
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
            modelSpinner.adapter = ArrayAdapter<String>(
                modelSpinner.context,
                android.R.layout.simple_spinner_dropdown_item,
                choices
            )

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
            compressionSpinner.adapter = ArrayAdapter<String>(
                compressionSpinner.context,
                android.R.layout.simple_spinner_dropdown_item,
                choices
            )

            val default = choiceHistory.get(model, layer)
            compressionSpinner.setSelection(choicePosition(choices, default))
        }
    }

    private fun updateModelConfig() {
        _modelConfig = modelConfigMap.filter { it.key == model }.flatMap { it.value }
            .first { it.layer == layer && it.encoder == compression }
    }

    companion object {
        private fun <T> choicePosition(choices: List<T>, value: T?) =
            if (value == null) 0 else maxOf(choices.indexOf(value), 0)

        private fun jsonToModelConfig(jsonObject: JsonObject, model: String? = null) = ModelConfig(
            model = model ?: jsonObject["model"]!!.content,
            layer = jsonObject["layer"]!!.content,
            encoder = jsonObject["encoder"]!!.content,
            decoder = jsonObject["decoder"]!!.content,
            encoder_args = jsonObject["encoder_args"]?.jsonObject,
            decoder_args = jsonObject["decoder_args"]?.jsonObject
        )

        @UseExperimental(UnstableDefault::class)
        private fun loadConfig(context: Context, filename: String): JsonObject {
            val inputStream = context.assets.open(filename)
            val jsonString = inputStream.bufferedReader().use { it.readText() }
            return Json.parse(JsonObjectSerializer, jsonString)
        }

        private fun loadModelConfigs(
            context: Context,
            filename: String
        ): LinkedHashMap<String, List<ModelConfig>> {
            return loadConfig(context, filename).map { (k, v) ->
                k to v.jsonArray.map { x -> jsonToModelConfig(x.jsonObject, k) }
            }.toMap() as LinkedHashMap
        }
    }
}

class ChoiceHistory {
    private val choiceHistory = lruMap<LinkedHashMap<String, String?>>()

    // TODO why is it lastOrNull rather than firstOrNull? o_0
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