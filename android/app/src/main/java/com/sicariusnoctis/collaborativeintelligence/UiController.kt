package com.sicariusnoctis.collaborativeintelligence

import android.content.Context
import android.util.Log
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
import kotlin.math.roundToInt

class UiController(
    context: Context,
    private val modelSpinner: Spinner,
    private val layerSeekBar: IndicatorSeekBar,
    private val compressionSpinner: Spinner
) {
    private val TAG = UiController::class.qualifiedName

    val modelConfig: ModelConfig
        @Synchronized get() = _modelConfig

    private lateinit var _modelConfig: ModelConfig
    private val modelConfigMap = loadModelConfigs(context, "models.json")
    private val model
        get() = modelSpinner.getItemAtPosition(modelSpinner.selectedItemPosition).toString()
    private val layer
        get() = when (layerChoices.size) {
            1 -> layerChoices.first()
            else -> layerChoices[(layerSeekBar.progressFloat / (layerSeekBar.max - layerSeekBar.min)).roundToInt()]
        }
    private val compression
        get() = compressionSpinner.getItemAtPosition(compressionSpinner.selectedItemPosition).toString()

    private val modelConfigs get() = modelConfigMap.filter { it.key == model }.flatMap { it.value }
    private val modelChoices get() = modelConfigMap.keys.toList()
    private val layerChoices get() = LinkedHashSet<String>(modelConfigs.map { it.layer }).toList()
    private val compressionChoices get() = modelConfigs.filter { it.layer == layer }.map { it.encoder }

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
                updateChoices(updateModel = false, updateLayer = true, updateCompression = true)
                updateModelConfig()
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        layerSeekBar.onSeekChangeListener = object : OnSeekChangeListener {
            override fun onSeeking(seekParams: SeekParams) {}

            override fun onStartTrackingTouch(seekBar: IndicatorSeekBar) {}

            override fun onStopTrackingTouch(seekBar: IndicatorSeekBar) {
                Log.i(TAG, layer)
                Log.i(TAG, seekBar.indicator.toString())
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
        // TODO keep hold onto position/options...? maybe for each model? but also when changing layers/etc...? idk? sure! TWO/nested saved states!

        if (updateModel) {
            modelSpinner.adapter = ArrayAdapter<String>(
                modelSpinner.context,
                android.R.layout.simple_spinner_dropdown_item,
                modelChoices
            )
        }

        if (updateLayer) {
            val layerChoicesCurrent = layerChoices
            layerSeekBar.tickCount = layerChoicesCurrent.count()
            layerSeekBar.customTickTexts(layerChoicesCurrent.toTypedArray())
        }

        if (updateCompression) {
            compressionSpinner.adapter = ArrayAdapter<String>(
                compressionSpinner.context,
                android.R.layout.simple_spinner_dropdown_item,
                compressionChoices
            )
        }
    }

    private fun updateModelConfig() {
        _modelConfig = modelConfigMap.filter { it.key == model }.flatMap { it.value }
            .first { it.layer == layer && it.encoder == compression }
    }

    companion object {
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