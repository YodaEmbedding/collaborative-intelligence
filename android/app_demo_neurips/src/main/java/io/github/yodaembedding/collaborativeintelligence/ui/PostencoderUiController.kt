package io.github.yodaembedding.collaborativeintelligence.ui

import android.view.View
import android.widget.AdapterView
import android.widget.Spinner
import io.github.yodaembedding.collaborativeintelligence.ModelConfig
import io.github.yodaembedding.collaborativeintelligence.PostencoderConfig
import io.github.yodaembedding.collaborativeintelligence.loadModelConfigMap
import io.github.yodaembedding.collaborativeintelligence.updateSpinner
import com.warkiz.widget.IndicatorSeekBar
import com.warkiz.widget.OnSeekChangeListener
import com.warkiz.widget.SeekParams
import io.reactivex.Observable
import io.reactivex.disposables.CompositeDisposable
import io.reactivex.subjects.PublishSubject

// TODO history
class PostencoderUiController(
    modelConfig: ModelConfig,
    modelConfigEvents: Observable<ModelConfig>,
    private val postencoderSpinner: Spinner,
    private val postencoderQualitySeekBar: IndicatorSeekBar
) {
    var postencoderConfig: PostencoderConfig
        @Synchronized get() = _postencoderConfig
        private set(value) {
            _postencoderConfig = value
            postencoderConfigEvents.onNext(value)
        }

    val postencoderConfigEvents = PublishSubject.create<PostencoderConfig>()

    private lateinit var _postencoderConfig: PostencoderConfig
    private val postencoderConfigMap = makeConfigMap()
    private val subscriptions = CompositeDisposable()

    private val type
        get() = postencoderSpinner.getItemAtPosition(
            postencoderSpinner.selectedItemPosition
        ).toString()

    private val quality
        get() = postencoderQualitySeekBar.progress

    init {
        initUiHandlers()
        updateChoices(modelConfig)
        subscriptions.add(modelConfigEvents.subscribe { updateChoices(it) })
    }

    private fun initUiHandlers() {
        postencoderSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?, view: View?, position: Int, id: Long
            ) {
                updatePostencoderConfig()
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        postencoderQualitySeekBar.onSeekChangeListener = object : OnSeekChangeListener {
            override fun onSeeking(seekParams: SeekParams) {}

            override fun onStartTrackingTouch(seekBar: IndicatorSeekBar) {}

            override fun onStopTrackingTouch(seekBar: IndicatorSeekBar) {
                updatePostencoderConfig()
            }
        }
    }

    private fun updateChoices(modelConfig: ModelConfig) {
        require(postencoderConfigMap.containsKey(modelConfig)) { "ModelConfig not found" }
        val choices = postencoderConfigMap[modelConfig]!!
        updateSpinner(postencoderSpinner, choices)
        updatePostencoderConfig()
    }

    private fun updatePostencoderConfig() {
        postencoderConfig = PostencoderConfig(type, quality)
    }

    private fun makeConfigMap() = loadModelConfigMap("models.json").flatMap { it.value }.map {
        it to when (it.layer) {
            "client" -> listOf("None")
            "server" -> listOf("jpeg", "None")
            else -> when (it.encoder) {
                "UniformQuantizationU8Encoder" -> listOf("jpeg", "None")
                else -> listOf("None")
            }
        }
    }.toMap()
}
