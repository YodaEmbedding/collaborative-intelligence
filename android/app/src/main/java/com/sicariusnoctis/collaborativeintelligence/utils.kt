package com.sicariusnoctis.collaborativeintelligence

import android.R
import android.os.Environment
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Spinner
import com.sicariusnoctis.collaborativeintelligence.processor.FrameRequest
import io.reactivex.Flowable
import kotlinx.serialization.UnstableDefault
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonObjectSerializer
import kotlinx.serialization.json.content
import java.io.File
import java.io.FileNotFoundException
import java.nio.file.Paths
import java.time.Instant

@UnstableDefault
fun loadJsonFromDefaultFolder(filename: String): JsonObject? {
    val folderRoot = "collaborative-intelligence"
    val sdcard = Environment.getExternalStorageDirectory().toString()
    val parent = Paths.get(sdcard, folderRoot).toString()
    return try {
        val inputStream = File(parent, filename)
        val jsonString = inputStream.bufferedReader().use { it.readText() }
        Json.parse(JsonObjectSerializer, jsonString)
    } catch (e: FileNotFoundException) {
        null
    }
}

@UnstableDefault
fun loadModelConfigMap(filename: String): LinkedHashMap<String, List<ModelConfig>> {
    return loadJsonFromDefaultFolder(filename)!!.map { (k, v) ->
        k to v.jsonArray.map { x -> jsonToModelConfig(x.jsonObject, k) }
    }.toMap() as LinkedHashMap
}

// TODO use ModelConfig.serializer() directly...
private fun jsonToModelConfig(jsonObject: JsonObject, model: String? = null) = ModelConfig(
    model = model ?: jsonObject["model"]!!.content,
    layer = jsonObject["layer"]!!.content,
    encoder = jsonObject["encoder"]!!.content,
    decoder = jsonObject["decoder"]!!.content,
    encoder_args = jsonObject["encoder_args"]?.jsonObject,
    decoder_args = jsonObject["decoder_args"]?.jsonObject
)

fun <R> timed(
    func: () -> R
): Triple<R, Instant, Instant> {
    val start = Instant.now()
    val result = func()
    val end = Instant.now()
    return Triple(result, start, end)
}

fun <T> Flowable<T>.onBackpressureLimitRate(
    onDrop: (T) -> Unit,
    limit: (T) -> Boolean
): Flowable<T> {
    return this
        // .onBackpressureDrop(onDrop)
        .filter {
            if (limit(it)) {
                true
            } else {
                onDrop(it)
                false
            }
        }
}

fun <T> Flowable<FrameRequest<T>>.doOnNextTimed(
    statistics: Statistics,
    timeFunc: (ModelStatistics, Int, Instant, Instant) -> Unit,
    onNext: (FrameRequest<T>) -> Unit
): Flowable<FrameRequest<T>> {
    return this.doOnNext { x ->
        val (_, start, end) = timed { onNext(x) }
        timeFunc(statistics[x.info.modelConfig], x.info.frameNumber, start, end)
    }
}

fun <R, T> Flowable<FrameRequest<T>>.mapTimed(
    statistics: Statistics,
    timeFunc: (ModelStatistics, Int, Instant, Instant) -> Unit,
    mapper: (FrameRequest<T>) -> FrameRequest<R>
): Flowable<FrameRequest<R>> {
    return this.map { x ->
        val (result, start, end) = timed { mapper(x) }
        timeFunc(statistics[x.info.modelConfig], x.info.frameNumber, start, end)
        result
    }
}

fun updateSpinner(spinner: Spinner, choices: List<String>) {
    spinner.textAlignment = View.TEXT_ALIGNMENT_TEXT_END
    val adapter = ArrayAdapter<String>(
        spinner.context, R.layout.simple_spinner_item, choices
    )
    adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
    spinner.adapter = adapter
}
