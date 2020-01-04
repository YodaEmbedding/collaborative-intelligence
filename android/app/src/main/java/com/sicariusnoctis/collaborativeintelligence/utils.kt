package com.sicariusnoctis.collaborativeintelligence

import android.os.Environment
import io.reactivex.Flowable
import kotlinx.serialization.UnstableDefault
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonObjectSerializer
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
