package com.sicariusnoctis.collaborativeintelligence

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

@Serializable
data class ModelEntry(
    val model: String,
    val layer: String,
    val encoder: String,
    val decoder: String,
    val encoder_args: JsonObject? = null,
    val decoder_args: JsonObject? = null,  // TODO this isn't REALLY needed...

    // TODO multiple outputs? each with a configurable encoder/decoder?
    // val outputs: {
    // }

    // val shape:
    val postencoder_args: JsonObject? = null
) {
    // fun toModelConfig(): ModelConfig { }

    // private fun dictString(name: String, args: JsonObject?): String =
    //     if (args == null) {
    //         name
    //     } else {
    //         "$name(${args.jsonObject.map { (k, v) -> "$k=$v" }.joinToString(", ")})"
    //     }
}
