package io.github.yodaembedding.collaborativeintelligence

import kotlinx.serialization.PolymorphicSerializer
import kotlinx.serialization.json.JSON
import kotlinx.serialization.json.Json
import kotlinx.serialization.modules.SerializersModule
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test

class SerializationTest {
    @Test
    fun jsonDeserialization() {
        val jsonSerializer = Json(
            context = SerializersModule {
                polymorphic(Response::class) {
                    ResultResponse::class with ResultResponse.serializer()
                    ConfirmationResponse::class with ConfirmationResponse.serializer()
                }
            }
        )

        val s = """{"type": "result", "frameNumber": 0, "inferenceTime": 30, "predictions": [""" +
                """{"name": "n01930112", "description": "nematode", "score": 0.5}, """ +
                """{"name": "n03729826", "description": "matchstick", "score": 0.25}, """ +
                """{"name": "n04286575", "description": "spotlight", "score": 0.125}""" +
                """]}"""

        val expected = ResultResponse(
            0, 30, listOf(
                Prediction("n01930112", "nematode", 0.5f),
                Prediction("n03729826", "matchstick", 0.25f),
                Prediction("n04286575", "spotlight", 0.125f)
            )
        )

        // val s2 = jsonSerializer.stringify(PolymorphicSerializer(Response::class), expected)
        val notExpected = ConfirmationResponse(0, 42)
        val actual = jsonSerializer.parse(PolymorphicSerializer(Response::class), s)

        assertEquals(expected, actual)
        assertNotEquals(notExpected, actual)
    }

    @Test
    fun modelConfig() {
        val s = """{"model": "resnet18", "layer": "client", "encoder": "None", "decoder": "None"}"""
        val mc = ModelConfig(
            model = "resnet18",
            layer = "client",
            encoder = "None",
            decoder = "None",
            encoder_args = null,
            decoder_args = null
        )

        assertEquals(mc, JSON.parse(ModelConfig.serializer(), s))
        // assertEquals(s, JSON.stringify(ModelConfig.serializer(), mc))
    }
}
