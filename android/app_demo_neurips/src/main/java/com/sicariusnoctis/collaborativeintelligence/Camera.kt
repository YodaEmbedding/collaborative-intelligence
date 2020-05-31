package com.sicariusnoctis.collaborativeintelligence

import android.content.Context
import android.util.Log
import io.fotoapparat.Fotoapparat
import io.fotoapparat.configuration.CameraConfiguration
import io.fotoapparat.log.logcat
import io.fotoapparat.log.loggers
import io.fotoapparat.parameter.FpsRange
import io.fotoapparat.parameter.Resolution
import io.fotoapparat.parameter.ScaleType
import io.fotoapparat.selector.*
import io.fotoapparat.util.FrameProcessor
import io.fotoapparat.view.CameraView

class Camera(context: Context, cameraView: CameraView, frameProcessor: FrameProcessor) {
    private val TAG = Camera::class.qualifiedName

    private val fotoapparat: Fotoapparat

    init {
        val cameraConfiguration = CameraConfiguration(
            pictureResolution = highestResolution(),
            previewResolution = ::selectPreviewResolution,
            previewFpsRange = ::selectFpsRange,
            focusMode = firstAvailable(
                continuousFocusPicture(),
                continuousFocusVideo(),
                autoFocus(),
                fixed()
            ),
            antiBandingMode = firstAvailable(
                auto(),
                hz50(),
                hz60(),
                none()
            ),
            jpegQuality = manualJpegQuality(90),
            sensorSensitivity = lowestSensorSensitivity(),
            frameProcessor = frameProcessor
        )

        fotoapparat = Fotoapparat(
            context = context,
            view = cameraView,
            scaleType = ScaleType.CenterCrop,
            lensPosition = back(),
            cameraConfiguration = cameraConfiguration,
            logger = loggers(
                logcat()
            ),
            cameraErrorCallback = { error -> Log.e(TAG, "$error") }
        )
    }

    fun start() {
        fotoapparat.start()
    }

    fun stop() {
        fotoapparat.stop()
    }

    fun getCurrentParameters() = fotoapparat.getCurrentParameters()

    companion object {
        @JvmStatic
        private fun selectPreviewResolution(iterable: Iterable<Resolution>): Resolution? = iterable
            .sortedBy { it.area }
            .firstOrNull { it.width >= 224 * 2 && it.height >= 224 * 2 }

        @JvmStatic
        private fun selectFpsRange(iterable: Iterable<FpsRange>): FpsRange? = iterable
            .sortedBy { it.min }
            .firstOrNull { it.min >= 5000 }
    }
}