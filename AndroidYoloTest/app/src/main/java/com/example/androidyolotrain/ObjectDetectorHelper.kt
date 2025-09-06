package com.example.androidyolotrain

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import android.util.Log // Importar Log
import java.nio.FloatBuffer
import java.util.Collections

class ObjectDetectorHelper(
    private val context: Context,
    private val objectDetectorListener: DetectorListener?
) {

    private var ortSession: OrtSession? = null
    private var env: OrtEnvironment? = null

    init {
        setupObjectDetector()
    }

    private fun setupObjectDetector() {
        try {
            env = OrtEnvironment.getEnvironment()
            val modelBytes = context.assets.open("best.onnx").readBytes()
            ortSession = env?.createSession(modelBytes, OrtSession.SessionOptions())
            Log.i("ObjectDetectorHelper", "ONNX Runtime initialized successfully.")
        } catch (e: Exception) {
            val errorMessage = "Failed to initialize ONNX Runtime: ${e.message}"
            Log.e("ObjectDetectorHelper", errorMessage, e)
            objectDetectorListener?.onError(errorMessage)
        }
    }

    fun detect(image: Bitmap) {
        if (ortSession == null || env == null) {
            objectDetectorListener?.onError("ONNX session not initialized.")
            return
        }

        val (inputTensor, ratioX, ratioY) = preprocess(image)
        val inputName = ortSession?.inputNames?.iterator()?.next()
        val inputs = Collections.singletonMap(inputName, inputTensor)

        try {
            val results = ortSession?.run(inputs)
            val outputTensor = results?.get(0) as OnnxTensor
            val detections = postprocess(outputTensor, ratioX, ratioY)
            objectDetectorListener?.onResults(detections)
            Log.i("ObjectDetectorHelper", "Inference completed successfully.")
        } catch (e: Exception) {
            val errorMessage = "Error during inference: ${e.message}"
            Log.e("ObjectDetectorHelper", errorMessage, e)
            objectDetectorListener?.onError(errorMessage)
        }
    }

    private fun preprocess(bitmap: Bitmap): Triple<OnnxTensor, Float, Float> {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
        val ratioX = bitmap.width.toFloat() / 640f
        val ratioY = bitmap.height.toFloat() / 640f

        val floatBuffer = FloatBuffer.allocate(3 * 640 * 640)
        floatBuffer.rewind()

        val pixels = IntArray(640 * 640)
        resizedBitmap.getPixels(pixels, 0, 640, 0, 0, 640, 640)

        for (i in 0 until 640 * 640) {
            val pixel = pixels[i]
            floatBuffer.put(((pixel shr 16) and 0xFF) / 255.0f)
            floatBuffer.put(((pixel shr 8) and 0xFF) / 255.0f)
            floatBuffer.put((pixel and 0xFF) / 255.0f)
        }
        floatBuffer.rewind()

        val shape = longArrayOf(1, 3, 640, 640)
        return Triple(OnnxTensor.createTensor(env, floatBuffer, shape), ratioX, ratioY)
    }

    private fun postprocess(outputTensor: OnnxTensor, ratioX: Float, ratioY: Float): List<DetectionResult> {
        val outputBuffer = outputTensor.floatBuffer
        val shape = outputTensor.info.shape // Shape is [1, num_classes + 4, num_detections]
        val numDetections = shape[2].toInt()
        val numClasses = shape[1].toInt() - 4

        val detections = mutableListOf<DetectionResult>()
        // Transpose the output buffer for easier processing
        val transposedBuffer = FloatArray(shape[1].toInt() * numDetections)
        outputBuffer.get(transposedBuffer)
        
        val detectionsData = Array(numDetections) { FloatArray(shape[1].toInt()) }
        for (i in 0 until numDetections) {
            for (j in 0 until shape[1].toInt()) {
                detectionsData[i][j] = transposedBuffer[j * numDetections + i]
            }
        }

        for (i in 0 until numDetections) {
            val detection = detectionsData[i]
            val scores = detection.sliceArray(4 until shape[1].toInt())
            val maxScore = scores.maxOrNull() ?: 0f

            if (maxScore > 0.5f) { // Confidence threshold
                val cx = detection[0] * ratioX
                val cy = detection[1] * ratioY
                val w = detection[2] * ratioX
                val h = detection[3] * ratioY

                val x1 = cx - w / 2
                val y1 = cy - h / 2
                val x2 = cx + w / 2
                val y2 = cy + h / 2

                detections.add(DetectionResult(RectF(x1, y1, x2, y2), "billete", maxScore))
            }
        }
        return detections
    }

    fun clearObjectDetector() {
        ortSession?.close()
        env?.close()
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(results: List<DetectionResult>)
    }

    data class DetectionResult(val boundingBox: RectF, val label: String, val score: Float)
}