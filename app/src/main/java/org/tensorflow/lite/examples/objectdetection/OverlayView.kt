/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetection
import java.util.LinkedList

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<ObjectDetection> = LinkedList<ObjectDetection>()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()
    private var groundTruthBoxPaint = Paint()

    private var imageWidth: Int = 1
    private var imageHeight: Int = 1
    private var coordinateTransformer: CoordinateTransformer? = null

    private var bounds = Rect()

    init {
        initPaints()
    }

    fun clear() {
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 60f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 60f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 12F
        boxPaint.style = Paint.Style.STROKE

        groundTruthBoxPaint.color = Color.GREEN
        groundTruthBoxPaint.strokeWidth = 12F
        groundTruthBoxPaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        val transformer = coordinateTransformer ?: return

        // Ground truth simulado en el espacio de coordenadas del input del modelo (ej. 640x640)
        val groundTruthBox = RectF(150f, 150f, 400f, 400f)

        // Dibuja la caja de ground truth para depuración
        val mappedGroundTruthBox = transformer.transform(groundTruthBox)
        canvas.drawRect(mappedGroundTruthBox, groundTruthBoxPaint)

        for (result in results) {
            val mappedBoundingBox = transformer.transform(result.boundingBox)
            canvas.drawRect(mappedBoundingBox, boxPaint)

            val iou = ObjectDetectorHelper.calculateIoU(mappedBoundingBox, mappedGroundTruthBox)
            val iouPercentage = (iou * 100).toInt()

            val percentage = (result.category.confidence * 100).toInt()
            val drawableText =
                "${result.category.label} | Confianza: ${percentage}% | IoU: ${iouPercentage}%"

            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                mappedBoundingBox.left,
                mappedBoundingBox.top,
                mappedBoundingBox.left + textWidth + Companion.BOUNDING_RECT_TEXT_PADDING,
                mappedBoundingBox.top + textHeight + Companion.BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            canvas.drawText(
                drawableText,
                mappedBoundingBox.left,
                mappedBoundingBox.top + bounds.height(),
                textPaint
            )
        }
    }

    fun setResults(
        detectionResults: List<ObjectDetection>,
        imageHeight: Int,
        imageWidth: Int,
    ) {
        results = detectionResults
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight

        coordinateTransformer = CoordinateTransformer(imageWidth, imageHeight, width, height)
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}
