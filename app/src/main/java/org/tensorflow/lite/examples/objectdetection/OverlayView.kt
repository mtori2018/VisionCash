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
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        val transformer = coordinateTransformer ?: return

        for (result in results) {
            val mappedBoundingBox = transformer.transform(result.boundingBox)
            canvas.drawRect(mappedBoundingBox, boxPaint)

            val percentage = (result.category.confidence * 100).toInt()
            val labelText = result.category.label
            val confidenceText = "Confianza: ${percentage}%"

            // Get text bounds for calculating background size
            textBackgroundPaint.getTextBounds(labelText, 0, labelText.length, bounds)
            val labelHeight = bounds.height()
            val labelWidth = bounds.width()

            textBackgroundPaint.getTextBounds(confidenceText, 0, confidenceText.length, bounds)
            val confidenceWidth = bounds.width()

            val textWidth = Math.max(labelWidth, confidenceWidth)
            val textHeight = labelHeight * 2 + Companion.BOUNDING_RECT_TEXT_PADDING

            // Draw background for text
            canvas.drawRect(
                mappedBoundingBox.left,
                mappedBoundingBox.top,
                mappedBoundingBox.left + textWidth + (Companion.BOUNDING_RECT_TEXT_PADDING * 2),
                mappedBoundingBox.top + textHeight,
                textBackgroundPaint
            )

            // Draw the label on the first line
            canvas.drawText(
                labelText,
                mappedBoundingBox.left + Companion.BOUNDING_RECT_TEXT_PADDING,
                mappedBoundingBox.top + labelHeight,
                textPaint
            )

            // Draw the confidence on the second line
            canvas.drawText(
                confidenceText,
                mappedBoundingBox.left + Companion.BOUNDING_RECT_TEXT_PADDING,
                mappedBoundingBox.top + labelHeight * 2,
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
