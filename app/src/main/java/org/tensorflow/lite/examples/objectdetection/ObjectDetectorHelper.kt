/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetection
import org.tensorflow.lite.task.core.BaseOptions

import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetector

import org.tensorflow.lite.examples.objectdetection.detectors.CustomYoloDetector
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions


/**
 * Clase auxiliar para gestionar la detección de objetos con TensorFlow Lite.
 *
 * Esta clase se encarga de inicializar el detector de objetos (en este caso, `CustomYoloDetector`),
 * procesar las imágenes de entrada y devolver los resultados de la detección a través de un listener.
 *
 * @param threshold Umbral de confianza mínimo para mostrar una detección.
 * @param numThreads Número de hilos a utilizar para la inferencia.
 * @param maxResults Número máximo de resultados de detección a mostrar.
 * @param currentDelegate Delegado de TFLite a utilizar (CPU, GPU, o NNAPI).
 * @param context Contexto de la aplicación Android.
 * @param objectDetectorListener Listener para notificar los resultados o errores.
 */
class ObjectDetectorHelper(
  var threshold: Float = 0.5f,
  var numThreads: Int = 2,
  var maxResults: Int = 3,
  var currentDelegate: Int = 0,
  val context: Context,
  val objectDetectorListener: DetectorListener?
) {

    // For this example this needs to be a var so it can be reset on changes. If the ObjectDetector
    // will not change, a lazy val would be preferable.
    private var objectDetector: ObjectDetector? = null

    init {
        setupObjectDetector()
    }

    /**
     * Limpia el detector de objetos, liberando los recursos.
     * Es importante llamar a este método cuando el detector ya no se necesita.
     */
    fun clearObjectDetector() {
        objectDetector = null
    }

    /**
     * Cierra el detector de objetos y libera sus recursos.
     * Debe llamarse cuando el detector ya no se va a utilizar.
     */
    fun close() {
        objectDetector?.close()
        objectDetector = null
    }


    // Initialize the object detector using current settings on the
    // thread that is using it. CPU and NNAPI delegates can be used with detectors
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the detector
    /**
     * Inicializa el detector de objetos (`CustomYoloDetector`) con la configuración actual.
     * Este método configura el detector con los parámetros definidos en la clase
     * (umbral, hilos, etc.) y lo prepara para la inferencia.
     *
     * En caso de error durante la inicialización, se notifica al `objectDetectorListener`.
     */
    fun setupObjectDetector() {
        try {
            objectDetector = CustomYoloDetector(
                confidenceThreshold = threshold,
                iouThreshold = 0.3f, // Puedes ajustar este valor si es necesario
                numThreads = numThreads,
                maxResults = maxResults,
                currentDelegate = currentDelegate,
                context = context,
            )
        } catch (e: Exception) {
            objectDetectorListener?.onError(e.toString())
        }
    }


    /**
     * Ejecuta la detección de objetos en un `Bitmap` de entrada.
     *
     * @param image El `Bitmap` de la imagen a procesar.
     * @param imageRotation La rotación de la imagen en grados, para corregir la orientación.
     */
    fun detect(image: Bitmap, imageRotation: Int) {


        // Create preprocessor for the image.
        // See https://www.tensorflow.org/lite/inference_with_metadata/lite_support#imageprocessor_architecture

        val imageProcessor =
            ImageProcessor.Builder()
                .add(Rot90Op(-imageRotation / 90))
                .build()

        // Preprocess the image and convert it into a TensorImage for detection.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        // Inference time is the difference between the system time at the start and finish of the
        // process
        var inferenceTime = SystemClock.uptimeMillis()

        val results = objectDetector?.detect(tensorImage, imageRotation)

        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        if (results != null) {
            objectDetectorListener?.onResults(
                results.detections,
                inferenceTime,
                results.image.height,
                results.image.width
            )
        }

    }

    /**
     * Interfaz para comunicar los resultados de la detección.
     * Las clases que implementen esta interfaz podrán recibir notificaciones
     * sobre los resultados de la detección o los errores que ocurran.
     */
    interface DetectorListener {
        /**
         * Se llama cuando ocurre un error durante la inicialización o la detección.
         * @param error Mensaje de error.
         */
        fun onError(error: String)

        /**
         * Se llama cuando la detección se completa con éxito.
         * @param results Lista de detecciones encontradas.
         * @param inferenceTime Tiempo que tardó la inferencia, en milisegundos.
         * @param imageHeight Altura de la imagen procesada.
         * @param imageWidth Ancho de la imagen procesada.
         */
        fun onResults(
            results: List<ObjectDetection>,
            inferenceTime: Long,
            imageHeight: Int,
            imageWidth: Int
        )
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        // Ya no necesitamos constantes de modelo, la app usará solo el tuyo.

        /**
         * Calcula la Intersección sobre Unión (IoU) entre dos cajas delimitadoras.
         *
         * @param box1 La primera caja delimitadora (p. ej., la predicha).
         * @param box2 La segunda caja delimitadora (p. ej., la real o ground truth).
         * @return El valor de IoU, un flotante entre 0.0 y 1.0.
         */
        fun calculateIoU(box1: android.graphics.RectF, box2: android.graphics.RectF): Float {
            val xA = maxOf(box1.left, box2.left)
            val yA = maxOf(box1.top, box2.top)
            val xB = minOf(box1.right, box2.right)
            val yB = minOf(box1.bottom, box2.bottom)

            // Calcula el área de la intersección
            val intersectionArea = maxOf(0f, xB - xA) * maxOf(0f, yB - yA)

            // Calcula el área de ambas cajas
            val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
            val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)

            // Calcula el área de la unión
            val unionArea = box1Area + box2Area - intersectionArea

            // Calcula el IoU
            return if (unionArea > 0) intersectionArea / unionArea else 0f
        }
    }
}
