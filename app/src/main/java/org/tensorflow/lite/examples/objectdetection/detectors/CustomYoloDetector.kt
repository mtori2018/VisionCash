package org.tensorflow.lite.examples.objectdetection.detectors

import android.content.Context
import android.graphics.RectF
import com.ultralytics.yolo.ImageProcessing
import com.ultralytics.yolo.models.LocalYoloModel
import com.ultralytics.yolo.predict.detect.DetectedObject
import com.ultralytics.yolo.predict.detect.TfliteDetector
import org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper
import org.tensorflow.lite.support.image.TensorImage


/**
 * Implementación de un detector de objetos utilizando un modelo YOLO personalizado.
 *
 * Esta clase utiliza la librería `com.ultralytics.yolo` para cargar y ejecutar
 * un modelo YOLO en formato TFLite. Se encarga de la inicialización, el preprocesamiento
 * de la imagen, la predicción y el postprocesamiento de los resultados.
 *
 * @param confidenceThreshold Umbral de confianza para filtrar detecciones.
 * @param iouThreshold Umbral de IoU (Intersection over Union) para la supresión de no máximos.
 * @param numThreads Número de hilos para la inferencia.
 * @param maxResults No se utiliza directamente en esta implementación, pero se mantiene por compatibilidad.
 * @param currentDelegate Delegado de TFLite a utilizar (CPU o GPU).
 * @param context Contexto de la aplicación.
 */
class CustomYoloDetector(
    var confidenceThreshold: Float = 0.5f,
    var iouThreshold: Float = 0.3f,
    var numThreads: Int = 2,
    var maxResults: Int = 3,
    var currentDelegate: Int = 0,
    val context: Context
): ObjectDetector {

    private var yolo: TfliteDetector
    private var ip: ImageProcessing

    init {

        yolo = TfliteDetector(context)
        yolo.setIouThreshold(iouThreshold)
        yolo.setConfidenceThreshold(confidenceThreshold)

        // Apuntamos directamente a nuestro modelo personalizado
        val modelPath = "custom_model.tflite"
        // Apuntamos al archivo de metadatos correcto
        val metadataPath = "CustomYoloDetector.yaml"

        val config = LocalYoloModel(
            "detect",
            "tflite",
            modelPath,
            metadataPath,
        )

        val useGPU = currentDelegate == 0
        yolo.loadModel(
            config,
            useGPU
        )

        ip = ImageProcessing()

    }

    /**
     * Realiza la detección de objetos en la imagen proporcionada.
     *
     * Este método toma una `TensorImage`, la convierte a `Bitmap`, la preprocesa,
     * ejecuta la inferencia con el modelo YOLO y finalmente postprocesa los resultados
     * para convertirlos en una lista de `ObjectDetection`.
     *
     * @param image La imagen de entrada como `TensorImage`.
     * @param imageRotation La rotación de la imagen, utilizada para ajustar las coordenadas de los cuadros.
     * @return Un objeto `DetectionResult` que contiene la lista de detecciones y la imagen procesada.
     */
    override fun detect(image: TensorImage, imageRotation: Int): DetectionResult  {

        val bitmap = image.bitmap

        val ppImage = yolo.preprocess(bitmap)
        val results = yolo.predict(ppImage)

        val detections = ArrayList<ObjectDetection>()

        // ASPECT_RATIO = 4:3
        // => imgW = imgH * 3/4
        var imgH: Int
        var imgW: Int
        if (imageRotation == 90 || imageRotation == 270) {
            imgH = ppImage.height
            imgW = imgH * 3 / 4
        }
        else {
            imgW = ppImage.width
            imgH = imgW * 3 / 4

        }


        for (result: DetectedObject in results) {
            val category = Category(
                result.label,
                result.confidence,
            )
            val yoloBox = result.boundingBox

            val left = yoloBox.left * imgW
            val top = yoloBox.top * imgH
            val right = yoloBox.right * imgW
            val bottom = yoloBox.bottom * imgH

            val bbox = RectF(
                left,
                top,
                right,
                bottom
            )
            val detection = ObjectDetection(
                bbox,
                category
            )
            detections.add(detection)
        }

        val ret = DetectionResult(ppImage, detections)
        ret.info = yolo.stats
        return ret

    }


}