package org.tensorflow.lite.examples.objectdetection

import android.graphics.RectF
import kotlin.math.min

/**
 * Clase de utilidad para transformar coordenadas entre diferentes sistemas de referencia.
 *
 * Esta clase es esencial para mapear las coordenadas de las cajas delimitadoras
 * (bounding boxes) desde el sistema de coordenadas de la imagen de entrada del modelo
 * al sistema de coordenadas de la vista en la pantalla del dispositivo.
 *
 * Maneja el escalado y el "letterboxing" (las bandas negras que se añaden para
 * mantener la relación de aspecto) para asegurar que las detecciones se dibujen
 * en la posición correcta.
 *
 * @param inputImageWidth Ancho de la imagen de entrada del modelo.
 * @param inputImageHeight Alto de la imagen de entrada del modelo.
 * @param viewWidth Ancho de la vista de destino (ej. OverlayView).
 * @param viewHeight Alto de la vista de destino (ej. OverlayView).
 */
class CoordinateTransformer(
    private val inputImageWidth: Int,
    private val inputImageHeight: Int,
    private val viewWidth: Int,
    private val viewHeight: Int
) {

    // Calcula el factor de escala para la transformación.
    // Se usa min para asegurar que la imagen quepa completamente en la vista
    // manteniendo la relación de aspecto (letterboxing).
    private val scale: Float = min(
        viewWidth.toFloat() / inputImageWidth,
        viewHeight.toFloat() / inputImageHeight
    )

    // Calcula el desplazamiento en el eje X para centrar la imagen escalada.
    // Si la imagen es más ancha que la vista, no hay desplazamiento.
    private val offsetX: Float = (viewWidth - inputImageWidth * scale) / 2

    // Calcula el desplazamiento en el eje Y para centrar la imagen escalada.
    // Si la imagen es más alta que la vista, no hay desplazamiento.
    private val offsetY: Float = (viewHeight - inputImageHeight * scale) / 2

    /**
     * Transforma una caja delimitadora (RectF) desde el sistema de coordenadas
     * de la imagen de entrada al sistema de coordenadas de la vista.
     *
     * @param box La caja delimitadora en el sistema de coordenadas de entrada.
     * @return La caja delimitadora transformada al sistema de coordenadas de la vista.
     */
    fun transform(box: RectF): RectF {
        return RectF(
            box.left * scale + offsetX,
            box.top * scale + offsetY,
            box.right * scale + offsetX,
            box.bottom * scale + offsetY
        )
    }
}