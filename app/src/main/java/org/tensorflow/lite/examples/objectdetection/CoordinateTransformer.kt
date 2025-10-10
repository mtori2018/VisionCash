package org.tensorflow.lite.examples.objectdetection

import android.graphics.RectF
import kotlin.math.max

/**
 * Clase de utilidad para transformar coordenadas entre diferentes sistemas de referencia.
 *
 * Esta clase es esencial para mapear las coordenadas de las cajas delimitadoras
 * (bounding boxes) desde el sistema de coordenadas de la imagen de entrada del modelo
 * al sistema de coordenadas de la vista en la pantalla del dispositivo.
 *
 * Maneja el escalado y el "fillStart" (escalado para llenar la vista y alineación al inicio)
 * para asegurar que las detecciones se dibujen en la posición correcta.
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
    // Se usa max para asegurar que la imagen llene completamente la vista
    // (fillStart), permitiendo el recorte si la relación de aspecto no coincide.
    private val scale: Float = max(
        viewWidth.toFloat() / inputImageWidth,
        viewHeight.toFloat() / inputImageHeight
    )

    // Con fillStart, la imagen se alinea a la parte superior izquierda,
    // por lo que los offsets son 0.
    private val offsetX: Float = 0f
    private val offsetY: Float = 0f

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