import tensorflow as tf
import numpy as np
import os

# --- Configuración ---
# Ruta al modelo TFLite original (Float32)
FP32_MODEL_PATH = os.path.join('app', 'src', 'main', 'assets', 'custom_model.tflite')

# Ruta donde se guardará el nuevo modelo cuantizado (INT8)
INT8_MODEL_PATH = os.path.join('app', 'src', 'main', 'assets', 'custom_model_int8.tflite')

# Ruta al archivo de datos de calibración
CALIBRATION_DATA_PATH = 'calibration_image_sample_data_20x128x128x3_float32.npy'

# --- Función principal de cuantización ---
def quantize_model():
    """
    Carga un modelo TFLite FP32, aplica cuantización INT8 completa utilizando
    un dataset de calibración y guarda el modelo cuantizado.
    """
    if not os.path.exists(FP32_MODEL_PATH):
        print(f"Error: No se encontró el modelo FP32 en '{FP32_MODEL_PATH}'")
        print("Asegúrate de exportar tu modelo a formato TFLite (Float32) primero.")
        return

    if not os.path.exists(CALIBRATION_DATA_PATH):
        print(f"Error: No se encontró el archivo de datos de calibración en '{CALIBRATION_DATA_PATH}'")
        return

    print("Cargando datos de calibración...")
    try:
        calibration_data = np.load(CALIBRATION_DATA_PATH)
        print(f"Datos de calibración cargados. Forma: {calibration_data.shape}")
    except Exception as e:
        print(f"Error al cargar los datos de calibración: {e}")
        return

    # Generador para el dataset de calibración
    def representative_dataset_gen():
        for i in range(calibration_data.shape[0]):
            # El input debe ser una lista de tensores
            yield [calibration_data[i:i+1]]

    print("Cargando el modelo FP32 para cuantización...")
    # Enfoque moderno y robusto para TF2.x
    # Cargamos el modelo como un módulo de TF
    try:
        # Este es un workaround para cargar un .tflite y reconvertirlo
        # Primero, obtenemos la firma del modelo cargándolo con el intérprete
        interpreter = tf.lite.Interpreter(model_path=FP32_MODEL_PATH)
        interpreter.allocate_tensors()
        # Obtenemos la función de inferencia
        infer = interpreter.get_signature_runner()

        # Creamos un módulo "envoltorio" para que el conversor lo pueda usar
        class TFLiteModule(tf.Module):
            def __init__(self, infer_func):
                super(TFLiteModule, self).__init__()
                self.infer = infer_func

            @tf.function(input_signature=[tf.TensorSpec(shape=[1, 128, 128, 3], dtype=tf.float32)])
            def __call__(self, x):
                return self.infer(x)

        module = TFLiteModule(infer)
        
        # Ahora sí, inicializamos el conversor desde el módulo
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [module.__call__.get_concrete_function()]
        )

    except Exception as e:
        print(f"Error al cargar el modelo o crear el conversor: {e}")
        return

    print("Configurando optimizaciones para cuantización INT8...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Forzar que tanto la entrada como la salida sean INT8 para máxima compatibilidad con hardware
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Iniciando el proceso de cuantización (puede tardar unos minutos)...")
    try:
        tflite_quant_model = converter.convert()
        print("¡Cuantización completada con éxito!")
    except Exception as e:
        print(f"Error durante la conversión: {e}")
        return

    print(f"Guardando el modelo cuantizado en: {INT8_MODEL_PATH}")
    with open(INT8_MODEL_PATH, 'wb') as f:
        f.write(tflite_quant_model)

    original_size = os.path.getsize(FP32_MODEL_PATH) / (1024 * 1024)
    quantized_size = len(tflite_quant_model) / (1024 * 1024)

    print("\n--- Resumen ---")
    print(f"Tamaño del modelo original (FP32): {original_size:.2f} MB")
    print(f"Tamaño del modelo cuantizado (INT8): {quantized_size:.2f} MB")
    print(f"Reducción de tamaño: {((original_size - quantized_size) / original_size) * 100:.2f}%")
    print("\nProceso finalizado.")

if __name__ == '__main__':
    quantize_model()