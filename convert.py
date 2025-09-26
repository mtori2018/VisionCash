import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

# Cargar el modelo ONNX
onnx_model_path = "app/train/weights/best.onnx"
tflite_model_path = "app/train/weights/best.flite"

print(f"Cargando modelo ONNX desde: {onnx_model_path}")
onnx_model = onnx.load(onnx_model_path)

# Preparar el backend de TensorFlow
# Esto convierte el modelo ONNX a un formato compatible con TensorFlow
print("Preparando backend de TensorFlow...")
tf_rep = prepare(onnx_model)

# Exportar a SavedModel (un paso intermedio necesario)
print("Exportando a formato SavedModel...")
saved_model_dir = "saved_model"
tf_rep.export_graph(saved_model_dir)

# Convertir el SavedModel a TensorFlow Lite
print(f"Convirtiendo SavedModel a TFLite: {tflite_model_path}")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # Opcional: optimiza el tamaño
tflite_model = converter.convert()

# Guardar el modelo TFLite
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"¡Éxito! Modelo guardado en: {tflite_model_path}")
