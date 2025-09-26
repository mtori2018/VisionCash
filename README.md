# Detección de Objetos en Android con YOLO y TensorFlow Lite

Esta es una aplicación de ejemplo para Android que demuestra cómo utilizar un modelo de detección de objetos YOLO, convertido a formato TensorFlow Lite, para detectar objetos en tiempo real utilizando la cámara del dispositivo.

## 🌟 Características

- Detección de objetos en tiempo real desde la cámara.
- Implementación de un modelo YOLO personalizado (`CustomYoloDetector.kt`).
- Interfaz de usuario simple que muestra los cuadros delimitadores y las etiquetas de los objetos detectados.
- Configuración de parámetros como el umbral de confianza, número de hilos y máximo de resultados.
- Anuncios por voz (Text-to-Speech) para los objetos detectados con alta confianza.

## 🛠️ Tecnologías Utilizadas

- **Lenguaje:** Kotlin
- **Framework:** Android Nativo
- **Machine Learning:** TensorFlow Lite
- **Modelo de Detección:** YOLO (personalizado)
- **Build Tool:** Gradle

## 🚀 Configuración y Puesta en Marcha

Sigue estos pasos para configurar y ejecutar el proyecto en tu entorno de desarrollo.

### Prerrequisitos

- Android Studio (versión recomendada: Iguana | 2023.2.1 o superior)
- Un dispositivo Android físico o un emulador con acceso a la cámara.

### Pasos de Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/tu-repositorio.git
    ```
2.  **Abre el proyecto en Android Studio:**
    - Abre Android Studio.
    - Selecciona `Open` y navega hasta el directorio donde clonaste el repositorio.
3.  **Sincroniza el proyecto con Gradle:**
    - Android Studio debería sincronizar el proyecto automáticamente. Si no, haz clic en `File` > `Sync Project with Gradle Files`.
4.  **Ejecuta la aplicación:**
    - Conecta tu dispositivo Android o inicia un emulador.
    - Haz clic en el botón `Run 'app'` (icono de play verde).

### Nota sobre el Modelo Personalizado

Este proyecto está preconfigurado para usar un modelo YOLOv8 TFLite personalizado. Los archivos del modelo se encuentran en `app/src/main/assets/`:
- `custom_model.tflite`: El modelo de TensorFlow Lite.
- `CustomYoloDetector.yaml`: El archivo de metadatos que contiene las etiquetas de los objetos.

Si deseas utilizar tu propio modelo, simplemente reemplaza estos dos archivos con los tuyos.

## 📖 Uso

Una vez que la aplicación se inicie, otorga los permisos de cámara necesarios. La aplicación comenzará a procesar el video de la cámara en tiempo real y dibujará cuadros delimitadores alrededor de los objetos detectados.

En la parte inferior de la pantalla, puedes deslizar hacia arriba una hoja de configuración para ajustar:
- **Umbral (Threshold):** Aumenta o disminuye el umbral de confianza mínimo para que una detección sea válida.
- **Resultados Máx. (Max Results):** Define el número máximo de objetos que se pueden detectar simultáneamente.
- **Hilos (Threads):** Ajusta el número de hilos que TensorFlow Lite utilizará para la inferencia. Un mayor número puede acelerar la detección en CPUs multinúcleo, a costa de un mayor consumo de recursos.

## 📂 Estructura del Proyecto

```
.
├── app/
│   ├── src/main/
│   │   ├── assets/
│   │   │   ├── custom_model.tflite    # Modelo TFLite
│   │   │   └── CustomYoloDetector.yaml  # Metadatos del modelo
│   │   ├── java/org/tensorflow/lite/examples/objectdetection/
│   │   │   ├── detectors/
│   │   │   │   └── CustomYoloDetector.kt  # Lógica principal para el modelo YOLO
│   │   │   ├── fragments/
│   │   │   │   └── CameraFragment.kt      # Gestiona la cámara y la UI
│   │   │   └── ObjectDetectorHelper.kt    # Ayudante para la detección
│   │   └── ...
│   └── ...
└── ...
```

## 📄 Licencia

Este proyecto se distribuye bajo la Licencia Apache 2.0. Consulta el archivo `LICENSE` para más detalles.
