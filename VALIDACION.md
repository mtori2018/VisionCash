# Plan de Pruebas y Validación: Detector de Billetes Chilenos

Este documento describe el plan de pruebas exhaustivas para el proyecto de detección de billetes chilenos utilizando un modelo YOLOv8 en una aplicación Android. El objetivo es evaluar la funcionalidad, fiabilidad, eficiencia y usabilidad del sistema bajo condiciones operativas realistas.

## 1. Análisis de Rendimiento del Modelo (Pruebas Cuantitativas)

Esta sección se basa en los datos generados durante el entrenamiento del modelo en la carpeta `DATASETS/runs/detect/train/`.

### 1.1. Métricas Clave de Entrenamiento

Se analizarán las siguientes métricas del archivo `results.csv` para evaluar el rendimiento del modelo en el conjunto de datos de validación:

- **Precisión (`metrics/precision(B)`)**: Mide la proporción de detecciones correctas.
- **Recall (`metrics/recall(B)`)**: Mide la proporción de billetes reales que fueron detectados.
- **mAP@0.50 (`metrics/mAP50(B)`)**: Mean Average Precision con un umbral de IoU de 0.50.
- **mAP@0.50-0.95 (`metrics/mAP50-95(B)`)**: Métrica más estricta, promediada sobre múltiples umbrales de IoU.

### 1.2. Evidencias Visuales

Se incluirán los siguientes gráficos generados por YOLOv8 para complementar el análisis cuantitativo:

- **`BoxPR_curve.png`**: Curva de Precisión-Recall.
- **`confusion_matrix.png`**: Matriz de confusión para analizar errores de clasificación.
- **`results.png`**: Gráficos de evolución de las métricas y pérdidas durante el entrenamiento.

## 2. Pruebas Funcionales y de Fiabilidad (Pruebas Cualitativas)

Estas pruebas se realizarán directamente en la aplicación Android para evaluar su comportamiento en escenarios de uso real.

| ID de Prueba | Descripción de la Prueba | Criterio de Aceptación | Resultado (Pasó/Falló) | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **PF-01** | Detección de billete de $1.000 | El billete es detectado correctamente con una confianza > 80%. | | |
| **PF-02** | Detección de billete de $2.000 | El billete es detectado correctamente con una confianza > 80%. | | |
| **PF-03** | Detección de billete de $5.000 | El billete es detectado correctamente con una confianza > 80%. | | |
| **PF-04** | Detección de billete de $10.000 | El billete es detectado correctamente con una confianza > 80%. | | |
| **PF-05** | Detección de billete de $20.000 | El billete es detectado correctamente con una confianza > 80%. | | |
| **PF-06** | Detección con baja iluminación | El sistema detecta billetes con una confianza aceptable (> 60%). | | |
| **PF-07** | Detección con luz solar directa | El sistema no se ve afectado por reflejos y detecta correctamente. | | |
| **PF-08** | Detección de billetes arrugados | El sistema es capaz de detectar billetes que no están en perfecto estado. | | |
| **PF-09** | Detección de billetes doblados | El sistema detecta billetes aunque una parte de ellos esté oculta. | | |
| **PF-10** | Detección con fondo complejo | El sistema aísla y detecta el billete correctamente sin confundirse con el fondo. | | |
| **PF-11** | Detección de múltiples billetes | El sistema detecta todos los billetes presentes en la imagen (`maxResults` = 3). | | |
| **PF-12** | Falsos positivos | El sistema no detecta billetes donde no los hay (apuntando a objetos aleatorios). | | |

## 3. Pruebas de Eficiencia y Rendimiento

Se medirá el rendimiento de la aplicación en el dispositivo para asegurar una experiencia de usuario fluida.

| ID de Prueba | Descripción de la Prueba | Métrica a Medir | Resultado | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **PE-01** | Tiempo de inferencia con CPU | `inferenceTime` promedio en ms. | | Se usará el delegado `DELEGATE_CPU`. |
| **PE-02** | Tiempo de inferencia con GPU | `inferenceTime` promedio en ms. | | Se usará el delegado `DELEGATE_GPU`. |
| **PE-03** | Tiempo de inferencia con NNAPI | `inferenceTime` promedio en ms. | | Se usará el delegado `DELEGATE_NNAPI`. |
| **PE-04** | Impacto de `numThreads` | Comparativa de `inferenceTime` con 2, 4 y 8 hilos en CPU. | | |
| **PE-05** | Uso de CPU y Memoria | Monitoreo del uso de recursos desde Android Studio Profiler. | | |

## 4. Pruebas de Usabilidad

Se evaluará la interacción del usuario con la aplicación.

| ID de Prueba | Descripción de la Prueba | Criterio de Aceptación | Resultado (Pasó/Falló) | Observaciones |
| :--- | :--- | :--- | :--- | :--- |
| **PU-01** | Fluidez de la cámara | El feed de la cámara no presenta lag o caídas de frames durante la detección. | | |
| **PU-02** | Claridad de la información | La caja delimitadora, la etiqueta del billete y el porcentaje de confianza son fáciles de leer. | | |
| **PU-03** | Respuesta de la interfaz | La aplicación responde rápidamente a las acciones del usuario (cambiar de modelo, etc.). | | |

Este plan servirá como base para la ejecución de las pruebas. A medida que las realices, puedes ir rellenando esta tabla para documentar los resultados.