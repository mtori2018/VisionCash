## 4. Pruebas Iniciales y Resultados Preliminares (30%)

### a) Descripción

Esta sección detalla las pruebas iniciales realizadas sobre el modelo de detección entrenado, con el objetivo de validar los aspectos fundamentales de su diseño e identificar áreas que requieren ajustes o mejoras. Se presenta un análisis exhaustivo de las métricas de rendimiento obtenidas durante el proceso de entrenamiento, lo que permite una evaluación crítica de la capacidad del modelo para detectar objetos de interés en el conjunto de datos de validación.

### b) Evidencias Esperadas

Los resultados del entrenamiento del modelo de detección, generados en la ruta `DATASETS/runs/detect/train`, proporcionan una visión detallada de su rendimiento. A continuación, se presentan las métricas clave y un análisis de los resultados, demostrando cómo esta información ha guiado ajustes en el diseño e implementación del modelo.

**Configuración del Entrenamiento (extraído de `DATASETS/runs/detect/train/args.yaml`):**
*   **Modelo:** `yolov8n.pt`
*   **Épocas:** 100
*   **Tamaño de imagen:** 640
*   **Batch Size:** 32

**Métricas de Rendimiento (última época, extraído de `DATASETS/runs/detect/train/results.csv`):**

| Métrica                 | Valor (Época 100) |
| :---------------------- | :---------------- |
| `metrics/precision(B)`  | 0.99187           |
| `metrics/recall(B)`     | 0.98986           |
| `metrics/mAP50(B)`      | 0.99410           |
| `metrics/mAP50-95(B)`   | 0.99142           |
| `train/box_loss`        | 0.15191           |
| `train/cls_loss`        | 0.13024           |
| `train/dfl_loss`        | 0.86309           |
| `val/box_loss`          | 0.15189           |
| `val/cls_loss`          | 0.12686           |
| `val/dfl_loss`          | 0.77060           |

**Análisis de Resultados:**

*   **Precisión (Precision) y Recall:** Los valores de precisión (0.99187) y recall (0.98986) en la última época son excepcionalmente altos, indicando que el modelo es muy bueno tanto en la identificación correcta de objetos como en la minimización de falsos positivos y falsos negativos. Esto sugiere que el modelo tiene una gran capacidad para distinguir entre las clases y localizar los objetos con exactitud.

*   **F1-score:** Aunque no se proporciona directamente, con valores tan altos de precisión y recall, el F1-score (que es la media armónica de precisión y recall) también sería muy alto, confirmando un excelente equilibrio entre ambas métricas.

*   **mAP (mean Average Precision):** El mAP50 (0.99410) y mAP50-95 (0.99142) son métricas cruciales que evalúan la precisión promedio en diferentes umbrales de IoU (Intersection over Union). Un mAP50 de casi 0.994 indica que el modelo es extremadamente preciso en la detección de objetos con un umbral de IoU del 50%. El mAP50-95, que promedia el mAP en umbrales de IoU desde 50% hasta 95%, también es muy alto (0.99142), lo que demuestra la robustez del modelo incluso con requisitos de localización más estrictos.

*   **Curvas de Pérdida (Loss Curves):** Las pérdidas de entrenamiento (`train/box_loss`, `train/cls_loss`, `train/dfl_loss`) y de validación (`val/box_loss`, `val/cls_loss`, `val/dfl_loss`) muestran una tendencia decreciente a lo largo de las épocas, lo que es un indicador positivo de que el modelo está aprendiendo y convergiendo. Los valores finales de pérdida son bajos, lo que sugiere que el modelo ha minimizado los errores de localización (box_loss), clasificación (cls_loss) y distribución de características (dfl_loss). La cercanía entre las pérdidas de entrenamiento y validación al final del entrenamiento (por ejemplo, `train/box_loss` 0.15191 vs `val/box_loss` 0.15189) sugiere que el modelo no está sufriendo de un sobreajuste significativo.

**Evidencias Visuales:**

Las siguientes imágenes, generadas durante el entrenamiento, complementan el análisis numérico:

*   **Matriz de Confusión:**
    *   [`confusion_matrix.png`](DATASETS/runs/detect/train/confusion_matrix.png)
    *   [`confusion_matrix_normalized.png`](DATASETS/runs/detect/train/confusion_matrix_normalized.png)
    La matriz de confusión (normalizada y no normalizada) mostrará visualmente la proporción de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos para cada clase. Se espera que la diagonal principal tenga valores cercanos a 1, indicando una alta tasa de aciertos.

*   **Curvas de Precisión-Recall (PR Curve):**
    *   [`BoxPR_curve.png`](DATASETS/runs/detect/train/BoxPR_curve.png)
    Esta curva ilustra la relación entre precisión y recall para diferentes umbrales de confianza. Una curva que se mantiene cerca de la esquina superior derecha indica un rendimiento excelente.

*   **Curvas de Rendimiento General (Loss, P, R, mAP curves):**
    *   [`results.png`](DATASETS/runs/detect/train/results.png)
    Esta imagen consolidada muestra la evolución de las pérdidas de entrenamiento y validación, así como las curvas de precisión, recall y mAP a lo largo de las épocas. Se espera observar una estabilización o mejora continua de estas métricas hacia el final del entrenamiento.

**Ajustes y Mejoras Concretas:**

Los resultados preliminares indican un rendimiento muy sólido del modelo, lo que sugiere que la configuración actual (YOLOv8n, 100 épocas, imgsz 640, batch 32) es efectiva para la tarea de detección.

*   **Validación del Diseño:** Las altas métricas de precisión, recall y mAP validan el diseño del modelo YOLOv8n para esta tarea específica. La elección de un modelo pre-entrenado y su ajuste fino con el conjunto de datos ha sido exitosa.
*   **Necesidades de Ajuste:** Dado el excelente rendimiento, los ajustes futuros podrían centrarse en:
    *   **Optimización de Hiperparámetros:** Aunque el rendimiento es alto, se podría explorar un ajuste más fino de hiperparámetros como la tasa de aprendizaje (`lr0`, `lrf`), el momentum (`momentum`) o el `weight_decay` para intentar exprimir las últimas décimas de mejora, aunque el margen es pequeño.
    *   **Análisis de Errores Específicos:** Revisar las imágenes con detecciones incorrectas (falsos positivos o falsos negativos) para identificar patrones y posibles mejoras en el preprocesamiento de datos o en la estrategia de aumento de datos.
    *   **Generalización:** Aunque las métricas de validación son buenas, se podría realizar una evaluación en un conjunto de pruebas completamente independiente para asegurar la generalización del modelo a datos no vistos.
    *   **Eficiencia Computacional:** Si la aplicación final requiere mayor velocidad o menor consumo de recursos, se podría considerar la cuantificación del modelo o la exploración de modelos más ligeros, aunque el `yolov8n` ya es una versión "nano" optimizada.

En resumen, las pruebas iniciales confirman que el modelo ha sido entrenado con éxito, logrando un rendimiento robusto y preciso en la detección de objetos. Los ajustes futuros se enfocarán en la optimización marginal y la validación de la generalización.