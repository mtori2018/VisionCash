graph TD
    subgraph "1. Capa de Presentación (UI)"
        direction LR
        A[MainActivity] -- Carga Fragmento --> B(CameraFragment);
        B -- Dibuja Resultados --> C(OverlayView);
        A -- Gestiona Navegación --> Nav(nav_graph.xml);
        B -- Muestra Controles --> UIControls[BottomSheet UI];
    end

    subgraph "2. Lógica de Control y Detección"
        direction TB
        B -- Inicia Detección --> D{ObjectDetectorHelper};
        D -- Implementa --> Listener(DetectorListener);
        B -- Escucha Resultados --> Listener;
        D -- Crea y Configura --> E[CustomYoloDetector];
    end

    subgraph "3. Núcleo de Inferencia (IA)"
        direction TB
        E -- Carga Modelo --> Model[custom_model.tflite];
        E -- Carga Metadatos --> Metadata[CustomYoloDetector.yaml];
        E -- Utiliza Motor --> G[TensorFlow Lite];
    end

    subgraph "4. Servicios del Sistema y Hardware"
        direction LR
        B -- Usa Cámara --> F[CameraX API];
        B -- Anuncia Resultados --> TTS[TextToSpeech];
        B -- Provee Feedback --> Vibrator[Vibrator Service];
    end

    %% Relaciones entre Subgrafos
    F -- Provee Frames --> B;
    E -- Devuelve Resultados --> D;
    D -- Notifica a --> B;

    %% Estilos para mayor claridad
    style A fill:#87CEEB,stroke:#333,stroke-width:2px
    style B fill:#ADD8E6,stroke:#333,stroke-width:2px
    style C fill:#B0E0E6,stroke:#333,stroke-width:2px
    style Nav fill:#E0FFFF,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style UIControls fill:#B0E0E6,stroke:#333,stroke-width:1px

    style D fill:#FFA07A,stroke:#333,stroke-width:2px
    style Listener fill:#FFDAB9,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style E fill:#FA8072,stroke:#333,stroke-width:2px

    style Model fill:#FFD700,stroke:#333,stroke-width:1px
    style Metadata fill:#FFD700,stroke:#333,stroke-width:1px
    style G fill:#FFC300,stroke:#333,stroke-width:2px

    style F fill:#98FB98,stroke:#333,stroke-width:2px
    style TTS fill:#90EE90,stroke:#333,stroke-width:1px
    style Vibrator fill:#90EE90,stroke:#333,stroke-width:1px