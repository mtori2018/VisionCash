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
package org.tensorflow.lite.examples.objectdetection.fragments

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.graphics.Bitmap
import android.content.Context
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.Navigation
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper
import org.tensorflow.lite.examples.objectdetection.R
import org.tensorflow.lite.examples.objectdetection.databinding.FragmentCameraBinding
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetection
import java.util.LinkedList
import java.util.Locale

/**
 * Fragmento principal que gestiona la cรกmara, la detecciรณn de objetos y la interfaz de usuario.
 *
 * Este fragmento se encarga de:
 * - Inicializar y configurar la cรกmara utilizando CameraX.
 * - Crear una instancia de `ObjectDetectorHelper` para procesar los frames de la cรกmara.
 * - Mostrar la vista previa de la cรกmara y los resultados de la detecciรณn (cuadros delimitadores).
 * - Gestionar los controles de la interfaz de usuario para ajustar los parรกmetros de detecciรณn.
 * - Implementar Text-to-Speech para anunciar los objetos detectados.
 */
class CameraFragment : Fragment(), ObjectDetectorHelper.DetectorListener, TextToSpeech.OnInitListener {

    private val TAG = "ObjectDetection"
    private lateinit var tts: TextToSpeech
    private var lastAnnouncementTime: Long = 0L

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var objectDetectorHelper: ObjectDetectorHelper
    private lateinit var bitmapBuffer: Bitmap
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    /** Blocking camera operations are performed using this executor */
    private lateinit var cameraExecutor: ExecutorService
    private var isFlashOn = false

    override fun onResume() {
        super.onResume()
        // Make sure that all permissions are still present, since the
        // user could have removed them while the app was in paused state.
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(requireActivity(), R.id.fragment_container)
                .navigate(CameraFragmentDirections.actionCameraToPermissions())
        }

        // Re-initialize the object detector and camera after resuming from a paused state.
        objectDetectorHelper.clearObjectDetector()
        fragmentCameraBinding.overlay.clear()
        fragmentCameraBinding.viewFinder.post {
            setUpCamera()
        }
    }

    override fun onPause() {
        super.onPause()
        cameraProvider?.unbindAll() // Libera todos los casos de uso de la cรกmara
        objectDetectorHelper.close() // Cierra el detector de objetos
    }

    override fun onStop() {
        super.onStop()
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        super.onDestroyView()

        // Shut down our background executor
        cameraExecutor.shutdown()
        tts.stop()
        tts.shutdown()
    }

    override fun onCreateView(
      inflater: LayoutInflater,
      container: ViewGroup?,
      savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding = FragmentCameraBinding.inflate(inflater, container, false)

        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        objectDetectorHelper = ObjectDetectorHelper(
            context = requireContext(),
            objectDetectorListener = this
        )

        // Initialize TextToSpeech engine
        tts = TextToSpeech(requireContext(), this)

        // Initialize our background executor
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Wait for the views to be properly laid out
        fragmentCameraBinding.viewFinder.post {
            // Set up the camera and its use cases
            setUpCamera()
        }

        // Attach listeners to UI control widgets
        initBottomSheetControls()
    }

    /**
     * Inicializa los listeners para los controles de la hoja inferior (bottom sheet).
     * Permite al usuario ajustar el umbral, el nรบmero mรกximo de resultados y los hilos de inferencia.
     */
    private fun initBottomSheetControls() {
       fragmentCameraBinding.flashButton.setOnClickListener {
           isFlashOn = !isFlashOn
           camera?.cameraControl?.enableTorch(isFlashOn)
           val announcement = if (isFlashOn) {
               getString(R.string.flash_on)
           } else {
               getString(R.string.flash_off)
           }
           tts.speak(announcement, TextToSpeech.QUEUE_FLUSH, null, "")
       }
   }


    // Update the values displayed in the bottom sheet. Reset detector.
    /**
     * Actualiza la interfaz de usuario de los controles en la hoja inferior.
     * Se llama cada vez que se modifica un parรกmetro de detecciรณn.
     * Tambiรฉn limpia el detector para que se reinicialice con la nueva configuraciรณn.
     */
    private fun updateControlsUi() {
        // Needs to be cleared instead of reinitialized because the GPU
        // delegate needs to be initialized on the thread using it when applicable
        objectDetectorHelper.clearObjectDetector()
        fragmentCameraBinding.overlay.clear()
    }

    // Initialize CameraX, and prepare to bind the camera use cases
    /**
     * Inicializa CameraX.
     * Obtiene una instancia del `ProcessCameraProvider` y, una vez disponible,
     * llama a `bindCameraUseCases()` para configurar la vista previa y el anรกlisis de imรกgenes.
     */
    private fun setUpCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                // CameraProvider
                cameraProvider = cameraProviderFuture.get()

                // Build and bind the camera use cases
                bindCameraUseCases()
            },
            ContextCompat.getMainExecutor(requireContext())
        )
    }

    // Declare and bind preview, capture and analysis use cases
    @SuppressLint("UnsafeOptInUsageError")
    /**
     * Declara y vincula los casos de uso de la cรกmara (vista previa y anรกlisis de imรกgenes).
     * Configura la cรกmara trasera, el aspect ratio y el analizador de imรกgenes que
     * procesarรก cada frame.
     */
    private fun bindCameraUseCases() {

        // CameraProvider
        val cameraProvider =
            cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        // CameraSelector - makes assumption that we're only using the back camera
        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()

        // Preview. Only using the 4:3 ratio because this is the closest to our models
        preview =
            Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .build()

        // ImageAnalysis. Using RGBA 8888 to match how our models work
        imageAnalyzer =
            ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                // The analyzer can then be assigned to the instance
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        if (!::bitmapBuffer.isInitialized) {
                            // The image rotation and RGB image buffer are initialized only once
                            // the analyzer has started running
                            bitmapBuffer = Bitmap.createBitmap(
                              image.width,
                              image.height,
                              Bitmap.Config.ARGB_8888
                            )
                        }

                        detectObjects(image)
                    }
                }

        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)

            // Attach the viewfinder's surface provider to preview use case
            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    /**
     * Procesa un `ImageProxy` de la cรกmara para la detecciรณn de objetos.
     * Convierte la imagen a un `Bitmap` y la pasa al `ObjectDetectorHelper`.
     * @param image El `ImageProxy` proporcionado por el analizador de CameraX.
     */
    private fun detectObjects(image: ImageProxy) {
        // Copy out RGB bits to the shared bitmap buffer
        image.use {
            bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer)
        }

        val imageRotation = image.imageInfo.rotationDegrees
        // Pass Bitmap and rotation to the object detector helper for processing and detection
        objectDetectorHelper.detect(bitmapBuffer, imageRotation)
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation = fragmentCameraBinding.viewFinder.display.rotation
    }

    // Update UI after objects have been detected. Extracts original image height/width
    // to scale and place bounding boxes properly through OverlayView
    /**
     * Callback que se ejecuta cuando el `ObjectDetectorHelper` devuelve resultados.
     * Actualiza la interfaz de usuario con el tiempo de inferencia, dibuja los cuadros
     * delimitadores en el `OverlayView` y anuncia el resultado principal mediante TTS.
     */
    override fun onResults(
        results: List<ObjectDetection>,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        activity?.runOnUiThread {
            fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                            String.format("%d ms", inferenceTime)

            // Announce the top result if confidence is high and cooldown has passed
            val currentTime = System.currentTimeMillis()
            if (!tts.isSpeaking && (currentTime - lastAnnouncementTime > ANNOUNCEMENT_COOLDOWN)) {
                results?.firstOrNull()?.let { topResult ->
                    if (topResult.category.confidence > 0.85) { // Umbral de confianza alto
                        val textToSpeak = topResult.category.label
                        tts.speak(textToSpeak, TextToSpeech.QUEUE_FLUSH, null, "")
                        vibratePhone()
                        lastAnnouncementTime = currentTime
                    }
                }
            }

            // Pass necessary information to OverlayView for drawing on the canvas
            fragmentCameraBinding.overlay.setResults(
                results ?: LinkedList<ObjectDetection>(),
                imageHeight,
                imageWidth
            )

            // Force a redraw
            fragmentCameraBinding.overlay.invalidate()
        }
    }

    override fun onError(error: String) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
    }

    // Callback for TextToSpeech initialization
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            // Set Spanish as the language for TTS
            val result = tts.setLanguage(Locale("es", "ES"))
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e(TAG, "The Language specified is not supported!")
            } else {
                Log.i(TAG, "TextToSpeech initialized successfully.")
                tts.speak(getString(R.string.welcome_message), TextToSpeech.QUEUE_FLUSH, null, "")
            }
        } else {
            Log.e(TAG, "TextToSpeech initialization failed!")
        }
    }

    private fun vibratePhone() {
        val vibrator = context?.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(200, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            //deprecated in API 26
            vibrator.vibrate(200)
        }
    }

    companion object {
        private const val ANNOUNCEMENT_COOLDOWN = 4000L // 4 segundos
    }
}
