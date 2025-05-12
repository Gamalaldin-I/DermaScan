package com.example.skindetection

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import com.bumptech.glide.Glide
import com.bumptech.glide.load.engine.DiskCacheStrategy
import com.example.skindetection.databinding.ActivityMainBinding
import com.google.android.material.bottomsheet.BottomSheetBehavior
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var imageCapture: ImageCapture
    private var filePath = ""
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var tfLite: TFLiteHelper
    private lateinit var viewModel: MainViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize ViewModel and TensorFlow Lite model
        viewModel = ViewModelProvider(this)[MainViewModel::class.java]
        tfLite = TFLiteHelper(this, "model.tflite")

        setupBottomSheet()
        checkPermissions()
        setupCameraExecutor()

        // Capture button click listener
        binding.btnCapture.setOnClickListener { captureImage() }
    }

    // Setup Bottom Sheet behavior
    private fun setupBottomSheet() {
        BottomSheetBehavior.from(findViewById(R.id.sheet)).apply {
            peekHeight = 150
            state = BottomSheetBehavior.STATE_COLLAPSED
        }
    }

    // Check and request camera permissions
    private fun checkPermissions() {
        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        } else {
            startCamera()
        }
    }

    // Initialize camera executor
    private fun setupCameraExecutor() {
        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    // Start camera preview and setup image capture
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            } catch (exc: Exception) {
                Log.e("CameraX", "Failed to bind camera", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // Handle image capture
    private fun captureImage() {
        binding.apply {
            progressCircular.visibility = View.VISIBLE
            shadowView.visibility = View.VISIBLE
            btnCapture.visibility = View.GONE
        }
        takePhoto()
    }

    // Take a photo and save it to cache
    private fun takePhoto() {
        filePath.takeIf { it.isNotEmpty() }?.let { File(it).delete() }
        val photoFile = File(externalCacheDir, "captured_image.jpg")
        filePath = photoFile.absolutePath
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(outputOptions, ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    processCapturedImage(photoFile)
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e("CameraX", "Failed to capture image", exception)
                }
            })
    }

    // Process the captured image and run it through the AI model
    private fun processCapturedImage(photoFile: File) {
        runOnUiThread {
            // Load the captured image into ImageView using Glide
            Glide.with(this)
                .load(photoFile)
                .skipMemoryCache(true)
                .diskCacheStrategy(DiskCacheStrategy.NONE)
                .into(binding.ivPreview)

            // Convert the image to ByteBuffer for TensorFlow Lite model
            val bitmap = BitmapFactory.decodeFile(photoFile.absolutePath)
            val byteBuffer = ImageProcessor.bitmapToByteBuffer(bitmap, 224)
            val result = tfLite.predict(byteBuffer)
            val diagnosis = viewModel.result(result)

            // Update UI with the diagnosis results
            updateUIWithResults(diagnosis)
        }
    }

    // Display diagnosis results in the Bottom Sheet
    private fun updateUIWithResults(diagnosis: SkinDisease) {
        binding.apply {
            engName.text = diagnosis.name
            arabName.text = diagnosis.arabicName
            danger.text = diagnosis.dangerLevel
            description.text = diagnosis.description
            treatment.text = diagnosis.treatment
            medicine.text = diagnosis.medicines
            safety.text = diagnosis.safetyLevel
            progressCircular.visibility = View.GONE
            shadowView.visibility = View.GONE
            btnCapture.visibility = View.VISIBLE
        }

        // Expand the Bottom Sheet with results
        BottomSheetBehavior.from(findViewById(R.id.sheet)).apply {
            peekHeight = 150
            state = BottomSheetBehavior.STATE_EXPANDED
        }
    }

    // Check if all required permissions are granted
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        tfLite.close()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
