package com.nystagmography

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.util.concurrent.Executors

class CameraActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: FaceOverlayView
    private lateinit var btnRecord: Button
    private lateinit var btnSwitchCamera: Button
    private lateinit var tvStatus: TextView

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var faceLandmarker: FaceLandmarker? = null

    private var isRecording = false
    private var recordStartTime = 0L
    private var useFrontCamera = true

    private val leftEyeXList = mutableListOf<Float>()
    private val leftEyeYList = mutableListOf<Float>()
    private val rightEyeXList = mutableListOf<Float>()
    private val rightEyeYList = mutableListOf<Float>()
    private val timestampList = mutableListOf<Float>()

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        btnRecord = findViewById(R.id.btnRecord)
        tvStatus = findViewById(R.id.tvStatus)

        setupFaceLandmarker()

        btnSwitchCamera = findViewById(R.id.btnSwitchCamera)

        btnRecord.setOnClickListener { toggleRecording() }
        btnSwitchCamera.setOnClickListener { switchCamera() }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun setupFaceLandmarker() {
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath("face_landmarker.task").build()
            )
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumFaces(1)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setResultListener { result, input ->
                onFaceLandmarkResult(result, input.width, input.height)
            }
            .setErrorListener { it.printStackTrace() }
            .build()

        faceLandmarker = FaceLandmarker.createFromOptions(this, options)
    }

    private fun onFaceLandmarkResult(result: FaceLandmarkerResult, imgW: Int, imgH: Int) {
        if (result.faceLandmarks().isEmpty()) {
            overlayView.clear()
            return
        }

        overlayView.setResults(result, imgW, imgH)

        if (!isRecording) return

        val landmarks = result.faceLandmarks()[0]
        val w = imgW.toFloat()
        val h = imgH.toFloat()

        // Eye dimensions
        val leftEyeW = eyeSpan(landmarks, FaceOverlayView.LEFT_EYE_INDICES) { it.x() * w }
        val leftEyeH = eyeSpan(landmarks, FaceOverlayView.LEFT_EYE_INDICES) { it.y() * h }
        val rightEyeW = eyeSpan(landmarks, FaceOverlayView.RIGHT_EYE_INDICES) { it.x() * w }
        val rightEyeH = eyeSpan(landmarks, FaceOverlayView.RIGHT_EYE_INDICES) { it.y() * h }

        // Face center (nose tip)
        val cx = landmarks[FaceOverlayView.NOSE_TIP].x() * w
        val cy = landmarks[FaceOverlayView.NOSE_TIP].y() * h

        // Iris centers
        val liX = landmarks[FaceOverlayView.LEFT_IRIS_INDICES[0]].x() * w
        val liY = landmarks[FaceOverlayView.LEFT_IRIS_INDICES[0]].y() * h
        val riX = landmarks[FaceOverlayView.RIGHT_IRIS_INDICES[0]].x() * w
        val riY = landmarks[FaceOverlayView.RIGHT_IRIS_INDICES[0]].y() * h

        // Relative positions normalized by eye size
        val lxr = (liX - cx) / leftEyeW
        val lyr = (liY - cy) / leftEyeH
        val rxr = (riX - cx) / rightEyeW
        val ryr = (riY - cy) / rightEyeH

        val t = (SystemClock.elapsedRealtime() - recordStartTime) / 1000f

        synchronized(this) {
            leftEyeXList.add(lxr)
            leftEyeYList.add(lyr)
            rightEyeXList.add(rxr)
            rightEyeYList.add(ryr)
            timestampList.add(t)
        }

        runOnUiThread { tvStatus.text = "Recording: %.1fs".format(t) }
    }

    private fun <T> eyeSpan(
        landmarks: List<T>,
        indices: IntArray,
        selector: (T) -> Float
    ): Float {
        val values = indices.map { selector(landmarks[it]) }
        return values.max() - values.min()
    }

    private fun toggleRecording() {
        if (!isRecording) {
            isRecording = true
            recordStartTime = SystemClock.elapsedRealtime()
            synchronized(this) {
                leftEyeXList.clear()
                leftEyeYList.clear()
                rightEyeXList.clear()
                rightEyeYList.clear()
                timestampList.clear()
            }
            btnRecord.text = "Stop"
            tvStatus.text = "Recording: 0.0s"
        } else {
            isRecording = false
            btnRecord.text = "Record"
            tvStatus.text = "Ready"

            synchronized(this) {
                if (timestampList.size >= 10) {
                    NystagmusData.leftEyeXRel = leftEyeXList.toFloatArray()
                    NystagmusData.leftEyeYRel = leftEyeYList.toFloatArray()
                    NystagmusData.rightEyeXRel = rightEyeXList.toFloatArray()
                    NystagmusData.rightEyeYRel = rightEyeYList.toFloatArray()
                    NystagmusData.timestamps = timestampList.toFloatArray()
                    startActivity(Intent(this@CameraActivity, ResultActivity::class.java))
                } else {
                    tvStatus.text = "Not enough data. Try again."
                }
            }
        }
    }

    private fun switchCamera() {
        useFrontCamera = !useFrontCamera
        startCamera()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(cameraExecutor, ::analyzeImage) }

            val cameraSelector = if (useFrontCamera) {
                CameraSelector.DEFAULT_FRONT_CAMERA
            } else {
                CameraSelector.DEFAULT_BACK_CAMERA
            }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeImage(imageProxy: ImageProxy) {
        val bmp = Bitmap.createBitmap(
            imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888
        )
        imageProxy.planes[0].buffer.rewind()
        bmp.copyPixelsFromBuffer(imageProxy.planes[0].buffer)

        val rotation = imageProxy.imageInfo.rotationDegrees.toFloat()
        val rotated = if (rotation != 0f) {
            val matrix = Matrix().apply { postRotate(rotation) }
            Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
        } else bmp

        val mpImage = BitmapImageBuilder(rotated).build()
        faceLandmarker?.detectAsync(mpImage, SystemClock.uptimeMillis())

        imageProxy.close()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        faceLandmarker?.close()
    }
}
