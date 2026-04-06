package com.drawing25d

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "Drawing25D"
        private const val CAMERA_PERMISSION_CODE = 100
    }

    private lateinit var previewView: PreviewView
    private lateinit var contourOverlay: ContourOverlay
    private lateinit var bounceView: BounceView
    private lateinit var statusText: TextView

    private lateinit var segHelper: SegmentationHelper
    private lateinit var depthHelper: DepthHelper
    private lateinit var cameraExecutor: ExecutorService

    private var currentContours: List<ContourAnalyzer.ContourInfo> = emptyList()
    private var isProcessing = false
    private var imageWidth = 0
    private var imageHeight = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Init OpenCV
        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV init failed")
            Toast.makeText(this, "OpenCV 초기화 실패", Toast.LENGTH_LONG).show()
            return
        }

        // Bind views
        previewView = findViewById(R.id.previewView)
        contourOverlay = findViewById(R.id.contourOverlay)
        bounceView = findViewById(R.id.bounceView)
        statusText = findViewById(R.id.statusText)

        // Init models
        segHelper = SegmentationHelper(this)
        depthHelper = DepthHelper(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Touch handler
        setupTouchHandler()

        // Camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE && grantResults.isNotEmpty()
            && grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            Toast.makeText(this, "카메라 권한이 필요합니다", Toast.LENGTH_LONG).show()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also { it.surfaceProvider = previewView.surfaceProvider }

            // Image analysis for contour detection
            val analyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        analyzeFrame(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, analyzer)
                Log.d(TAG, "Camera started")
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        if (isProcessing) {
            imageProxy.close()
            return
        }

        val bitmap = imageProxy.toBitmap()
        imageWidth = bitmap.width
        imageHeight = bitmap.height

        val viewW = contourOverlay.width
        val viewH = contourOverlay.height

        if (viewW <= 0 || viewH <= 0) {
            imageProxy.close()
            return
        }

        val contours = ContourAnalyzer.findContours(bitmap, viewW, viewH)
        currentContours = contours

        val paths = contours.map { it.path }

        runOnUiThread {
            contourOverlay.updateContours(paths)
            if (contours.isNotEmpty()) {
                statusText.text = "${contours.size}개 그림 감지 — 터치하세요"
            } else {
                statusText.text = "그림을 비춰주세요"
            }
        }

        imageProxy.close()
    }

    private fun setupTouchHandler() {
        // Touch on the whole screen
        val rootLayout = findViewById<View>(R.id.rootLayout)
        rootLayout.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN && !isProcessing) {
                handleTouch(event.x, event.y)
            }
            true
        }
    }

    private fun handleTouch(touchX: Float, touchY: Float) {
        if (currentContours.isEmpty()) return

        // If bounce is active, stop it and go back to camera mode
        if (bounceView.isActive()) {
            bounceView.stopBounce()
            contourOverlay.visibility = View.VISIBLE
            statusText.text = "그림을 터치하세요"
            return
        }

        // Convert touch coords to image coords
        val viewW = contourOverlay.width.toFloat()
        val viewH = contourOverlay.height.toFloat()
        val imgX = touchX / viewW * imageWidth
        val imgY = touchY / viewH * imageHeight

        val idx = ContourAnalyzer.findContourAt(currentContours, imgX, imgY)
        if (idx < 0) return

        // Found a contour — process it
        val contour = currentContours[idx]
        val normX = contour.center.x / imageWidth
        val normY = contour.center.y / imageHeight

        isProcessing = true
        statusText.text = "처리 중..."
        contourOverlay.clear()

        // Capture current frame and process in background
        val bitmap = previewView.bitmap ?: return

        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) {
                processDrawing(bitmap, normX, normY)
            }

            if (result != null) {
                // Show bounce animation
                contourOverlay.visibility = View.INVISIBLE
                statusText.text = "터치하면 다시 카메라로"
                bounceView.startBounce(result)
            } else {
                statusText.text = "추출 실패 — 다시 시도하세요"
                contourOverlay.visibility = View.VISIBLE
            }

            isProcessing = false
        }
    }

    private fun processDrawing(bitmap: Bitmap, normX: Float, normY: Float): Bitmap? {
        return try {
            Log.d(TAG, "Segmenting at ($normX, $normY)")

            // 1. Segmentation
            val mask = segHelper.segment(bitmap, normX, normY)

            // Check foreground ratio
            val maskPixels = IntArray(mask.width * mask.height)
            mask.getPixels(maskPixels, 0, mask.width, 0, 0, mask.width, mask.height)
            val fgCount = maskPixels.count { (it and 0xFF) > 128 }
            val fgRatio = fgCount.toFloat() / maskPixels.size
            Log.d(TAG, "Foreground: ${(fgRatio * 100).toInt()}%")

            if (fgRatio < 0.005f) {
                Log.w(TAG, "Almost no foreground detected")
                return null
            }

            // 2. Apply mask → transparent cutout
            val cutout = segHelper.applyMask(bitmap, mask)

            // 3. Depth estimation (for future use, saved but not displayed yet)
            val depth = depthHelper.estimateDepth(bitmap, mask)
            Log.d(TAG, "Depth estimation done")

            cutout
        } catch (e: Exception) {
            Log.e(TAG, "Processing failed", e)
            null
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        segHelper.close()
        depthHelper.close()
    }
}
