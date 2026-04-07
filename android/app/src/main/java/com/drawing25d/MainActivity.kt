package com.drawing25d

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
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

    @Volatile
    private var currentContours: List<ContourAnalyzer.ContourInfo> = emptyList()
    @Volatile
    private var isProcessing = false
    @Volatile
    private var isAnalyzing = false
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
        Log.d(TAG, "OpenCV initialized")

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

            val preview = Preview.Builder()
                .build()
                .also { it.surfaceProvider = previewView.surfaceProvider }

            val analyzer = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 480))
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
        // Skip if processing a selection or already analyzing
        if (isProcessing || isAnalyzing) {
            imageProxy.close()
            return
        }

        isAnalyzing = true

        try {
            val bitmap = imageProxy.toBitmap()

            // Handle rotation
            val rotation = imageProxy.imageInfo.rotationDegrees
            val rotatedBitmap = if (rotation != 0) {
                val matrix = Matrix()
                matrix.postRotate(rotation.toFloat())
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            } else {
                bitmap
            }

            imageWidth = rotatedBitmap.width
            imageHeight = rotatedBitmap.height

            val viewW = contourOverlay.width
            val viewH = contourOverlay.height

            if (viewW > 0 && viewH > 0) {
                val contours = ContourAnalyzer.findContours(rotatedBitmap, viewW, viewH)
                currentContours = contours
                val paths = contours.map { it.path }

                runOnUiThread {
                    if (!isProcessing && !bounceView.isActive()) {
                        contourOverlay.updateContours(paths)
                        statusText.text = if (contours.isNotEmpty()) {
                            "${contours.size}개 그림 감지 — 터치하세요"
                        } else {
                            "그림을 비춰주세요"
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Analysis error", e)
        } finally {
            isAnalyzing = false
            imageProxy.close()
        }
    }

    private fun setupTouchHandler() {
        val rootLayout = findViewById<View>(R.id.rootLayout)
        rootLayout.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN && !isProcessing) {
                handleTouch(event.x, event.y)
            }
            true
        }
    }

    private fun handleTouch(touchX: Float, touchY: Float) {
        // If bounce is active, stop it and go back to camera mode
        if (bounceView.isActive()) {
            bounceView.stopBounce()
            contourOverlay.visibility = View.VISIBLE
            statusText.text = "그림을 터치하세요"
            return
        }

        // Convert touch coords to normalized [0, 1]
        val viewW = contourOverlay.width.toFloat()
        val viewH = contourOverlay.height.toFloat()
        if (viewW <= 0 || viewH <= 0) return

        val normX = touchX / viewW
        val normY = touchY / viewH
        Log.d(TAG, "Touch at norm($normX, $normY)")

        // Find which contour was touched
        val contours = currentContours
        if (contours.isEmpty()) {
            statusText.text = "감지된 그림이 없습니다"
            return
        }

        val hitIdx = ContourAnalyzer.findContourAt(
            contours, touchX, touchY,
            viewW.toInt(), viewH.toInt(),
            imageWidth, imageHeight
        )

        if (hitIdx < 0) {
            statusText.text = "그림 영역을 터치하세요"
            return
        }

        isProcessing = true
        statusText.text = "처리 중..."
        contourOverlay.clear()

        // Capture current camera frame
        val bitmap = previewView.bitmap
        if (bitmap == null) {
            isProcessing = false
            statusText.text = "프레임 캡처 실패"
            return
        }

        val selectedContour = contours[hitIdx]

        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) {
                processWithContour(bitmap, selectedContour)
            }

            if (result != null) {
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

    /**
     * Extract drawing using contour mask directly.
     * No ML model needed — contour from preview is the segmentation.
     */
    private fun processWithContour(bitmap: Bitmap, contour: ContourAnalyzer.ContourInfo): Bitmap? {
        return try {
            Log.d(TAG, "Contour mask extraction (area=${contour.area.toInt()})")
            Log.d(TAG, "  bitmap: ${bitmap.width}x${bitmap.height}, analyzer image: ${imageWidth}x${imageHeight}")

            // Scale contour points from analyzer image coords to bitmap coords
            val scaleX = bitmap.width.toDouble() / imageWidth.coerceAtLeast(1)
            val scaleY = bitmap.height.toDouble() / imageHeight.coerceAtLeast(1)
            Log.d(TAG, "  scale: ${scaleX}x${scaleY}")

            val scaledContour = if (scaleX != 1.0 || scaleY != 1.0) {
                val points = contour.contourMat.toArray()
                val scaled = points.map { pt ->
                    org.opencv.core.Point(pt.x * scaleX, pt.y * scaleY)
                }
                val scaledMat = org.opencv.core.MatOfPoint(*scaled.toTypedArray())
                ContourAnalyzer.ContourInfo(contour.path, contour.area, contour.center, scaledMat)
            } else {
                contour
            }

            val mask = ContourAnalyzer.contourToMask(scaledContour, bitmap.width, bitmap.height)

            // Check mask is not empty
            val maskPixels = IntArray(mask.width * mask.height)
            mask.getPixels(maskPixels, 0, mask.width, 0, 0, mask.width, mask.height)
            val fgCount = maskPixels.count { (it and 0xFF) > 128 }
            val fgRatio = fgCount.toFloat() / maskPixels.size
            Log.d(TAG, "  Contour mask foreground: ${(fgRatio * 100).toInt()}%")

            if (fgRatio < 0.001f) {
                Log.w(TAG, "  Empty contour mask — skipping")
                return null
            }

            // Apply mask → transparent cutout
            val fullCutout = segHelper.applyMask(bitmap, mask)

            // Crop to non-transparent bounding box
            cropToContent(fullCutout)
        } catch (e: Exception) {
            Log.e(TAG, "Processing failed", e)
            null
        }
    }

    /**
     * Crop bitmap to the bounding box of non-transparent pixels.
     */
    private fun cropToContent(bitmap: Bitmap): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        var minX = w; var minY = h; var maxX = 0; var maxY = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val alpha = (pixels[y * w + x] ushr 24) and 0xFF
                if (alpha > 0) {
                    if (x < minX) minX = x
                    if (y < minY) minY = y
                    if (x > maxX) maxX = x
                    if (y > maxY) maxY = y
                }
            }
        }

        if (maxX <= minX || maxY <= minY) return bitmap

        // Add padding
        val pad = 10
        minX = (minX - pad).coerceAtLeast(0)
        minY = (minY - pad).coerceAtLeast(0)
        maxX = (maxX + pad).coerceAtMost(w - 1)
        maxY = (maxY + pad).coerceAtMost(h - 1)

        return Bitmap.createBitmap(bitmap, minX, minY, maxX - minX + 1, maxY - minY + 1)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        segHelper.close()
        depthHelper.close()
    }
}
