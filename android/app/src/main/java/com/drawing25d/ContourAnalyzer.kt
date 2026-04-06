package com.drawing25d

import android.graphics.Bitmap
import android.graphics.Path
import android.graphics.PointF
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Analyzes camera frames to find drawing contours.
 * Uses adaptive threshold + contour detection (same logic as Python version).
 */
object ContourAnalyzer {

    data class ContourInfo(
        val path: Path,          // for drawing on overlay
        val mask: Mat,           // filled contour mask
        val area: Double,
        val center: PointF,
        val contour: MatOfPoint,
    )

    /**
     * Find drawing contours in a camera frame bitmap.
     *
     * @param bitmap Camera frame
     * @param viewW Target view width (for coordinate scaling)
     * @param viewH Target view height (for coordinate scaling)
     * @param minArea Minimum contour area in pixels
     * @return List of ContourInfo, sorted by area descending
     */
    fun findContours(
        bitmap: Bitmap,
        viewW: Int,
        viewH: Int,
        minArea: Int = 800
    ): List<ContourInfo> {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)

        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)

        // Adaptive threshold (same as Python)
        val binary = Mat()
        Imgproc.adaptiveThreshold(
            gray, binary, 255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY_INV,
            31, 10.0
        )

        // Morphological close + dilate
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(15.0, 15.0))
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, kernel, Point(-1.0, -1.0), 2)
        Imgproc.dilate(binary, binary, kernel)

        // Find contours
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val imgW = bitmap.width
        val imgH = bitmap.height
        val totalArea = imgW * imgH
        val scaleX = viewW.toFloat() / imgW
        val scaleY = viewH.toFloat() / imgH

        val results = mutableListOf<ContourInfo>()

        for (contour in contours) {
            val area = Imgproc.contourArea(contour)
            if (area < minArea || area > totalArea * 0.8) continue

            // Create filled mask
            val mask = Mat.zeros(imgH, imgW, CvType.CV_8UC1)
            Imgproc.drawContours(mask, listOf(contour), -1, Scalar(255.0), -1)

            // Convert contour to Path (scaled to view coords)
            val path = Path()
            val points = contour.toArray()
            if (points.isNotEmpty()) {
                path.moveTo(points[0].x.toFloat() * scaleX, points[0].y.toFloat() * scaleY)
                for (i in 1 until points.size) {
                    path.lineTo(points[i].x.toFloat() * scaleX, points[i].y.toFloat() * scaleY)
                }
                path.close()
            }

            // Center point
            val moments = Imgproc.moments(contour)
            val cx = if (moments.m00 > 0) (moments.m10 / moments.m00).toFloat() else 0f
            val cy = if (moments.m00 > 0) (moments.m01 / moments.m00).toFloat() else 0f

            results.add(ContourInfo(path, mask, area, PointF(cx, cy), contour))
        }

        // Cleanup
        gray.release()
        binary.release()
        hierarchy.release()
        mat.release()

        return results.sortedByDescending { it.area }
    }

    /**
     * Find which contour contains a point (in image coordinates).
     */
    fun findContourAt(contours: List<ContourInfo>, x: Float, y: Float): Int {
        // Prefer smaller (more specific) contours
        for (i in contours.indices.reversed()) {
            val result = Imgproc.pointPolygonTest(
                MatOfPoint2f(*contours[i].contour.toArray()),
                Point(x.toDouble(), y.toDouble()),
                false
            )
            if (result >= 0) return i
        }
        return -1
    }
}
