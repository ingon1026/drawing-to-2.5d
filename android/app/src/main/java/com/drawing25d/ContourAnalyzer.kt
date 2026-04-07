package com.drawing25d

import android.graphics.Bitmap
import android.graphics.Path
import android.graphics.PointF
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Lightweight contour detection for visual preview only.
 * Actual segmentation is done by magic_touch TFLite.
 */
object ContourAnalyzer {

    data class ContourInfo(
        val path: Path,
        val area: Double,
        val center: PointF,  // in original image coordinates
    )

    /**
     * Find drawing contours for preview overlay.
     * Intentionally minimal morphology to avoid merging separate drawings.
     */
    fun findContours(
        bitmap: Bitmap,
        viewW: Int,
        viewH: Int,
        minArea: Int = 2000,
    ): List<ContourInfo> {
        // Downsample for speed
        val scale = 0.5f
        val smallW = (bitmap.width * scale).toInt()
        val smallH = (bitmap.height * scale).toInt()
        val small = Bitmap.createScaledBitmap(bitmap, smallW, smallH, false)

        val mat = Mat()
        Utils.bitmapToMat(small, mat)

        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)

        val binary = Mat()
        Imgproc.adaptiveThreshold(
            gray, binary, 255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY_INV,
            21, 8.0
        )

        // Minimal close — just connect broken strokes within same drawing
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, kernel, Point(-1.0, -1.0), 1)

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val totalArea = smallW * smallH
        val maxArea = totalArea * 0.25  // max 25%
        val minAreaScaled = (minArea * scale * scale).toInt()

        val scaleX = viewW.toFloat() / smallW
        val scaleY = viewH.toFloat() / smallH
        val invScale = 1f / scale

        val results = mutableListOf<ContourInfo>()

        for (contour in contours) {
            val area = Imgproc.contourArea(contour)
            if (area < minAreaScaled || area > maxArea) continue

            val path = Path()
            val points = contour.toArray()
            if (points.isEmpty()) continue

            path.moveTo(points[0].x.toFloat() * scaleX, points[0].y.toFloat() * scaleY)
            for (i in 1 until points.size) {
                path.lineTo(points[i].x.toFloat() * scaleX, points[i].y.toFloat() * scaleY)
            }
            path.close()

            val moments = Imgproc.moments(contour)
            val cx = if (moments.m00 > 0) (moments.m10 / moments.m00 * invScale).toFloat() else 0f
            val cy = if (moments.m00 > 0) (moments.m01 / moments.m00 * invScale).toFloat() else 0f

            results.add(ContourInfo(path, area / (scale * scale), PointF(cx, cy)))
        }

        mat.release()
        gray.release()
        binary.release()
        hierarchy.release()
        kernel.release()

        return results.sortedByDescending { it.area }
    }
}
