package com.drawing25d

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * MediaPipe magic_touch.tflite wrapper + OpenCV mask postprocessing.
 */
class SegmentationHelper(context: Context) {

    companion object {
        private const val MODEL_FILE = "magic_touch.tflite"
        private const val INPUT_SIZE = 512
        private const val THRESHOLD = 0.1f
        private const val HEATMAP_SIGMA = 10f
    }

    private val interpreter: Interpreter

    // Pre-allocate buffers for performance
    private val inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 4 * 4).apply {
        order(ByteOrder.nativeOrder())
    }
    private val outputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 4).apply {
        order(ByteOrder.nativeOrder())
    }

    init {
        val model = loadModel(context, MODEL_FILE)
        val options = Interpreter.Options().apply {
            numThreads = 4
        }
        interpreter = Interpreter(model, options)
    }

    /**
     * Run segmentation + postprocessing at a touch point.
     */
    fun segment(bitmap: Bitmap, touchX: Float, touchY: Float): Bitmap {
        val origW = bitmap.width
        val origH = bitmap.height

        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        // Build input tensor
        inputBuffer.rewind()
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val cx = touchX * INPUT_SIZE
        val cy = touchY * INPUT_SIZE
        val sigmaSquared2 = 2f * HEATMAP_SIGMA * HEATMAP_SIGMA

        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val pixel = pixels[y * INPUT_SIZE + x]
                inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
                inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
                inputBuffer.putFloat((pixel and 0xFF) / 255f)

                val dx = x - cx
                val dy = y - cy
                inputBuffer.putFloat(Math.exp(-((dx * dx + dy * dy) / sigmaSquared2).toDouble()).toFloat())
            }
        }
        inputBuffer.rewind()

        // Run inference
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        // Threshold → binary mask at 512x512
        val maskPixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        for (i in 0 until INPUT_SIZE * INPUT_SIZE) {
            maskPixels[i] = if (outputBuffer.float > THRESHOLD) 0xFFFFFFFF.toInt() else 0xFF000000.toInt()
        }

        val maskSmall = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        maskSmall.setPixels(maskPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // Resize to original
        val maskFull = Bitmap.createScaledBitmap(maskSmall, origW, origH, false)

        // Postprocess with OpenCV
        return postprocessMask(maskFull)
    }

    /**
     * OpenCV morphology postprocessing — same as PC pipeline.
     */
    private fun postprocessMask(mask: Bitmap): Bitmap {
        val mat = Mat()
        Utils.bitmapToMat(mask, mat)

        // Convert to single channel
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)

        // Morphological close (fill gaps in strokes)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.morphologyEx(gray, gray, Imgproc.MORPH_CLOSE, kernel, Point(-1.0, -1.0), 3)

        // Morphological open (remove noise)
        Imgproc.morphologyEx(gray, gray, Imgproc.MORPH_OPEN, kernel, Point(-1.0, -1.0), 2)

        // Keep largest connected component
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val numLabels = Imgproc.connectedComponentsWithStats(gray, labels, stats, centroids)

        if (numLabels > 1) {
            var maxArea = 0
            var maxLabel = 1
            for (i in 1 until numLabels) {
                val area = stats.get(i, Imgproc.CC_STAT_AREA)[0].toInt()
                if (area > maxArea) {
                    maxArea = area
                    maxLabel = i
                }
            }
            // Zero out everything except largest
            val result = Mat.zeros(gray.size(), CvType.CV_8UC1)
            for (y in 0 until labels.rows()) {
                for (x in 0 until labels.cols()) {
                    if (labels.get(y, x)[0].toInt() == maxLabel) {
                        result.put(y, x, 255.0)
                    }
                }
            }
            gray.release()
            labels.release()
            stats.release()
            centroids.release()

            // Fill holes
            val filled = fillHoles(result)

            // Smooth edges
            Imgproc.GaussianBlur(filled, filled, Size(5.0, 5.0), 0.0)
            Imgproc.threshold(filled, filled, 127.0, 255.0, Imgproc.THRESH_BINARY)

            // Convert back to RGBA bitmap
            val rgbaMat = Mat()
            Imgproc.cvtColor(filled, rgbaMat, Imgproc.COLOR_GRAY2RGBA)
            val resultBitmap = Bitmap.createBitmap(mask.width, mask.height, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(rgbaMat, resultBitmap)

            mat.release()
            kernel.release()
            result.release()
            filled.release()
            rgbaMat.release()

            return resultBitmap
        }

        // Fallback: just return smoothed mask
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)
        Imgproc.threshold(gray, gray, 127.0, 255.0, Imgproc.THRESH_BINARY)
        val rgbaMat = Mat()
        Imgproc.cvtColor(gray, rgbaMat, Imgproc.COLOR_GRAY2RGBA)
        val resultBitmap = Bitmap.createBitmap(mask.width, mask.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgbaMat, resultBitmap)

        mat.release()
        gray.release()
        kernel.release()
        labels.release()
        stats.release()
        centroids.release()
        rgbaMat.release()

        return resultBitmap
    }

    private fun fillHoles(mask: Mat): Mat {
        val floodFilled = mask.clone()
        val fillMask = Mat.zeros(mask.rows() + 2, mask.cols() + 2, CvType.CV_8UC1)
        Imgproc.floodFill(floodFilled, fillMask, Point(0.0, 0.0), Scalar(255.0))
        Core.bitwise_not(floodFilled, floodFilled)
        val result = Mat()
        Core.bitwise_or(mask, floodFilled, result)
        floodFilled.release()
        fillMask.release()
        return result
    }

    /**
     * Apply mask to image → transparent PNG bitmap
     */
    fun applyMask(bitmap: Bitmap, mask: Bitmap): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

        val srcPixels = IntArray(w * h)
        val maskPixels = IntArray(w * h)
        bitmap.getPixels(srcPixels, 0, w, 0, 0, w, h)
        mask.getPixels(maskPixels, 0, w, 0, 0, w, h)

        val outPixels = IntArray(w * h)
        for (i in srcPixels.indices) {
            outPixels[i] = if ((maskPixels[i] and 0xFF) > 128) srcPixels[i] else Color.TRANSPARENT
        }

        result.setPixels(outPixels, 0, w, 0, 0, w, h)
        return result
    }

    fun close() {
        interpreter.close()
    }

    private fun loadModel(context: Context, filename: String): MappedByteBuffer {
        val fd = context.assets.openFd(filename)
        val inputStream = FileInputStream(fd.fileDescriptor)
        val channel = inputStream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }
}
