package com.drawing25d

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * MediaPipe magic_touch.tflite wrapper
 *
 * Input:  [1, 512, 512, 4] float32 (RGB + keypoint heatmap)
 * Output: [1, 512, 512, 1] float32 (segmentation confidence)
 */
class SegmentationHelper(context: Context) {

    companion object {
        private const val MODEL_FILE = "magic_touch.tflite"
        private const val INPUT_SIZE = 512
        private const val THRESHOLD = 0.1f
        private const val HEATMAP_SIGMA = 10f
    }

    private val interpreter: Interpreter

    init {
        val model = loadModel(context, MODEL_FILE)
        interpreter = Interpreter(model)
    }

    /**
     * Run segmentation at a touch point.
     *
     * @param bitmap Input image (any size, will be resized)
     * @param touchX Normalized x (0~1)
     * @param touchY Normalized y (0~1)
     * @return Binary mask bitmap (black/white, same size as input)
     */
    fun segment(bitmap: Bitmap, touchX: Float, touchY: Float): Bitmap {
        val origW = bitmap.width
        val origH = bitmap.height

        // Resize to 512x512
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        // Build input tensor [1, 512, 512, 4]
        val inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 4 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val cx = touchX * INPUT_SIZE
        val cy = touchY * INPUT_SIZE

        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val pixel = pixels[y * INPUT_SIZE + x]
                val r = ((pixel shr 16) and 0xFF) / 255f
                val g = ((pixel shr 8) and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f

                // Gaussian heatmap
                val dx = x - cx
                val dy = y - cy
                val heatmap = Math.exp(-((dx * dx + dy * dy) / (2 * HEATMAP_SIGMA * HEATMAP_SIGMA)).toDouble()).toFloat()

                inputBuffer.putFloat(r)
                inputBuffer.putFloat(g)
                inputBuffer.putFloat(b)
                inputBuffer.putFloat(heatmap)
            }
        }
        inputBuffer.rewind()

        // Output [1, 512, 512, 1]
        val outputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 4)
        outputBuffer.order(ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        // Threshold → binary mask at 512x512
        val maskPixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        for (i in 0 until INPUT_SIZE * INPUT_SIZE) {
            val confidence = outputBuffer.float
            maskPixels[i] = if (confidence > THRESHOLD) 0xFFFFFFFF.toInt() else 0xFF000000.toInt()
        }

        val maskSmall = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        maskSmall.setPixels(maskPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // Resize back to original size
        return Bitmap.createScaledBitmap(maskSmall, origW, origH, false)
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
            // mask white = foreground
            outPixels[i] = if ((maskPixels[i] and 0xFF) > 128) srcPixels[i] else 0x00000000
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
