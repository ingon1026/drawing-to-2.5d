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
 * MiDaS v2.1 small TFLite wrapper
 *
 * Input:  [1, 256, 256, 3] float32 (RGB, 0~1)
 * Output: [1, 256, 256, 1] float32 (relative depth)
 */
class DepthHelper(context: Context) {

    companion object {
        private const val MODEL_FILE = "midas_v21_small.tflite"
        private const val INPUT_SIZE = 256
    }

    private val interpreter: Interpreter

    init {
        val model = loadModel(context, MODEL_FILE)
        interpreter = Interpreter(model)
    }

    /**
     * Estimate depth map from a bitmap.
     *
     * @param bitmap Input image (any size)
     * @param mask Optional binary mask — depth outside is zeroed
     * @return Depth bitmap (grayscale, same size as input). Brighter = closer.
     */
    fun estimateDepth(bitmap: Bitmap, mask: Bitmap? = null): Bitmap {
        val origW = bitmap.width
        val origH = bitmap.height

        // Resize to 256x256
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        // Build input [1, 256, 256, 3]
        val inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
            inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
            inputBuffer.putFloat((pixel and 0xFF) / 255f)
        }
        inputBuffer.rewind()

        // Output [1, 256, 256, 1]
        val outputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 4)
        outputBuffer.order(ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        // Read depth values and normalize to [0, 255]
        val depthValues = FloatArray(INPUT_SIZE * INPUT_SIZE)
        var dMin = Float.MAX_VALUE
        var dMax = Float.MIN_VALUE
        for (i in depthValues.indices) {
            depthValues[i] = outputBuffer.float
            if (depthValues[i] < dMin) dMin = depthValues[i]
            if (depthValues[i] > dMax) dMax = depthValues[i]
        }

        val range = dMax - dMin
        val depthPixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        for (i in depthValues.indices) {
            val norm = if (range > 1e-6f) ((depthValues[i] - dMin) / range * 255).toInt().coerceIn(0, 255) else 0
            depthPixels[i] = 0xFF000000.toInt() or (norm shl 16) or (norm shl 8) or norm
        }

        val depthSmall = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        depthSmall.setPixels(depthPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // Resize to original
        val depthFull = Bitmap.createScaledBitmap(depthSmall, origW, origH, true)

        // Apply mask if provided
        if (mask != null) {
            val dPixels = IntArray(origW * origH)
            val mPixels = IntArray(origW * origH)
            depthFull.getPixels(dPixels, 0, origW, 0, 0, origW, origH)
            mask.getPixels(mPixels, 0, origW, 0, 0, origW, origH)
            for (i in dPixels.indices) {
                if ((mPixels[i] and 0xFF) < 128) {
                    dPixels[i] = 0xFF000000.toInt() // black outside mask
                }
            }
            depthFull.setPixels(dPixels, 0, origW, 0, 0, origW, origH)
        }

        return depthFull
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
