package com.drawing25d

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.*

/**
 * Layered pseudo-3D bounce view.
 * Renders: shadow → side face → top face, with auto-tilt on bounce.
 */
class BounceView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    companion object {
        private const val GRAVITY = 1200f
        private const val BOUNCE_DAMPING = 0.65f
        private const val SQUASH_AMOUNT = 0.15f
        private const val SQUASH_DECAY = 10f
        private const val SWAY_AMPLITUDE = 10f
        private const val SWAY_SPEED = 1.0f
        private const val TILT_BOUNCE = 3f
        private const val TILT_DAMPING = 0.95f
        private const val TILT_SPRING = -5f
        private const val TILT_MAX = 8f
        private const val SIDE_THICKNESS = 5
        private const val FPS = 60
    }

    private var topBitmap: Bitmap? = null      // original cutout
    private var sideBitmap: Bitmap? = null      // darkened version for side face
    private var originalBitmap: Bitmap? = null
    private var spriteW = 0f
    private var spriteH = 0f

    // Physics
    private var cx = 0f
    private var cy = 0f
    private var vy = 0f
    private var squash = 0f
    private var swayPhase = 0f
    private var tilt = 0f
    private var tiltVel = 0f
    private var groundY = 0f
    private var isAnimating = false
    private var lastTime = 0L

    private val paint = Paint().apply {
        isFilterBitmap = true
        isAntiAlias = true
    }

    private val shadowPaint = Paint().apply {
        isAntiAlias = true
    }

    fun startBounce(bitmap: Bitmap) {
        originalBitmap = bitmap
        layoutSprite()
        resetPhysics()
        isAnimating = true
        lastTime = System.nanoTime()
        invalidate()
    }

    private fun layoutSprite() {
        val bmp = originalBitmap ?: return
        val vw = width.toFloat()
        val vh = height.toFloat()
        if (vw <= 0 || vh <= 0) return

        val maxDim = min(vw, vh) * 0.35f
        val bmpMax = max(bmp.width, bmp.height).toFloat()
        val scale = if (bmpMax > maxDim) maxDim / bmpMax else 1f

        topBitmap = if (scale < 1f) {
            Bitmap.createScaledBitmap(bmp, (bmp.width * scale).toInt(), (bmp.height * scale).toInt(), true)
        } else {
            bmp.copy(Bitmap.Config.ARGB_8888, false)
        }

        // Build side face — darken to ~25%
        sideBitmap = buildSideFace(topBitmap!!)

        spriteW = topBitmap!!.width.toFloat()
        spriteH = topBitmap!!.height.toFloat()
        groundY = vh * 0.85f
    }

    private fun buildSideFace(src: Bitmap): Bitmap {
        val w = src.width
        val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        val out = IntArray(w * h)
        for (i in pixels.indices) {
            val a = (pixels[i] ushr 24) and 0xFF
            val r = ((pixels[i] shr 16) and 0xFF) * 25 / 100
            val g = ((pixels[i] shr 8) and 0xFF) * 20 / 100
            val b = (pixels[i] and 0xFF) * 15 / 100
            out[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
        }

        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        result.setPixels(out, 0, w, 0, 0, w, h)
        return result
    }

    private fun resetPhysics() {
        cx = width / 2f
        cy = height * 0.08f
        vy = 0f
        squash = 0f
        swayPhase = 0f
        tilt = 0f
        tiltVel = 0f
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (isAnimating && originalBitmap != null) {
            layoutSprite()
            resetPhysics()
            lastTime = System.nanoTime()
        }
    }

    fun stopBounce() {
        isAnimating = false
        topBitmap = null
        sideBitmap = null
        originalBitmap = null
        invalidate()
    }

    fun isActive(): Boolean = isAnimating

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (!isAnimating || topBitmap == null || sideBitmap == null) return

        val now = System.nanoTime()
        val dt = ((now - lastTime) / 1_000_000_000f).coerceAtMost(0.05f)
        lastTime = now

        // Physics
        vy += GRAVITY * dt
        cy += vy * dt
        swayPhase += SWAY_SPEED * dt * 2 * Math.PI.toFloat()
        val swayOffset = sin(swayPhase) * SWAY_AMPLITUDE

        // Bounce
        val bottom = cy + spriteH / 2
        if (bottom >= groundY) {
            cy = groundY - spriteH / 2
            vy = -abs(vy) * BOUNCE_DAMPING
            squash = SQUASH_AMOUNT
            tiltVel += TILT_BOUNCE * (if (sin(swayPhase) > 0) 1f else -1f)
            if (abs(vy) < 30) vy = 0f
        }

        // Squash decay
        if (squash > 0.001f) squash *= exp(-SQUASH_DECAY * dt) else squash = 0f

        // Tilt spring
        tiltVel += TILT_SPRING * tilt * dt
        tiltVel *= TILT_DAMPING
        tilt += tiltVel * dt * 60
        tilt = tilt.coerceIn(-TILT_MAX, TILT_MAX)

        // Dimensions
        val sx = 1f + squash
        val sy = 1f - squash
        val drawW = (spriteW * sx).toInt().coerceAtLeast(1)
        val drawH = (spriteH * sy).toInt().coerceAtLeast(1)

        val rx = (cx + swayOffset - drawW / 2).toInt()
        val ry = (cy + spriteH / 2 - drawH).toInt()

        // 1. Shadow
        val heightAbove = (groundY - (cy + spriteH / 2)).coerceAtLeast(0f)
        val spread = (1f - heightAbove / 400f).coerceAtLeast(0.3f)
        val shadowW = (spriteW * spread * 0.9f).toInt()
        val shadowH = (16f * spread).toInt().coerceAtLeast(6)
        val shadowAlpha = (50 * spread).toInt()
        shadowPaint.color = Color.argb(shadowAlpha, 0, 0, 0)
        canvas.drawOval(
            cx + swayOffset - shadowW / 2f,
            groundY - shadowH / 2f,
            cx + swayOffset + shadowW / 2f,
            groundY + shadowH / 2f,
            shadowPaint
        )

        // 2. Side face layers
        val scaledSide = Bitmap.createScaledBitmap(sideBitmap!!, drawW, drawH, true)
        val sideVisible = abs(tilt) / TILT_MAX
        val sidePixels = (SIDE_THICKNESS * sideVisible + 1).toInt()
        val srcRect = Rect(0, 0, drawW, drawH)
        for (i in sidePixels downTo 1) {
            val offsetX = (tilt * 0.15f * i).toInt()
            val dstRect = Rect(rx + offsetX, ry + i, rx + offsetX + drawW, ry + i + drawH)
            canvas.drawBitmap(scaledSide, srcRect, dstRect, paint)
        }

        // 3. Top face
        val scaledTop = Bitmap.createScaledBitmap(topBitmap!!, drawW, drawH, true)
        canvas.drawBitmap(scaledTop, rx.toFloat(), ry.toFloat(), paint)

        postInvalidateDelayed((1000 / FPS).toLong())
    }
}
