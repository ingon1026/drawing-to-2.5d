package com.drawing25d

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.*

/**
 * Canvas-based 2.5D bounce animation view.
 * Displays a cutout bitmap bouncing on screen with squash-stretch effect.
 */
class BounceView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    companion object {
        private const val GRAVITY = 1200f    // px/s^2
        private const val BOUNCE_DAMPING = 0.7f
        private const val SQUASH_AMOUNT = 0.25f
        private const val SQUASH_DECAY = 8f
        private const val SWAY_AMPLITUDE = 20f
        private const val SWAY_SPEED = 1.5f
        private const val FPS = 60
    }

    private var spriteBitmap: Bitmap? = null
    private var spriteW = 0f
    private var spriteH = 0f

    // Physics
    private var cx = 0f
    private var cy = 0f
    private var vy = 0f
    private var squash = 0f
    private var swayPhase = 0f
    private var groundY = 0f
    private var isAnimating = false
    private var lastTime = 0L

    private val paint = Paint().apply {
        isFilterBitmap = true
        isAntiAlias = true
    }

    private val shadowPaint = Paint().apply {
        color = Color.argb(50, 0, 0, 0)
        isAntiAlias = true
    }

    /**
     * Start bounce animation with a cutout bitmap.
     */
    fun startBounce(bitmap: Bitmap) {
        // Scale to fit ~40% of view height
        val maxH = height * 0.4f
        val scale = if (bitmap.height > maxH) maxH / bitmap.height else 1f
        spriteBitmap = if (scale < 1f) {
            Bitmap.createScaledBitmap(bitmap, (bitmap.width * scale).toInt(), (bitmap.height * scale).toInt(), true)
        } else {
            bitmap
        }

        spriteW = spriteBitmap!!.width.toFloat()
        spriteH = spriteBitmap!!.height.toFloat()
        cx = width / 2f
        cy = height * 0.1f  // start near top
        vy = 0f
        squash = 0f
        swayPhase = 0f
        groundY = height * 0.85f
        isAnimating = true
        lastTime = System.nanoTime()

        invalidate()
    }

    fun stopBounce() {
        isAnimating = false
        spriteBitmap = null
        invalidate()
    }

    fun isActive(): Boolean = isAnimating

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (!isAnimating || spriteBitmap == null) return

        val now = System.nanoTime()
        val dt = ((now - lastTime) / 1_000_000_000f).coerceAtMost(0.05f)
        lastTime = now

        // Physics update
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
            if (abs(vy) < 30) vy = 0f
        }

        // Squash decay
        if (squash > 0.001f) {
            squash *= exp(-SQUASH_DECAY * dt)
        } else {
            squash = 0f
        }

        // Draw shadow
        val heightAbove = (groundY - (cy + spriteH / 2)).coerceAtLeast(0f)
        val spread = (1f - heightAbove / 400f).coerceAtLeast(0.3f)
        val shadowW = spriteW * spread * 0.8f
        val shadowH = 12f * spread
        canvas.drawOval(
            cx + swayOffset - shadowW / 2,
            groundY - shadowH / 2,
            cx + swayOffset + shadowW / 2,
            groundY + shadowH / 2,
            shadowPaint
        )

        // Draw sprite with squash-stretch
        val sx = 1f + squash
        val sy = 1f - squash
        val drawW = spriteW * sx
        val drawH = spriteH * sy

        val dstLeft = cx + swayOffset - drawW / 2
        val dstTop = cy + spriteH / 2 - drawH  // bottom-aligned
        val dstRect = RectF(dstLeft, dstTop, dstLeft + drawW, dstTop + drawH)
        val srcRect = Rect(0, 0, spriteBitmap!!.width, spriteBitmap!!.height)

        canvas.drawBitmap(spriteBitmap!!, srcRect, dstRect, paint)

        // Keep animating
        postInvalidateDelayed((1000 / FPS).toLong())
    }
}
