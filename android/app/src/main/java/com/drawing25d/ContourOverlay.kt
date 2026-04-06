package com.drawing25d

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

/**
 * Transparent overlay that draws contour-based segment highlights.
 * Updated each frame from camera analyzer.
 */
class ContourOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    companion object {
        private val SEGMENT_COLORS = intArrayOf(
            0x80FF5050.toInt(), // blue-ish
            0x8050FF50.toInt(), // green
            0x805050FF.toInt(), // red
            0x80FFFF50.toInt(), // cyan
            0x80FF50FF.toInt(), // magenta
            0x8050FFFF.toInt(), // yellow
        )
    }

    // List of contour paths to draw
    private var contourPaths: List<Path> = emptyList()
    private var highlightIndex: Int = -1

    private val fillPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val highlightPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val strokePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        isAntiAlias = true
        color = Color.WHITE
    }

    /**
     * Update contours from camera frame analysis.
     * Call from UI thread.
     *
     * @param paths List of Path objects (in view coordinates)
     * @param highlight Index of path to highlight (-1 for none)
     */
    fun updateContours(paths: List<Path>, highlight: Int = -1) {
        contourPaths = paths
        highlightIndex = highlight
        invalidate()
    }

    fun clear() {
        contourPaths = emptyList()
        highlightIndex = -1
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for ((i, path) in contourPaths.withIndex()) {
            val colorIdx = i % SEGMENT_COLORS.size
            if (i == highlightIndex) {
                highlightPaint.color = SEGMENT_COLORS[colorIdx] or 0x40000000.toInt() // more opaque
                canvas.drawPath(path, highlightPaint)
                canvas.drawPath(path, strokePaint)
            } else {
                fillPaint.color = SEGMENT_COLORS[colorIdx]
                canvas.drawPath(path, fillPaint)
            }
        }
    }
}
