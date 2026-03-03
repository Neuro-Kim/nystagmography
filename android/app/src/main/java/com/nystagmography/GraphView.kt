package com.nystagmography

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class GraphView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var title = ""
    private var yLabel = ""
    private var invertY = false
    private val series = mutableListOf<Series>()

    data class Series(val data: FloatArray, val color: Int, val label: String)

    private var timestamps = floatArrayOf()

    // iOS-style colors
    private val axisPaint = Paint().apply {
        color = Color.parseColor("#C7C7CC")
        strokeWidth = 1f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.parseColor("#8E8E93")
        textSize = 24f
        isAntiAlias = true
    }

    private val titlePaint = Paint().apply {
        color = Color.parseColor("#1C1C1E")
        textSize = 32f
        isAntiAlias = true
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }

    private val gridPaint = Paint().apply {
        color = Color.parseColor("#E5E5EA")
        strokeWidth = 1f
        style = Paint.Style.STROKE
    }

    private val linePaint = Paint().apply {
        strokeWidth = 2.5f
        style = Paint.Style.STROKE
        isAntiAlias = true
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    private val legendDotPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    fun setData(
        title: String,
        yLabel: String,
        timestamps: FloatArray,
        vararg seriesList: Series
    ) {
        this.title = title
        this.yLabel = yLabel
        this.timestamps = timestamps
        this.series.clear()
        this.series.addAll(seriesList)
        invalidate()
    }

    fun setInvertY(invert: Boolean) {
        this.invertY = invert
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (timestamps.isEmpty() || series.isEmpty()) return

        val padding = 90f
        val rightPad = 24f
        val topPad = 64f
        val bottomPad = 72f
        val graphLeft = padding
        val graphTop = topPad
        val graphRight = width - rightPad
        val graphBottom = height - bottomPad
        val graphW = graphRight - graphLeft
        val graphH = graphBottom - graphTop

        // Title
        canvas.drawText(title, graphLeft, topPad - 24f, titlePaint)

        // Legend (next to title)
        var legendX = graphLeft + titlePaint.measureText(title) + 16f
        val legendY = topPad - 24f
        for (s in series) {
            legendDotPaint.color = s.color
            canvas.drawCircle(legendX + 5f, legendY - 5f, 5f, legendDotPaint)
            val labelPaint = Paint(textPaint).apply { textSize = 22f }
            canvas.drawText(s.label, legendX + 14f, legendY, labelPaint)
            legendX += 14f + labelPaint.measureText(s.label) + 16f
        }

        // Find data range
        val tMin = timestamps.min()
        val tMax = timestamps.max()
        var yMin = Float.MAX_VALUE
        var yMax = Float.MIN_VALUE
        for (s in series) {
            for (v in s.data) {
                if (v < yMin) yMin = v
                if (v > yMax) yMax = v
            }
        }
        val yRange = (yMax - yMin).coerceAtLeast(0.001f)
        val yPadding = yRange * 0.1f
        yMin -= yPadding
        yMax += yPadding
        val tRange = (tMax - tMin).coerceAtLeast(0.001f)

        // Horizontal grid lines
        for (i in 0..4) {
            val y = graphTop + graphH * i / 4f
            canvas.drawLine(graphLeft, y, graphRight, y, gridPaint)
            val label = "%.2f".format(
                if (invertY) yMin + (yMax - yMin) * i / 4f
                else yMax - (yMax - yMin) * i / 4f
            )
            canvas.drawText(label, 4f, y + 8f, textPaint)
        }

        // Vertical grid lines + time labels
        for (i in 0..4) {
            val x = graphLeft + graphW * i / 4f
            canvas.drawLine(x, graphTop, x, graphBottom, gridPaint)
            val label = "%.1fs".format(tMin + tRange * i / 4f)
            canvas.drawText(label, x - 18f, graphBottom + 36f, textPaint)
        }

        // Bottom axis line
        canvas.drawLine(graphLeft, graphBottom, graphRight, graphBottom, axisPaint)

        // Draw series with smooth paths
        for (s in series) {
            linePaint.color = s.color
            val path = Path()
            val n = minOf(timestamps.size, s.data.size)
            for (i in 0 until n) {
                val x = graphLeft + (timestamps[i] - tMin) / tRange * graphW
                val y = if (invertY) {
                    graphTop + (s.data[i] - yMin) / (yMax - yMin) * graphH
                } else {
                    graphBottom - (s.data[i] - yMin) / (yMax - yMin) * graphH
                }
                if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            canvas.drawPath(path, linePaint)
        }
    }
}
