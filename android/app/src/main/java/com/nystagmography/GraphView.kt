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

    private val axisPaint = Paint().apply {
        color = Color.DKGRAY
        strokeWidth = 2f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.DKGRAY
        textSize = 28f
        isAntiAlias = true
    }

    private val titlePaint = Paint().apply {
        color = Color.BLACK
        textSize = 34f
        isAntiAlias = true
        isFakeBoldText = true
    }

    private val gridPaint = Paint().apply {
        color = Color.LTGRAY
        strokeWidth = 1f
        style = Paint.Style.STROKE
        pathEffect = DashPathEffect(floatArrayOf(8f, 4f), 0f)
    }

    private val linePaint = Paint().apply {
        strokeWidth = 3f
        style = Paint.Style.STROKE
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

        val padding = 100f
        val rightPad = 40f
        val topPad = 70f
        val bottomPad = 80f
        val graphLeft = padding
        val graphTop = topPad
        val graphRight = width - rightPad
        val graphBottom = height - bottomPad
        val graphW = graphRight - graphLeft
        val graphH = graphBottom - graphTop

        // Title
        canvas.drawText(title, graphLeft, topPad - 20f, titlePaint)

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

        // Draw axes
        canvas.drawLine(graphLeft, graphTop, graphLeft, graphBottom, axisPaint)
        canvas.drawLine(graphLeft, graphBottom, graphRight, graphBottom, axisPaint)

        // Grid lines and labels (5 horizontal, 5 vertical)
        for (i in 0..4) {
            val y = graphTop + graphH * i / 4f
            canvas.drawLine(graphLeft, y, graphRight, y, gridPaint)
            val label = "%.2f".format(if (invertY) yMin + (yMax - yMin) * i / 4f else yMax - (yMax - yMin) * i / 4f)
            canvas.drawText(label, 5f, y + 10f, textPaint)
        }
        for (i in 0..4) {
            val x = graphLeft + graphW * i / 4f
            canvas.drawLine(x, graphTop, x, graphBottom, gridPaint)
            val label = "%.1fs".format(tMin + tRange * i / 4f)
            canvas.drawText(label, x - 20f, graphBottom + 40f, textPaint)
        }

        // Draw series
        for (s in series) {
            linePaint.color = s.color
            val path = Path()
            val n = minOf(timestamps.size, s.data.size)
            for (i in 0 until n) {
                val x = graphLeft + (timestamps[i] - tMin) / tRange * graphW
                val y = if (invertY) graphTop + (s.data[i] - yMin) / (yMax - yMin) * graphH else graphBottom - (s.data[i] - yMin) / (yMax - yMin) * graphH
                if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            canvas.drawPath(path, linePaint)
        }

        // Legend
        var legendX = graphLeft + 10f
        val legendY = graphTop + 30f
        for (s in series) {
            linePaint.color = s.color
            canvas.drawLine(legendX, legendY, legendX + 30f, legendY, linePaint)
            canvas.drawText(s.label, legendX + 35f, legendY + 8f, textPaint)
            legendX += 35f + textPaint.measureText(s.label) + 20f
        }
    }
}
