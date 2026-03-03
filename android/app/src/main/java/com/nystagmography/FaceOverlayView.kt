package com.nystagmography

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult

class FaceOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var result: FaceLandmarkerResult? = null
    private var imageWidth = 1
    private var imageHeight = 1
    var isMirrored = true

    private val irisPaint = Paint().apply {
        color = Color.parseColor("#30D158")  // iOS green
        style = Paint.Style.FILL
    }

    private val eyeContourPaint = Paint().apply {
        color = Color.parseColor("#FFD60A")  // iOS yellow
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }

    private val nosePaint = Paint().apply {
        color = Color.parseColor("#FF453A")  // iOS red
        style = Paint.Style.FILL
    }

    companion object {
        val LEFT_EYE_INDICES = intArrayOf(362, 385, 387, 263, 373, 380)
        val RIGHT_EYE_INDICES = intArrayOf(33, 160, 158, 133, 153, 144)
        val LEFT_IRIS_INDICES = intArrayOf(473, 474, 475, 476, 477)
        val RIGHT_IRIS_INDICES = intArrayOf(468, 469, 470, 471, 472)
        const val NOSE_TIP = 4
    }

    fun setResults(r: FaceLandmarkerResult, w: Int, h: Int) {
        result = r
        imageWidth = w
        imageHeight = h
        postInvalidate()
    }

    fun clear() {
        result = null
        postInvalidate()
    }

    private fun scaleFactor(): Float =
        maxOf(width.toFloat() / imageWidth, height.toFloat() / imageHeight)

    private fun scaleX(x: Float): Float {
        val sx = if (isMirrored) 1f - x else x
        val s = scaleFactor()
        return sx * imageWidth * s - (imageWidth * s - width) / 2f
    }

    private fun scaleY(y: Float): Float {
        val s = scaleFactor()
        return y * imageHeight * s - (imageHeight * s - height) / 2f
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val r = result ?: return
        if (r.faceLandmarks().isEmpty()) return
        val landmarks = r.faceLandmarks()[0]

        // Eye contours
        for (indices in arrayOf(LEFT_EYE_INDICES, RIGHT_EYE_INDICES)) {
            val path = Path()
            for ((i, idx) in indices.withIndex()) {
                val x = scaleX(landmarks[idx].x())
                val y = scaleY(landmarks[idx].y())
                if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            path.close()
            canvas.drawPath(path, eyeContourPaint)
        }

        // Iris centers
        for (idx in intArrayOf(LEFT_IRIS_INDICES[0], RIGHT_IRIS_INDICES[0])) {
            canvas.drawCircle(scaleX(landmarks[idx].x()), scaleY(landmarks[idx].y()), 6f, irisPaint)
        }

        // Iris contour points
        for (indices in arrayOf(LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES)) {
            for (i in 1 until indices.size) {
                canvas.drawCircle(
                    scaleX(landmarks[indices[i]].x()),
                    scaleY(landmarks[indices[i]].y()),
                    4f, irisPaint
                )
            }
        }

        // Nose tip (face center reference)
        canvas.drawCircle(
            scaleX(landmarks[NOSE_TIP].x()),
            scaleY(landmarks[NOSE_TIP].y()),
            5f, nosePaint
        )
    }
}
