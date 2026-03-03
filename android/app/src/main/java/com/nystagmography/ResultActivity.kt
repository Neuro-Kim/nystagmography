package com.nystagmography

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.*

class ResultActivity : AppCompatActivity() {

    private lateinit var graphHorizontal: GraphView
    private lateinit var graphVertical: GraphView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)

        graphHorizontal = findViewById(R.id.graphHorizontal)
        graphVertical = findViewById(R.id.graphVertical)
        val tvAnalysis = findViewById<TextView>(R.id.tvAnalysis)

        val d = NystagmusData
        if (d.timestamps.size < 10) {
            tvAnalysis.text = "Not enough data."
            return
        }

        // Horizontal eye movement graph (inverted Y to match visual left-right)
        graphHorizontal.setInvertY(true)
        graphHorizontal.setData(
            "Horizontal Eye Movement",
            "Relative X (eye-width normalized)",
            d.timestamps,
            GraphView.Series(d.leftEyeXRel, Color.RED, "Left Eye X"),
            GraphView.Series(d.rightEyeXRel, Color.BLUE, "Right Eye X")
        )

        // Vertical eye movement graph
        graphVertical.setData(
            "Vertical Eye Movement",
            "Relative Y (eye-height normalized)",
            d.timestamps,
            GraphView.Series(d.leftEyeYRel, Color.RED, "Left Eye Y"),
            GraphView.Series(d.rightEyeYRel, Color.BLUE, "Right Eye Y")
        )

        // Basic analysis
        tvAnalysis.text = analyzeNystagmus(d)

        findViewById<Button>(R.id.btnSavePng).setOnClickListener { saveGraphsAsPng() }
    }

    private fun saveGraphsAsPng() {
        val w = graphHorizontal.width
        val h1 = graphHorizontal.height
        val h2 = graphVertical.height
        if (w == 0 || h1 == 0 || h2 == 0) {
            Toast.makeText(this, "Graph not ready", Toast.LENGTH_SHORT).show()
            return
        }

        val combined = Bitmap.createBitmap(w, h1 + h2, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(combined)
        canvas.drawColor(Color.WHITE)

        val b1 = Bitmap.createBitmap(w, h1, Bitmap.Config.ARGB_8888)
        graphHorizontal.draw(Canvas(b1))
        canvas.drawBitmap(b1, 0f, 0f, null)
        b1.recycle()

        val b2 = Bitmap.createBitmap(w, h2, Bitmap.Config.ARGB_8888)
        graphVertical.draw(Canvas(b2))
        canvas.drawBitmap(b2, 0f, h1.toFloat(), null)
        b2.recycle()

        val filename = "nystagmus_graph_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())}.png"

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)
            }
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            if (uri != null) {
                contentResolver.openOutputStream(uri)?.use { out ->
                    combined.compress(Bitmap.CompressFormat.PNG, 100, out)
                }
                Toast.makeText(this, "Saved to Pictures/$filename", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(this, "Failed to save image", Toast.LENGTH_SHORT).show()
            }
        } else {
            val dir = getExternalFilesDir(Environment.DIRECTORY_PICTURES) ?: filesDir
            dir.mkdirs()
            val file = File(dir, filename)
            FileOutputStream(file).use { out ->
                combined.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            Toast.makeText(this, "Saved: ${file.absolutePath}", Toast.LENGTH_LONG).show()
        }

        combined.recycle()
    }

    private fun analyzeNystagmus(d: NystagmusData): String {
        val data = d.leftEyeXRel
        if (data.size < 10) return "Not enough data for analysis."

        // Detrend: remove linear trend
        val n = data.size
        val t = d.timestamps
        val meanT = t.average().toFloat()
        val meanX = data.average().toFloat()
        var num = 0f
        var den = 0f
        for (i in 0 until n) {
            num += (t[i] - meanT) * (data[i] - meanX)
            den += (t[i] - meanT) * (t[i] - meanT)
        }
        val slope = if (den > 0) num / den else 0f
        val detrended = FloatArray(n) { data[it] - (slope * (t[it] - meanT) + meanX) }

        // Amplitude (standard deviation)
        val amplitude = sqrt(detrended.map { it * it }.average()).toFloat()

        // Estimate frequency via zero crossings
        var crossings = 0
        for (i in 1 until n) {
            if (detrended[i - 1] * detrended[i] < 0) crossings++
        }
        val duration = t.last() - t.first()
        val freqEstimate = if (duration > 0) crossings / (2f * duration) else 0f

        val sb = StringBuilder()
        sb.appendLine("=== Nystagmus Analysis ===")
        sb.appendLine("Duration: %.1f s".format(duration))
        sb.appendLine("Samples: $n")
        sb.appendLine("Dominant Frequency: %.2f Hz".format(freqEstimate))
        sb.appendLine("Average Amplitude: %.4f".format(amplitude))
        sb.appendLine()

        when {
            freqEstimate < 2f ->
                sb.appendLine("Low-frequency: possibly vestibular nystagmus")
            freqEstimate <= 5f ->
                sb.appendLine("Mid-frequency: possibly congenital nystagmus")
            else ->
                sb.appendLine("High-frequency: possibly drug-induced or central nystagmus")
        }

        return sb.toString()
    }
}
