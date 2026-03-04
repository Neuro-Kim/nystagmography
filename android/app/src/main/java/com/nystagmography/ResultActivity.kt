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

class ResultActivity : AppCompatActivity() {

    private lateinit var graphHorizontal: GraphView
    private lateinit var graphVertical: GraphView
    private lateinit var graphSPV: GraphView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)

        graphHorizontal = findViewById(R.id.graphHorizontal)
        graphVertical = findViewById(R.id.graphVertical)
        graphSPV = findViewById(R.id.graphSPV)
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

        // Run SPV analysis
        val result = NystagmusAnalyzer.analyze(d.leftEyeXRel, d.timestamps)

        // Beat markers on horizontal graph (orange dashed lines at beat start times)
        val markerColor = Color.parseColor("#FF9500")
        val markers = result.beats.map { GraphView.Marker(it.startTime, markerColor) }
        graphHorizontal.setMarkers(markers)

        // SPV graph: plot each beat's SPV as a step over its time span
        if (result.beats.isNotEmpty()) {
            val spvTimestamps = mutableListOf<Float>()
            val spvValues = mutableListOf<Float>()
            for (beat in result.beats) {
                spvTimestamps.add(beat.startTime)
                spvValues.add(beat.spv)
                spvTimestamps.add(beat.endTime)
                spvValues.add(beat.spv)
            }
            graphSPV.setData(
                "Slow Phase Velocity (SPV)",
                "SPV (normalized/s)",
                spvTimestamps.toFloatArray(),
                GraphView.Series(spvValues.toFloatArray(), Color.parseColor("#FF9500"), "SPV")
            )
        }

        // Analysis text
        tvAnalysis.text = formatResults(result)

        findViewById<Button>(R.id.btnSavePng).setOnClickListener { saveGraphsAsPng() }
        findViewById<Button>(R.id.btnBack).setOnClickListener { finish() }
    }

    private fun formatResults(r: NystagmusAnalyzer.AnalysisResult): String {
        val sb = StringBuilder()
        sb.appendLine("=== Nystagmus Analysis ===")
        sb.appendLine("Duration: %.1f s".format(r.duration))
        sb.appendLine("Samples: ${r.sampleCount}")
        sb.appendLine("Dominant Frequency: %.2f Hz".format(r.dominantFrequency))
        sb.appendLine("Average Amplitude: %.4f".format(r.amplitude))
        sb.appendLine()
        sb.appendLine("=== Beat Detection ===")
        sb.appendLine("Beats detected: ${r.beatCount}")
        sb.appendLine("Beat frequency: %.2f /s".format(r.beatFrequency))
        sb.appendLine("Mean SPV: %.4f /s".format(r.meanSPV))
        sb.appendLine("Direction: ${r.direction}")
        sb.appendLine()

        when {
            r.dominantFrequency < 2f ->
                sb.appendLine("Low-frequency: possibly vestibular nystagmus")
            r.dominantFrequency <= 5f ->
                sb.appendLine("Mid-frequency: possibly congenital nystagmus")
            else ->
                sb.appendLine("High-frequency: possibly drug-induced or central nystagmus")
        }

        return sb.toString()
    }

    private fun saveGraphsAsPng() {
        val w = graphHorizontal.width
        val h1 = graphHorizontal.height
        val h2 = graphVertical.height
        val h3 = graphSPV.height
        if (w == 0 || h1 == 0 || h2 == 0 || h3 == 0) {
            Toast.makeText(this, "Graph not ready", Toast.LENGTH_SHORT).show()
            return
        }

        val combined = Bitmap.createBitmap(w, h1 + h2 + h3, Bitmap.Config.ARGB_8888)
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

        val b3 = Bitmap.createBitmap(w, h3, Bitmap.Config.ARGB_8888)
        graphSPV.draw(Canvas(b3))
        canvas.drawBitmap(b3, 0f, (h1 + h2).toFloat(), null)
        b3.recycle()

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
}
