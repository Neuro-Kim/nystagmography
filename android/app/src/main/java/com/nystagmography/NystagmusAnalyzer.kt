package com.nystagmography

import kotlin.math.abs

object NystagmusAnalyzer {

    data class Beat(
        val startTime: Float,
        val endTime: Float,
        val spv: Float,
        val direction: String // "left" or "right"
    )

    data class AnalysisResult(
        val meanSPV: Float,
        val direction: String, // "left-beating", "right-beating", "indeterminate"
        val beatCount: Int,
        val beatFrequency: Float, // beats per second
        val beats: List<Beat>,
        val velocity: FloatArray,
        val dominantFrequency: Float,
        val amplitude: Float,
        val duration: Float,
        val sampleCount: Int
    )

    fun analyze(positions: FloatArray, timestamps: FloatArray): AnalysisResult {
        val n = positions.size
        val duration = timestamps.last() - timestamps.first()

        // 1. Velocity calculation: v[i] = (pos[i+1] - pos[i]) / (t[i+1] - t[i])
        val rawVelocity = FloatArray(n - 1) { i ->
            val dt = (timestamps[i + 1] - timestamps[i]).coerceAtLeast(0.001f)
            (positions[i + 1] - positions[i]) / dt
        }

        // 2. Median filter (window=5) for noise removal
        val velocity = medianFilter(rawVelocity, 5)

        // 3. Fast phase detection: adaptive threshold = 3 × median(|velocity|)
        val absVelocities = FloatArray(velocity.size) { abs(velocity[it]) }
        val medianAbsVel = median(absVelocities)
        val threshold = (3f * medianAbsVel).coerceAtLeast(0.01f)

        val isFast = BooleanArray(velocity.size) { abs(velocity[it]) > threshold }

        // 4. Phase cleanup: fast < 2 samples → noise, slow < 3 samples → bridge
        cleanupPhases(isFast, minFast = 2, minSlow = 3)

        // 5. Beat extraction: each slow phase segment → one beat with mean velocity = SPV
        val beats = mutableListOf<Beat>()
        var i = 0
        while (i < isFast.size) {
            if (!isFast[i]) {
                val start = i
                while (i < isFast.size && !isFast[i]) i++
                val end = i // exclusive
                val meanVel = velocity.slice(start until end).average().toFloat()
                val dir = if (meanVel > 0) "right" else "left"
                beats.add(Beat(
                    startTime = timestamps[start],
                    endTime = timestamps[(end).coerceAtMost(timestamps.size - 1)],
                    spv = abs(meanVel),
                    direction = dir
                ))
            } else {
                i++
            }
        }

        // 6. Summary statistics
        val meanSPV = if (beats.isNotEmpty()) beats.map { it.spv }.average().toFloat() else 0f
        val leftCount = beats.count { it.direction == "left" }
        val rightCount = beats.count { it.direction == "right" }
        val direction = when {
            beats.isEmpty() -> "indeterminate"
            leftCount > rightCount -> "left-beating"
            rightCount > leftCount -> "right-beating"
            else -> "indeterminate"
        }
        val beatFrequency = if (duration > 0) beats.size / duration else 0f

        // Detrend + amplitude/frequency (preserved from original analysis)
        val meanT = timestamps.average().toFloat()
        val meanX = positions.average().toFloat()
        var num = 0f; var den = 0f
        for (j in 0 until n) {
            num += (timestamps[j] - meanT) * (positions[j] - meanX)
            den += (timestamps[j] - meanT) * (timestamps[j] - meanT)
        }
        val slope = if (den > 0) num / den else 0f
        val detrended = FloatArray(n) { positions[it] - (slope * (timestamps[it] - meanT) + meanX) }
        val amplitude = kotlin.math.sqrt(detrended.map { it.toDouble() * it }.average()).toFloat()
        var crossings = 0
        for (j in 1 until n) {
            if (detrended[j - 1] * detrended[j] < 0) crossings++
        }
        val freqEstimate = if (duration > 0) crossings / (2f * duration) else 0f

        return AnalysisResult(
            meanSPV = meanSPV,
            direction = direction,
            beatCount = beats.size,
            beatFrequency = beatFrequency,
            beats = beats,
            velocity = velocity,
            dominantFrequency = freqEstimate,
            amplitude = amplitude,
            duration = duration,
            sampleCount = n
        )
    }

    private fun medianFilter(data: FloatArray, windowSize: Int): FloatArray {
        val half = windowSize / 2
        return FloatArray(data.size) { i ->
            val from = (i - half).coerceAtLeast(0)
            val to = (i + half).coerceAtMost(data.size - 1)
            median(data.sliceArray(from..to))
        }
    }

    private fun median(arr: FloatArray): Float {
        if (arr.isEmpty()) return 0f
        val sorted = arr.sorted()
        val mid = sorted.size / 2
        return if (sorted.size % 2 == 0) (sorted[mid - 1] + sorted[mid]) / 2f else sorted[mid]
    }

    private fun cleanupPhases(isFast: BooleanArray, minFast: Int, minSlow: Int) {
        // Remove fast runs shorter than minFast
        var i = 0
        while (i < isFast.size) {
            if (isFast[i]) {
                val start = i
                while (i < isFast.size && isFast[i]) i++
                if (i - start < minFast) {
                    for (j in start until i) isFast[j] = false
                }
            } else {
                i++
            }
        }
        // Bridge slow gaps shorter than minSlow (mark as fast to merge)
        i = 0
        while (i < isFast.size) {
            if (!isFast[i]) {
                val start = i
                while (i < isFast.size && !isFast[i]) i++
                if (i - start < minSlow) {
                    for (j in start until i) isFast[j] = true
                }
            } else {
                i++
            }
        }
    }
}
