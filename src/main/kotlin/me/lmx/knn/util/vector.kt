package me.lmx.knn.util

import kotlin.math.sqrt

fun magnitude(vector: FloatArray): Float {
    var magnitude = 0.0f
    for (aFloat in vector) {
        magnitude += aFloat * aFloat
    }
    return sqrt(magnitude.toDouble()).toFloat()
}

fun normalize(vector: FloatArray): FloatArray {
    val result = FloatArray(vector.size)
    val normFactor = 1 / magnitude(vector)
    for (i in vector.indices) {
        result[i] = vector[i] * normFactor
    }
    return result
}