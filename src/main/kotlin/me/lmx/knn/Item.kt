package me.lmx.knn

import java.io.Serializable

data class Item(
    var id: String,
    var vector: FloatArray,
    var version: Long = 0
    ) : Serializable {
    val dimensions: Int
        get() = vector.size

    companion object {
        const val serialVersionUID: Long = 1
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Item) return false

        if (id != other.id) return false
        if (!vector.contentEquals(other.vector)) return false
        if (version != other.version) return false

        return true
    }

    override fun hashCode(): Int {
        var result = id.hashCode()
        result = 31 * result + vector.contentHashCode()
        result = 31 * result + version.hashCode()
        return result
    }
}