package me.lmx.knn

import java.io.Serializable

data class SearchResult(
    var distance: Float,
    var item: Item
    ) : Comparable<SearchResult>, Serializable {

    companion object {
        const val serialVersionUID: Long = 1
    }

    fun item(): Item = item

    fun distance(): Float = distance

    override fun compareTo(other: SearchResult): Int = distance.compareTo(other.distance)
}