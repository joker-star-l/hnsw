package me.lmx.knn

import kotlinx.coroutines.*
import java.io.Serializable

interface Index : Serializable {

    suspend fun add(item: Item): Boolean

    suspend fun remove(id: String, version: Long): Boolean

    suspend fun contains(id: String): Boolean = get(id) != null

    suspend fun addAll(items: Collection<Item>) {
        val scope = CoroutineScope(Dispatchers.Default)
        val jobs = mutableListOf<Job>()
        items.forEach {
            jobs.add(scope.launch {
                add(it)
            })
        }
        jobs.joinAll()
    }

    suspend fun size(): Int

    suspend fun get(id: String): Item?

    suspend fun items(): Collection<Item>

    suspend fun findNearest(vector: FloatArray, k: Int): List<SearchResult>

    suspend fun findNeighbors(id: String, k: Int): List<SearchResult> = get(id)?.let { item ->
        findNearest(item.vector, k + 1)
            .asSequence()
            .filter { result -> result.item().id != id }
            .take(k)
            .toList()
    } ?: emptyList()
}