package me.lmx.knn.hnsw


import kotlinx.coroutines.runBlocking
import me.lmx.knn.Item
import me.lmx.knn.SearchResult
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import java.util.*

internal class HNSWIndexTest {
    lateinit var index: HNSWIndex

    val dimensions: Int = 2
    val maxItemCount: Int = 100
    val m: Int = 12
    val ef: Int = 20
    val efConstruction: Int = 200

    val item1: Item = Item("1", floatArrayOf(0.1f, 0.2f), 1)
    val item2: Item = Item("2", floatArrayOf(0.2f, 0.3f), 1)
    val item3: Item = Item("3", floatArrayOf(0.4f, 0.9f), 1)

    @BeforeEach
    fun setUp() {
        index = HNSWIndex(
            dimensions = dimensions,
            maxItemCount = maxItemCount,
            m = m,
            ef = ef,
            efConstruction = efConstruction,
            removeEnabled = true
        )
    }

    @Test
    fun returnDimensions() {
        assert(index.dimensions == dimensions)
    }

    @Test
    fun returnM() {
        assert(index.m == m)
    }

    @Test
    fun returnEf() {
        assert(index.ef == ef)
    }

    @Test
    fun changeEf() {
        val newEfValue = 100
        index.ef = newEfValue
        assert(index.ef == newEfValue)
    }

    @Test
    fun returnEfConstruction() {
        assert(index.efConstruction == efConstruction)
    }

    @Test
    fun returnMaxItemCount() {
        assert(index.maxItemCount == maxItemCount)
    }

    @Test
    fun returnsSize() = runBlocking {
        assert(index.size() == 0)
        index.add(item1)
        assert(index.size() == 1)
    }

    @Test
    fun addAndGet() = runBlocking {
        assert(index.get(item1.id) == null)
        index.add(item1)
        assert(index.get(item1.id) == item1)
    }

    @Test
    fun addAndContains() = runBlocking {
        assert(!index.contains(item1.id))
        index.add(item1)
        assert(index.contains(item1.id))
    }

    @Test
    fun returnsItems() = runBlocking {
        assert(index.items().isEmpty())
        index.add(item1)
        assert(index.items().size == 1)
        assert(index.items().contains(item1))
    }

    @Test
    fun removeItem() = runBlocking {
        index.add(item1)
        assert(index.remove(item1.id, item1.version))
        assert(index.size() == 0)
        assert(index.items().isEmpty())
        assert(index.get(item1.id) == null)
        assert(index.exactView.size() == 0)
        assert(index.exactView.items().isEmpty())
        assert(index.exactView.get(item1.id) == null)
    }

    @Test
    fun addNewerItem() = runBlocking {
        val newerItem = Item(item1.id, floatArrayOf(0f, 0f), item1.version + 1)
        index.add(item1)
        index.add(newerItem)
        assert(index.size() == 1)
        assert(index.get(item1.id) == newerItem)
    }

    @Test
    fun addOlderItem() = runBlocking {
        val olderItem = Item(item1.id, floatArrayOf(0f, 0f), item1.version - 1)
        index.add(item1)
        index.add(olderItem)
        assert(index.size() == 1)
        assert(index.get(item1.id) == item1)
    }

    @Test
    fun removeUnknownItem() = runBlocking {
        index.add(item1)
        assert(!index.remove("foo", 0))
    }

    @Test
    fun removeWithOldVersionIgnored() = runBlocking {
        index.add(item1)
        assert(!index.remove(item1.id, item1.version - 1))
        assert(index.size() == 1)
    }

    @Test
    fun findNearest() = runBlocking {
        index.addAll(listOf(item1, item2, item3))
        val nearest = index.findNearest(item1.vector, 10)
        println(nearest)
    }

    @Test
    fun findNeighbors() = runBlocking {
        index.addAll(listOf(item1, item2, item3))
        val nearest = index.findNeighbors(item1.id, 10)
        println(nearest)
    }
}