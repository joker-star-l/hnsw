package me.lmx.knn.hnsw

import me.lmx.knn.Index
import me.lmx.knn.Item
import me.lmx.knn.SearchResult
import me.lmx.knn.util.ReentrantMutex
import java.io.Serializable
import java.util.*
import java.util.concurrent.atomic.AtomicReferenceArray
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random

data class HNSWIndex(
    val dimensions: Int, // 向量维度
    val maxItemCount: Int, // 索引容量
    val m: Int = 10, // 插入节点时建立的连接数
    val maxM: Int = m, // 非 layer0 节点最大可以建立的连接数
    val maxM0: Int = m * 2, // layer0 节点最大可以建立的连接数
    val levelLambda: Double = 1 / ln(m.toDouble()), // 计算节点层级的归一化参数
    var ef: Int = 10, // 需要搜索的近邻数，仅在检索时起作用，ef >= k
    var efConstruction: Int = 200, // 需要搜索的近邻数，仅在插入时起作用
    val removeEnabled: Boolean = false, // 是否允许删除
    val lookup: MutableMap<String, Int> = HashMap(), // id 到节点数组下标的映射
    val deletedItemVersions: MutableMap<String, Long> = HashMap(), // 已删除节点 id 到版本号的映射
    ) : Index {
    val nodes: AtomicReferenceArray<Node> = AtomicReferenceArray(maxItemCount) // 节点数组
    var nodeCount: Int = 0 // 节点数量
    @Volatile var entryPoint: Node? = null // 图入口
    val excludedCandidates: BitSet = BitSet(maxItemCount) // 位图，用于并发控制

    val exactView: ExactView = ExactView() // 提供暴力准确检索能力的索引

    val globalMutex: ReentrantMutex = ReentrantMutex() // 全局锁
    val mutexes: MutableMap<String, ReentrantMutex> = HashMap() // 节点锁
    val excludedCandidatesMutex: ReentrantMutex = ReentrantMutex() // 位图锁

    val calDistance: (FloatArray, FloatArray) -> Float = { u, v ->
        var sum = 0f
        for (i in u.indices) {
            val dp = u[i] - v[i]
            sum += dp * dp
        }
        sqrt(sum.toDouble()).toFloat()
    } // 欧氏距离

    init {
        efConstruction = max(efConstruction, m)
    }

    companion object {
        const val serialVersionUID: Long = 1

        fun assignLevel(lambda: Double): Int = (-ln(Random.nextDouble()) * lambda).toInt() // 节点层级分布概率函数
    }

    override suspend fun size(): Int {
        globalMutex.withLock {
            return lookup.size
        }
    }

    override suspend fun get(id: String): Item? {
        globalMutex.withLock {
            val idx = lookup[id]
            return idx?.let {
                nodes[idx].item
            }
        }
    }

    override suspend fun items(): Collection<Item> {
        globalMutex.withLock {
            val results = ArrayList<Item>(size())
            val iterator = ItemIterator()
            while (iterator.hasNext()) {
                results.add(iterator.next())
            }
            return results
        }
    }

    override suspend fun remove(id: String, version: Long): Boolean {
        if (!removeEnabled) {
            return false
        }

        globalMutex.withLock {
            val idx = lookup[id]
            val node = idx?.let {
                nodes[idx]
            }
            node?.let {
                if (version < node.item.version) {
                    return false
                }
                node.deleted = true
                lookup.remove(id)
                deletedItemVersions[id] = version
                return true
            } ?: let {
                return false
            }
        }
    }

    override suspend fun add(item: Item): Boolean {
        if (item.dimensions != dimensions) {
            throw RuntimeException("Dimensions error!")
        }

        val randomLevel = assignLevel(levelLambda)
        val connections = Array<MutableList<Int>>(randomLevel + 1) {
            val levelM = if (it == 0) maxM0 else maxM
            ArrayList(levelM)
        }

        globalMutex.lock()
        try {
            val idx = lookup[item.id]
            idx?.let {
                if (!removeEnabled) {
                    return false
                }

                val node = nodes[idx]
                if (item.version < node.item.version) {
                    return false
                }

                if (node.item.vector.contentEquals(item.vector)) {
                    node.item = item
                    return true
                } else {
                    remove(item.id, item.version)
                }
            } ?: let {
                if (item.version < deletedItemVersions[item.id] ?: -1) {
                    return false
                }
            }

            if (nodeCount >= maxItemCount) {
                throw RuntimeException("The number of elements exceeds the limit!")
            }

            val newIdx = nodeCount++

            excludedCandidatesMutex.withLock {
                excludedCandidates.set(newIdx)
            }

            val newNode = Node(newIdx, connections, item, false)

            nodes.set(newIdx, newNode)
            lookup[item.id] = newIdx
            deletedItemVersions.remove(item.id)

            val mutex = mutexes.computeIfAbsent(item.id) { ReentrantMutex() }
            val entryPointCopy = entryPoint

            try {
                mutex.withLock {
                    newNode.withLock {
                        entryPoint?.let {
                            if (randomLevel <= it.maxLevel()) {
                                globalMutex.unlock()
                            }
                        }

                        entryPointCopy?.let {
                            var currNode = it
                            if (randomLevel < entryPointCopy.maxLevel()) { // search
                                var currDistance = calDistance(item.vector, currNode.item.vector)
                                for (activeLevel in entryPointCopy.maxLevel() downTo randomLevel + 1) {
                                    var changed = true
                                    while (changed) {
                                        changed = false
                                        currNode.withLock {
                                            val candidateConnections = currNode.connections[activeLevel]
                                            for (candidateId in candidateConnections) {
                                                val candidateNode = nodes[candidateId]
                                                val candidateDistance = calDistance(item.vector, candidateNode.item.vector)
                                                if (candidateDistance < currDistance) {
                                                    currDistance = candidateDistance
                                                    currNode = candidateNode
                                                    changed = true
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            for (level in min(randomLevel, entryPointCopy.maxLevel()) downTo 0) {
                                val topCandidates = searchBaseLayer(currNode, item.vector, efConstruction, level)
                                mutuallyConnectNewElement(newNode, topCandidates, level)
                            }
                        }

                        if (entryPoint == null || randomLevel > entryPointCopy?.maxLevel() ?: Int.MAX_VALUE) {
                            entryPoint = newNode
                        }

                        return true
                    }
                }
            } finally {
                excludedCandidatesMutex.withLock {
                    excludedCandidates.clear(newIdx)
                }
            }
        } finally {
            if (globalMutex.holdsLock()) {
                globalMutex.unlock()
            }
        }
    }

    private suspend fun mutuallyConnectNewElement(newNode: Node, topCandidates: PriorityQueue<NodeIdxAndDistance>, level: Int)  {
        val n = if (level == 0) maxM0 else maxM
        val newNodeVector = newNode.item.vector
        val newNodeConnections = newNode.connections[level]
        while (topCandidates.size > 0) {
            val selectedNeighbor = topCandidates.poll()

            if (excludedCandidatesMutex.withLock { excludedCandidates[selectedNeighbor.nodeId] }) {
                continue
            }

            newNodeConnections.add(selectedNeighbor.nodeId)
            val neighborNode = nodes[selectedNeighbor.nodeId]
            neighborNode.withLock {
                val neighborNodeVector = neighborNode.item.vector
                val neighborNodeConnections = neighborNode.connections[level]
                if (neighborNodeConnections.size < n) {
                    neighborNodeConnections.add(newNode.id)
                } else {
                    val distance = calDistance(newNodeVector, neighborNodeVector)
                    val candidates = PriorityQueue<NodeIdxAndDistance>(Comparator.naturalOrder<NodeIdxAndDistance>().reversed())
                    candidates.add(NodeIdxAndDistance(newNode.id, distance))
                    neighborNodeConnections.forEach { id ->
                        candidates.add(NodeIdxAndDistance(id, calDistance(neighborNodeVector, nodes[id].item.vector)))
                    }
                    getNeighborsByHeuristic(candidates, n)
                    neighborNodeConnections.clear()
                    while (candidates.size > 0) {
                        neighborNodeConnections.add(candidates.poll().nodeId)
                    }
                }
            }
        }
    }

    private fun getNeighborsByHeuristic(topCandidates: PriorityQueue<NodeIdxAndDistance>, m: Int) {
        if (topCandidates.size <= m) {
            return
        }

        val queueClosest = PriorityQueue<NodeIdxAndDistance>()
        val returnList = mutableListOf<NodeIdxAndDistance>()
        while (topCandidates.size > 0) {
            queueClosest.add(topCandidates.poll())
        }
        while (queueClosest.size > 0) {
            if (returnList.size >= m) {
                break
            }

            val currPair = queueClosest.poll()
            var good = true
            for (secondPair in returnList) {
                val distance = calDistance(
                    nodes[secondPair.nodeId].item.vector,
                    nodes[currPair.nodeId].item.vector
                )
                if (distance < currPair.distance) {
                    good = false
                    break
                }
            }
            if (good) {
                returnList.add(currPair)
            }
        }

        topCandidates.addAll(returnList)
    }

    override suspend fun findNearest(destination: FloatArray, k: Int): List<SearchResult> {
        entryPoint?.let {
            val entryPointCopy = it
            var currNode = entryPointCopy
            var currDistance = calDistance(destination, currNode.item.vector)
            for (activeLevel in entryPointCopy.maxLevel() downTo 1) {
                var changed = true
                while (changed) {
                    changed = false
                    currNode.withLock {
                        val candidateConnections = currNode.connections[activeLevel]
                        for (candidateId in candidateConnections) {
                            val candidateDistance = calDistance(destination, nodes[candidateId].item.vector)
                            if (candidateDistance < currDistance) {
                                currDistance = candidateDistance
                                currNode = nodes[candidateId]
                                changed = true
                            }
                        }
                    }
                }
            }
            val topCandidates = searchBaseLayer(currNode, destination, max(ef, k), 0)
            while (topCandidates.size > k) {
                topCandidates.poll()
            }
            val results = ArrayList<SearchResult>(topCandidates.size)
            while (topCandidates.size > 0) {
                val pair = topCandidates.poll()
                results.add(0, SearchResult(pair.distance, nodes[pair.nodeId].item))
            }
            return results
        } ?: let {
            return emptyList()
        }
    }

    private suspend fun searchBaseLayer(entryPointNode: Node, destination: FloatArray, k: Int, layer: Int): PriorityQueue<NodeIdxAndDistance> {
        val visitedBitSet = BitSet(maxItemCount)
        val topCandidates = PriorityQueue<NodeIdxAndDistance>(Comparator.naturalOrder<NodeIdxAndDistance>().reversed())
        val candidateSet = PriorityQueue<NodeIdxAndDistance>()

        var lowerBound: Float
        if (entryPointNode.deleted) {
            lowerBound = Float.MAX_VALUE
            val pair = NodeIdxAndDistance(entryPointNode.id, lowerBound)
            candidateSet.add(pair)
        } else {
            lowerBound = calDistance(destination, entryPointNode.item.vector)
            val pair = NodeIdxAndDistance(entryPointNode.id, lowerBound)
            topCandidates.add(pair)
            candidateSet.add(pair)
        }
        visitedBitSet.set(entryPointNode.id)

        while (!candidateSet.isEmpty()) {
            val currPair = candidateSet.poll()
            if (currPair.distance > lowerBound) {
                break
            }

            val node = nodes[currPair.nodeId]
            node.withLock {
                val candidates = node.connections[layer]
                for (candidateId in candidates) {
                    if (!visitedBitSet[candidateId]) {
                        visitedBitSet.set(candidateId)
                        val candidateNode = nodes[candidateId]
                        val candidateDistance = calDistance(destination, candidateNode.item.vector)
                        if (topCandidates.size < k || lowerBound > candidateDistance) {
                            val candidatePair = NodeIdxAndDistance(candidateId, candidateDistance)
                            candidateSet.add(candidatePair)
                            if (!candidateNode.deleted) {
                                topCandidates.add(candidatePair)
                            }
                            if (topCandidates.size > k) {
                                topCandidates.poll()
                            }
                            if (topCandidates.size > 0) {
                                lowerBound = topCandidates.peek().distance
                            }
                        }
                    }
                }
            }
        }

        return topCandidates
    }

    inner class ExactView : Index {

        override suspend fun size(): Int = this@HNSWIndex.size()

        override suspend fun get(id: String): Item? = this@HNSWIndex.get(id)

        override suspend fun items(): Collection<Item> = this@HNSWIndex.items()

        override suspend fun findNearest(vector: FloatArray, k: Int): List<SearchResult> {
            val queue = PriorityQueue<SearchResult>(k, Comparator.naturalOrder<SearchResult>().reversed())

            for (i in 0 until nodeCount) {
                val node = nodes[i]
                if (node == null || node.deleted) {
                    continue
                }
                val distance = calDistance(node.item.vector, vector)
                val searchResult = SearchResult(distance, node.item)
                queue.add(searchResult)
                if (queue.size > k) {
                    queue.poll()
                }
            }

            val results = ArrayList<SearchResult>(queue.size)

            while (queue.size > 0) {
                results.add(0, queue.poll())
            }

            return results
        }

        override suspend fun add(item: Item): Boolean = this@HNSWIndex.add(item)

        override suspend fun remove(id: String, version: Long): Boolean = this@HNSWIndex.remove(id, version)
    }

    inner class ItemIterator(
        var done : Int = 0,
        var index: Int = 0
        ) : Iterator<Item> {

        override fun hasNext(): Boolean = done < lookup.size

        /**
         * @throws IndexOutOfBoundsException
         */
        override fun next(): Item {
            var node: Node
            do {
                node = nodes.get(index++)
            } while (node.deleted)
            done++
            return node.item
        }
    }

    data class Node(
        var id: Int,
        var connections: Array<MutableList<Int>>,
        @Volatile var item: Item,
        @Volatile var deleted: Boolean
        ) : Serializable {
        val mutex: ReentrantMutex = ReentrantMutex()

        companion object {
            const val serialVersionUID: Long = 1
        }

        fun maxLevel(): Int = connections.size - 1

        suspend inline fun <T> withLock(action: () -> T) : T {
            return mutex.withLock(action)
        }

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Node) return false

            if (id != other.id) return false
            if (!connections.contentEquals(other.connections)) return false
            if (item != other.item) return false
            if (deleted != other.deleted) return false

            return true
        }

        override fun hashCode(): Int {
            var result = id
            result = 31 * result + connections.contentHashCode()
            result = 31 * result + item.hashCode()
            result = 31 * result + deleted.hashCode()
            return result
        }
    }

    data class NodeIdxAndDistance(
        val nodeId: Int,
        val distance: Float
        ) : Comparable<NodeIdxAndDistance> {

        override fun compareTo(other: NodeIdxAndDistance): Int = distance.compareTo(other.distance)
    }
}