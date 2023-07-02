package me.lmx.knn.util

import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlin.coroutines.coroutineContext

class ReentrantMutex {
    private val mutex: Mutex = Mutex()
    private var owner: Job? = null
    private var counter: Int = 0

    suspend fun lock() {
        coroutineContext[Job]?.let {
            if (owner == it) {
                counter++
            } else {
                mutex.lock()
                owner = it
                counter++
            }
        }?: throw RuntimeException("Job cannot be null!")
    }

    suspend fun unlock() {
        coroutineContext[Job]?.let {
            if (owner == it) {
                counter--
                if (counter == 0) {
                    owner = null
                    mutex.unlock()
                }
            } else {
                throw RuntimeException("Mutex is not locked or locked by other owner!")
            }
        }?: throw RuntimeException("Job cannot be null!")
    }

    suspend inline fun <T> withLock(action: () -> T) : T {
        lock()
        try {
            return action()
        } finally {
            unlock()
        }
    }

    suspend fun holdsLock(): Boolean {
        coroutineContext[Job]?.let {
            return it == owner
        } ?: throw RuntimeException("Job cannot be null!")
    }
}
