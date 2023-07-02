package me.lmx.knn.example

import kotlinx.coroutines.runBlocking
import me.lmx.knn.Item
import me.lmx.knn.SearchResult
import me.lmx.knn.hnsw.HNSWIndex
import me.lmx.knn.util.normalize
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.URL
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.*
import java.util.stream.Collectors
import java.util.zip.GZIPInputStream
import kotlin.system.measureTimeMillis

const val WORDS_FILE_URL: String = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"
val TMP_PATH: Path = Paths.get(System.getProperty("java.io.tmpdir"))

fun main(args: Array<String>) = runBlocking {
    val k = args[0].toInt()
    val size = if (args.size == 2) args[1].toLong() else 2000000

    val file = TMP_PATH.resolve("cc.en.300.vec.gz")
    if (!Files.exists(file)) {
        downloadFile(WORDS_FILE_URL, file)
    } else {
        println("Input file already downloaded. Using $file")
    }

    val words = loadWordVectors(file, size)

    println("Constructing index.")

    val index = HNSWIndex(
        dimensions = 300,
        maxItemCount = words.size,
        m = 16,
        ef = 200,
        efConstruction = 200
    )

    val duration = measureTimeMillis {
        index.addAll(words)
    }

    println("Creating index with ${index.size()} words took ${"%.3f".format(duration / 1000.0)} seconds which is ${"%.3f".format(duration / 1000.0 / 60)} minutes.")

    val exactIndex = index.exactView

    val scanner = Scanner(System.`in`)

    while (true) {
        println("Enter an english word: ")

        val input: String = scanner.nextLine()

        val approximateResults: List<SearchResult>
        val d1 = measureTimeMillis { approximateResults = index.findNeighbors(input, k) }


        val groundTruthResults : List<SearchResult>
        val d2 = measureTimeMillis { groundTruthResults = exactIndex.findNeighbors(input, k) }

        println("${"%.3f".format(d1 / 1000.0)} seconds, which is ${"%.3f".format(d1 / 1000.0 / 60)} minutes.")
        println("Most similar words found using HNSW index: \n")

        for (result in approximateResults) {
            println("${result.item().id} ${"%.2f".format(result.distance())}")
        }

        println("\n${"%.3f".format(d2 / 1000.0)} seconds, which is ${"%.3f".format(d2 / 1000.0 / 60)} minutes.")
        println("Most similar words found using exact index: \n")

        for (result in groundTruthResults) {
            println("${result.item().id} ${"%.2f".format(result.distance())}")
        }

        val correct = groundTruthResults
            .asSequence()
            .map { approximateResults.contains(it) }
            .sumOf { if (it) 1.toInt() else 0 }

        println("\nSpeedup: ${"%.4f".format(d2 / d1.toDouble() - 1)}")
        println("Accuracy: ${"%.4f".format(correct / groundTruthResults.size.toDouble())}\n")
    }
}

fun downloadFile(url: String, path: Path) {
    println("Downloading $url to $path. This may take a while.")

    URL(url).openStream().use { Files.copy(it, path) }
}

fun loadWordVectors(path: Path, size: Long): List<Item> {
    println("Loading words from $path")

    BufferedReader(InputStreamReader(GZIPInputStream(Files.newInputStream(path)), StandardCharsets.UTF_8)).use {
        return it.lines()
            .skip(1)
            .limit(size)
            .map { line ->
                val tokens = line.split(" ").toTypedArray()
                val word = tokens[0]
                val vector = FloatArray(tokens.size - 1)
                for (i in 1 until tokens.size - 1) {
                    vector[i] = tokens[i].toFloat()
                }
                Item(word, normalize(vector))
            }
            .collect(Collectors.toList())
    }
}
