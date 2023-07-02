# HNSW 索引

HNSW 索引的 kotlin 语言实现。

## 目录结构

```
|-- src/
    |-- main/
        |-- kotlin/me/lmx/knn/
            |-- example/
                |-- fastText.kt            // 使用fasttext数据集的样例
            |-- hnsw/
                |-- HNSWIndex.kt           // HNSW 索引类
            |-- util/
                |-- ReentrantMutex.kt      // 协程可重入锁
                |-- vector.kt              // 向量计算通用方法
            |-- Index.kt                   // 索引接口
            |-- Item.kt                    // 索引项类
            |-- SearchResult.kt            // 搜索结果类
    |-- test
        |-- kotlin/me/lmx/knn/hnsw/
            |-- HNSWIndexTest.kt           // HNSW 功能测试
```

## 参考

1 Malkov Y A, Yashunin D A. 2018. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE transactions on pattern analysis and machine intelligence.

2 2023. hnswlib. https://github.com/jelmerk/hnswlib

