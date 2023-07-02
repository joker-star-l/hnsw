import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.6.21"
    id("com.github.johnrengelman.shadow") version "7.1.1"
}

group = "me.lmx"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.6.4")

    runtimeOnly("org.jetbrains.kotlinx:kotlinx-coroutines-core-jvm:1.6.4")
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<KotlinCompile>() {
    kotlinOptions.jvmTarget = "1.8"
}

tasks.shadowJar {
    manifest {
        attributes(mapOf("Main-Class" to "me.lmx.knn.example.FastTextKt"))
    }
}