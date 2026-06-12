package org.nativelab.phonolab

import android.content.Context
import android.content.SharedPreferences
import java.io.File

class PhonoLabStore(context: Context) {
    val root: File = File(context.filesDir, "PhonoLab")
    val runtimeDir: File = File(root, "runtime")
    val runtimeBinDir: File = File(runtimeDir, "bin")
    val sourceDir: File = File(runtimeDir, "src/llama.cpp")
    val modelsDir: File = File(root, "models")
    val downloadsDir: File = File(root, "downloads")
    val stateDir: File = File(root, "state")

    private val prefs: SharedPreferences =
        context.getSharedPreferences("phonolab_prefs", Context.MODE_PRIVATE)

    init {
        listOf(root, runtimeDir, runtimeBinDir, modelsDir, downloadsDir, stateDir).forEach {
            it.mkdirs()
        }
    }

    fun isFirstLaunch(): Boolean = !prefs.getBoolean("has_launched", false)

    fun markLaunched() {
        prefs.edit().putBoolean("has_launched", true).apply()
    }

    fun isBinaryInstalled(): Boolean = prefs.getBoolean("binary_installed", false)

    fun markBinaryInstalled() {
        prefs.edit().putBoolean("binary_installed", true).apply()
    }

    fun safeChild(parent: File, relativeName: String): File {
        val cleaned = relativeName.replace("\\", "/").trim()
        require(cleaned.isNotEmpty()) { "Empty file name" }
        require(!cleaned.startsWith("/")) { "Unsafe absolute path: $relativeName" }
        val parts = cleaned.split("/").filter { it.isNotBlank() }
        require(parts.none { it == "." || it == ".." }) { "Unsafe path segment: $relativeName" }
        val base = parent.canonicalFile
        val out = parts.fold(base) { current, part -> File(current, part) }.canonicalFile
        require(out.path == base.path || out.path.startsWith(base.path + File.separator)) {
            "Path escapes PhonoLab storage: $relativeName"
        }
        return out
    }

    fun modelFiles(): List<File> {
        if (!modelsDir.exists()) return emptyList()
        return modelsDir.walkTopDown()
            .filter { it.isFile && it.extension.equals("gguf", ignoreCase = true) }
            .sortedBy { it.name.lowercase() }
            .toList()
    }
}
