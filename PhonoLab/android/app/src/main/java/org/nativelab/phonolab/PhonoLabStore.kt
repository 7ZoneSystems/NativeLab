package org.nativelab.phonolab

import android.content.Context
import android.content.SharedPreferences
import java.io.File

/**
 * App storage backed by StorageManager.
 * All paths come from the user-selected folder (or app-private fallback).
 */
class PhonoLabStore(context: Context, storageManager: StorageManager? = null) {

    private val sm = storageManager ?: StorageManager(context)
    private val prefs: SharedPreferences =
        context.getSharedPreferences("phonolab_prefs", Context.MODE_PRIVATE)

    /** Base PhonoLab directory. */
    val root: File = sm.getBaseDir() ?: File(context.filesDir, "PhonoLab").also { it.mkdirs() }

    /** llama-server binary + .so files. */
    val runtimeDir: File = sm.getLlamaServerDir()

    /** Legacy bin dir (same as runtimeDir now). */
    val runtimeBinDir: File = runtimeDir

    /** llama.cpp source (if pulled). */
    val sourceDir: File = File(runtimeDir, "src/llama.cpp")

    /** Downloaded GGUF models. */
    val modelsDir: File = sm.getModelsDir()

    /** Temp downloads. */
    val downloadsDir: File = sm.getDownloadsDir()

    /** State files. */
    val stateDir: File = File(root, "state").also { it.mkdirs() }

    /** Chat sessions. */
    val sessionsDir: File = sm.getSessionsDir()

    /** Config files. */
    val configDir: File = sm.getConfigDir()

    init {
        listOf(root, runtimeDir, modelsDir, downloadsDir, stateDir, sessionsDir, configDir).forEach {
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
