package org.nativelab.phonolab.data

import org.json.JSONArray
import org.json.JSONObject
import org.nativelab.phonolab.PhonoLabStore
import java.io.File

data class ModelConfig(
    val path: String,
    var name: String = "",
    var repo: String = "",
    var quant: String = "",
    var ctx: Int = 2048,
    var threads: Int = 4,
    var temperature: Float = 0.7f,
    var topP: Float = 0.9f,
    var topK: Int = 40,
    var minP: Float = 0.0f,
    var repeatPenalty: Float = 1.1f,
    var maxTokens: Int = 384,
    var seed: Int = -1,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("path", path)
        put("name", name)
        put("repo", repo)
        put("quant", quant)
        put("ctx", ctx)
        put("threads", threads)
        put("temperature", temperature.toDouble())
        put("top_p", topP.toDouble())
        put("top_k", topK)
        put("min_p", minP.toDouble())
        put("repeat_penalty", repeatPenalty.toDouble())
        put("max_tokens", maxTokens)
        put("seed", seed)
    }

    companion object {
        fun fromJson(obj: JSONObject) = ModelConfig(
            path = obj.optString("path", ""),
            name = obj.optString("name", ""),
            repo = obj.optString("repo", ""),
            quant = obj.optString("quant", ""),
            ctx = obj.optInt("ctx", 2048),
            threads = obj.optInt("threads", 4),
            temperature = obj.optDouble("temperature", 0.7).toFloat(),
            topP = obj.optDouble("top_p", 0.9).toFloat(),
            topK = obj.optInt("top_k", 40),
            minP = obj.optDouble("min_p", 0.0).toFloat(),
            repeatPenalty = obj.optDouble("repeat_penalty", 1.1).toFloat(),
            maxTokens = obj.optInt("max_tokens", 384),
            seed = obj.optInt("seed", -1),
        )

        fun defaults(path: String, name: String) = ModelConfig(
            path = path,
            name = name,
            threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4),
        )
    }
}

class ModelManager(private val store: PhonoLabStore) {

    private val registryFile = File(store.configDir, "model_registry.json")
    private val models = mutableMapOf<String, ModelConfig>()

    init {
        registryFile.parentFile?.mkdirs()
        load()
    }

    private fun load() {
        if (!registryFile.exists()) return
        try {
            val obj = JSONObject(registryFile.readText())
            val arr = obj.optJSONArray("models") ?: return
            for (i in 0 until arr.length()) {
                val cfg = ModelConfig.fromJson(arr.getJSONObject(i))
                if (cfg.path.isNotEmpty()) models[cfg.path] = cfg
            }
        } catch (_: Exception) { }
    }

    fun save() {
        val obj = JSONObject().apply {
            put("version", 1)
            put("models", JSONArray().apply {
                models.values.forEach { put(it.toJson()) }
            })
        }
        registryFile.writeText(obj.toString(2))
    }

    fun all(): List<ModelConfig> = models.values.sortedBy { it.name.lowercase() }

    fun get(path: String): ModelConfig? = models[path]

    fun add(file: File, repo: String = "", quant: String = ""): ModelConfig {
        val key = file.absolutePath
        val existing = models[key]
        if (existing != null) {
            existing.repo = repo.ifEmpty { existing.repo }
            existing.quant = quant.ifEmpty { existing.quant }
            existing.name = existing.name.ifEmpty { file.name }
            save()
            return existing
        }
        val cfg = ModelConfig.defaults(key, file.name).apply {
            this.repo = repo
            this.quant = quant
        }
        models[key] = cfg
        save()
        return cfg
    }

    fun update(path: String, block: (ModelConfig) -> Unit) {
        val cfg = models[path] ?: return
        block(cfg)
        save()
    }

    fun remove(path: String) {
        models.remove(path)
        save()
        // Delete the actual file from disk
        try {
            val file = File(path)
            if (file.exists() && file.extension.equals("gguf", ignoreCase = true)) {
                file.delete()
                // Also delete parent dir if empty (e.g. models/<key>/)
                file.parentFile?.let { parent ->
                    if (parent.listFiles()?.isEmpty() == true) {
                        parent.delete()
                    }
                }
            }
        } catch (_: Exception) { }
    }

    fun discoverGgufFiles(): List<File> {
        if (!store.modelsDir.exists()) return emptyList()
        return store.modelsDir.walkTopDown()
            .filter { it.isFile && it.extension.equals("gguf", ignoreCase = true) }
            .sortedBy { it.name.lowercase() }
            .toList()
    }

    fun syncDiscovery() {
        // Deduplicate: use canonical path as key to avoid double entries
        for (file in discoverGgufFiles()) {
            val key = try { file.canonicalPath } catch (_: Exception) { file.absolutePath }
            if (!models.containsKey(key)) {
                // Also check if a non-canonical key already exists
                val existing = models[file.absolutePath]
                if (existing != null) {
                    // Re-key under canonical path
                    models.remove(file.absolutePath)
                    models[key] = existing
                } else {
                    models[key] = ModelConfig.defaults(key, file.name)
                }
            }
        }
        save()
    }
}
