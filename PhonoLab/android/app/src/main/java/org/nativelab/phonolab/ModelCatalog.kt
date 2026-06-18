package org.nativelab.phonolab

data class ModelCandidate(
    val key: String,
    val label: String,
    val repo: String,
    val quantPreferences: List<String>,
    val minRamMb: Int,
    val ctx: Int = 2048,
    val tier: String = "balanced",
)

data class RemoteFile(
    val name: String,
    val size: Long,
)

object ModelCatalog {
    val items = listOf(
        ModelCandidate(
            "smollm2-360m",
            "SmolLM2 360M Instruct",
            "bartowski/SmolLM2-360M-Instruct-GGUF",
            listOf("Q4_K_M", "Q4_0", "Q5_K_M", "Q3_K_M"),
            2048,
            ctx = 2048,
            tier = "tiny",
        ),
        ModelCandidate(
            "qwen25-05b",
            "Qwen2.5 0.5B Instruct",
            "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
            listOf("Q4_K_M", "Q4_0", "Q5_K_M", "Q3_K_M"),
            3072,
            ctx = 2048,
            tier = "minimal",
        ),
        ModelCandidate(
            "llama32-1b",
            "Llama 3.2 1B Instruct",
            "bartowski/Llama-3.2-1B-Instruct-GGUF",
            listOf("Q4_K_M", "Q4_0", "Q5_K_M", "Q3_K_M"),
            4096,
            ctx = 2048,
            tier = "low",
        ),
        ModelCandidate(
            "qwen25-15b",
            "Qwen2.5 1.5B Instruct",
            "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
            listOf("Q4_K_M", "Q4_0", "Q5_K_M", "Q3_K_M"),
            6144,
            ctx = 4096,
            tier = "balanced",
        ),
        ModelCandidate(
            "tinyllama-11b",
            "TinyLlama 1.1B Chat",
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            listOf("Q4_K_M", "Q4_0", "Q5_K_M", "Q3_K_M"),
            4096,
            ctx = 2048,
            tier = "fallback",
        ),
    )

    fun chooseForDevice(totalRamMb: Int): ModelCandidate {
        val eligible = items.filter { totalRamMb >= it.minRamMb }
        if (eligible.isEmpty()) return items.first()
        if (totalRamMb < 6144) return eligible[minOf(eligible.size - 1, 2)]
        return eligible.last()
    }
}
