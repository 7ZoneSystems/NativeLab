package org.nativelab.phonolab

data class ModelFamily(
    val name: String,
    val family: String,
    val template: String,
    val systemPrefix: String = "",
    val systemSuffix: String = "",
    val userPrefix: String = "",
    val userSuffix: String = "",
    val assistantPrefix: String = "",
    val assistantSuffix: String = "",
    val stopTokens: List<String> = emptyList(),
    val bos: String = "<s>",
    val eos: String = "</s>",
)

data class VisionModelInfo(
    val isVision: Boolean = false,
    val label: String = "",
    val familyHint: String = "",
    val needsMmproj: Boolean = true,
)

data class ApiProvider(
    val name: String,
    val baseUrl: String,
    val format: String = "openai",
    val models: List<String> = emptyList(),
)

private val VLM_PATTERNS = listOf(
    listOf("llava", "bakllava", "moondream") to Triple("LLaVA VLM", "llava", true),
    listOf("minicpm-v", "minicpmv", "minicpm-o", "minicpmo") to Triple("MiniCPM-V", "minicpm-v", true),
    listOf("qwen2-vl", "qwen2.5-vl", "qwen-vl", "qwen2vl", "qwen25vl", "qwenvl") to Triple("Qwen-VL", "qwen-vl", true),
    listOf("internvl", "intern-vl") to Triple("InternVL", "internvl", true),
    listOf("pixtral") to Triple("Pixtral", "pixtral", true),
    listOf("paligemma") to Triple("PaliGemma", "paligemma", true),
    listOf("gemma-3", "gemma3") to Triple("Gemma 3 Vision", "gemma3", true),
    listOf("llama-vision", "llama3.2-vision", "llama-3.2-vision", "vision-instruct") to Triple("Llama Vision", "llama-vision", true),
    listOf("mllama", "multi-modal", "multimodal", "vlm") to Triple("Vision-Language Model", "vlm", true),
)

fun detectVisionModel(filename: String): VisionModelInfo {
    val name = filename.substringAfterLast("/").substringBeforeLast(".").lowercase()
    for ((keywords, info) in VLM_PATTERNS) {
        if (keywords.any { it in name }) {
            return VisionModelInfo(true, info.first, info.second, info.third)
        }
    }
    return VisionModelInfo()
}

fun detectMmprojForModel(modelPath: java.io.File): String {
    if (!modelPath.exists()) return ""
    val folder = modelPath.parentFile ?: return ""
    val stem = modelPath.nameWithoutExtension.lowercase()
    val candidates = mutableListOf<Pair<Int, java.io.File>>()
    for (f in folder.listFiles()?.filter { it.extension == "gguf" && it != modelPath } ?: emptyList()) {
        val n = f.name.lowercase()
        if (listOf("mmproj", "projector", "vision", "clip").any { it in n }) {
            var score = 10
            for (token in stem.split(Regex("[-_. ]+"))) {
                if (token.length > 2 && token in n) score += 1
            }
            candidates.add(score to f)
        }
    }
    if (candidates.isEmpty()) return ""
    candidates.sortWith(compareByDescending<Pair<Int, java.io.File>> { it.first }.thenBy { it.second.name.lowercase() })
    return candidates.first().second.absolutePath
}

fun detectModelFamily(filename: String): ModelFamily {
    val name = filename.substringAfterLast("/").substringBeforeLast(".").lowercase()
    for ((keywords, familyKey) in FAMILY_DETECTION_PATTERNS) {
        for (kw in keywords) {
            if (kw in name) {
                return FamilyTemplates.templates[familyKey] ?: FamilyTemplates.templates["default"]!!
            }
        }
    }
    return FamilyTemplates.templates["default"]!!
}

private val FAMILY_DETECTION_PATTERNS: List<Pair<List<String>, String>> = listOf(
    listOf("deepseek-r1") to "deepseek-r1",
    listOf("deepseek-coder", "deepseek_coder") to "deepseek-coder",
    listOf("deepseek") to "deepseek",
    listOf("mixtral") to "mixtral",
    listOf("mistral") to "mistral",
    listOf("llama-3", "llama3", "llama_3", "meta-llama-3") to "llama3",
    listOf("codellama", "code-llama", "code_llama") to "codellama",
    listOf("llama-2", "llama2", "llama_2") to "llama2",
    listOf("llama") to "llama2",
    listOf("phi-3", "phi3") to "phi3",
    listOf("phi") to "phi",
    listOf("qwen") to "qwen",
    listOf("gemma") to "gemma",
    listOf("yi-") to "yi",
    listOf("command-r", "command_r") to "command-r",
    listOf("orca", "openorca") to "orca",
    listOf("falcon") to "falcon",
    listOf("vicuna") to "vicuna",
    listOf("openchat") to "openchat",
    listOf("neural-chat", "neural_chat") to "neural-chat",
    listOf("starling") to "starling",
    listOf("zephyr") to "zephyr",
    listOf("solar") to "solar",
)

object FamilyTemplates {
    val templates: Map<String, ModelFamily> = mapOf(
        "deepseek" to ModelFamily(
            name = "DeepSeek", family = "deepseek", template = "deepseek",
            userPrefix = "User: ", userSuffix = "\n\nAssistant:",
            assistantPrefix = "", assistantSuffix = "\n\n",
            stopTokens = listOf("<|EOT|>", "\nUser:", "\n\nUser:"),
        ),

        "deepseek-coder" to ModelFamily(
            name = "DeepSeek-Coder", family = "deepseek-coder", template = "deepseek",
            userPrefix = "### Instruction:\n", userSuffix = "\n### Response:\n",
            assistantPrefix = "", assistantSuffix = "\n",
            stopTokens = listOf("<|EOT|>", "\n### Instruction:"),
        ),

        "deepseek-r1" to ModelFamily(
            name = "DeepSeek-R1", family = "deepseek-r1", template = "deepseek-r1",
            userPrefix = "<|User|>", userSuffix = "<|Assistant|><think>\n",
            assistantPrefix = "", assistantSuffix = "</think>\n",
            stopTokens = listOf("<|end_of_sentence|>", "<|EOT|>", "\n<|User|>"),
        ),

        "mistral" to ModelFamily(
            name = "Mistral Instruct", family = "mistral", template = "mistral",
            userPrefix = "[INST] ", userSuffix = " [/INST]",
            assistantPrefix = "", assistantSuffix = "</s>",
            stopTokens = listOf("</s>", "[INST]", "[/INST]", "### Human:", "### Assistant:",
                "[ORG_NAME]", "[NAME]", "[USER]", "\n[", "```\n\n["),
            bos = "<s>", eos = "</s>",
        ),

        "mixtral" to ModelFamily(
            name = "Mixtral MoE", family = "mixtral", template = "mistral",
            userPrefix = "[INST] ", userSuffix = " [/INST]",
            assistantPrefix = "", assistantSuffix = "</s>",
            stopTokens = listOf("</s>", "[INST]"),
            bos = "<s>", eos = "</s>",
        ),

        "llama2" to ModelFamily(
            name = "LLaMA-2 Chat", family = "llama2", template = "llama2",
            systemPrefix = "[INST] <<SYS>>\n", systemSuffix = "\n<</SYS>>\n\n",
            userPrefix = "", userSuffix = " [/INST]",
            assistantPrefix = "", assistantSuffix = "</s><s>[INST] ",
            stopTokens = listOf("</s>", "[INST]"),
            bos = "<s>", eos = "</s>",
        ),

        "llama3" to ModelFamily(
            name = "LLaMA-3 Instruct", family = "llama3", template = "llama3",
            systemPrefix = "<|start_header_id|>system<|end_header_id|\n\n",
            systemSuffix = "<|eot_id|>",
            userPrefix = "<|start_header_id|>user<|end_header_id|\n\n",
            userSuffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|\n\n",
            assistantPrefix = "", assistantSuffix = "<|eot_id|>",
            stopTokens = listOf("<|eot_id|>", "<|end_of_text|>"),
            bos = "<|begin_of_text|>", eos = "<|end_of_text|>",
        ),

        "phi" to ModelFamily(
            name = "Phi Instruct", family = "phi", template = "phi",
            systemPrefix = "<|system|>\n", systemSuffix = "<|end|>\n",
            userPrefix = "<|user|>\n", userSuffix = "<|end|>\n<|assistant|>\n",
            assistantPrefix = "", assistantSuffix = "<|end|>\n",
            stopTokens = listOf("<|end|>", "<|user|>"),
            bos = "", eos = "</s>",
        ),

        "phi3" to ModelFamily(
            name = "Phi-3 Instruct", family = "phi3", template = "phi",
            systemPrefix = "<|system|>\n", systemSuffix = "<|end|>\n",
            userPrefix = "<|user|>\n", userSuffix = "<|end|>\n<|assistant|>\n",
            assistantPrefix = "", assistantSuffix = "<|end|>\n",
            stopTokens = listOf("<|end|>", "<|user|>", "</s>"),
            bos = "", eos = "</s>",
        ),

        "qwen" to ModelFamily(
            name = "Qwen ChatML", family = "qwen", template = "chatml",
            systemPrefix = "<im_start>>system\n", systemSuffix = "<im_end>>\n",
            userPrefix = "<im_start>>user\n", userSuffix = "<im_end>>\n<im_start>>assistant\n",
            assistantPrefix = "", assistantSuffix = "<im_end>>\n",
            stopTokens = listOf("<im_end>>", "<im_end>>"),
            bos = "", eos = "<im_end>>",
        ),

        "gemma" to ModelFamily(
            name = "Gemma Instruct", family = "gemma", template = "gemma",
            userPrefix = "<start_of_turn>user\n", userSuffix = "<end_of_turn>\n<start_of_turn>model\n",
            assistantPrefix = "", assistantSuffix = "<end_of_turn>\n",
            stopTokens = listOf("<end_of_turn>", "<eos>"),
            bos = "<bos>", eos = "<eos>",
        ),

        "codellama" to ModelFamily(
            name = "CodeLlama Instruct", family = "codellama", template = "llama2",
            systemPrefix = "[INST] <<SYS>>\n", systemSuffix = "\n<</SYS>>\n\n",
            userPrefix = "", userSuffix = " [/INST]",
            assistantPrefix = "", assistantSuffix = "</s><s>",
            stopTokens = listOf("</s>", "[INST]", "Source:"),
            bos = "<s>", eos = "</s>",
        ),

        "falcon" to ModelFamily(
            name = "Falcon", family = "falcon", template = "falcon",
            userPrefix = "User: ", userSuffix = "\nAssistant:",
            assistantPrefix = "", assistantSuffix = "\n",
            stopTokens = listOf("User:", "\nUser:", "endoftext"),
            bos = "", eos = "endoftext",
        ),

        "vicuna" to ModelFamily(
            name = "Vicuna", family = "vicuna", template = "vicuna",
            systemPrefix = "", systemSuffix = "\n\n",
            userPrefix = "USER: ", userSuffix = "\nASSISTANT:",
            assistantPrefix = "", assistantSuffix = "</s>",
            stopTokens = listOf("</s>", "USER:"),
            bos = "<s>", eos = "</s>",
        ),

        "openchat" to ModelFamily(
            name = "OpenChat", family = "openchat", template = "openchat",
            userPrefix = "GPT4 Correct User: ", userSuffix = "<im_end>>GPT4 Correct Assistant:",
            assistantPrefix = "", assistantSuffix = "<im_end>>",
            stopTokens = listOf("<im_end>>"),
            bos = "<s>", eos = "</s>",
        ),

        "neural-chat" to ModelFamily(
            name = "Neural Chat", family = "neural-chat", template = "neural-chat",
            systemPrefix = "### System:\n", systemSuffix = "\n",
            userPrefix = "### User:\n", userSuffix = "\n### Assistant:\n",
            assistantPrefix = "", assistantSuffix = "\n",
            stopTokens = listOf("### User:", "endoftext"),
            bos = "", eos = "endoftext",
        ),

        "starling" to ModelFamily(
            name = "Starling", family = "starling", template = "openchat",
            userPrefix = "GPT4 Correct User: ", userSuffix = "<im_end>>GPT4 Correct Assistant:",
            assistantPrefix = "", assistantSuffix = "<im_end>>",
            stopTokens = listOf("<im_end>>"),
            bos = "<s>", eos = "</s>",
        ),

        "yi" to ModelFamily(
            name = "Yi Chat", family = "yi", template = "chatml",
            systemPrefix = "<im_start>>system\n", systemSuffix = "<im_end>>\n",
            userPrefix = "<im_start>>user\n", userSuffix = "<im_end>>\n<im_start>>assistant\n",
            assistantPrefix = "", assistantSuffix = "<im_end>>\n",
            stopTokens = listOf("<im_end>>", "endoftext"),
            bos = "", eos = "endoftext",
        ),

        "command-r" to ModelFamily(
            name = "Command-R", family = "command-r", template = "command-r",
            systemPrefix = "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
            systemSuffix = "<|END_OF_TURN_TOKEN|>",
            userPrefix = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
            userSuffix = "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
            assistantPrefix = "", assistantSuffix = "<|END_OF_TURN_TOKEN|>",
            stopTokens = listOf("<|END_OF_TURN_TOKEN|>"),
            bos = "", eos = "<|END_OF_TURN_TOKEN|>",
        ),

        "orca" to ModelFamily(
            name = "Orca / ChatML", family = "orca", template = "chatml",
            systemPrefix = "<im_start>>system\n", systemSuffix = "<im_end>>\n",
            userPrefix = "<im_start>>user\n", userSuffix = "<im_end>>\n<im_start>>assistant\n",
            assistantPrefix = "", assistantSuffix = "<im_end>>\n",
            stopTokens = listOf("<im_end>>"),
            bos = "", eos = "endoftext",
        ),

        "zephyr" to ModelFamily(
            name = "Zephyr", family = "zephyr", template = "zephyr",
            systemPrefix = "<|system|>\n", systemSuffix = "</s>\n",
            userPrefix = "<|user|>\n", userSuffix = "</s>\n<|assistant|>\n",
            assistantPrefix = "", assistantSuffix = "</s>\n",
            stopTokens = listOf("</s>", "<|user|>"),
            bos = "<s>", eos = "</s>",
        ),

        "solar" to ModelFamily(
            name = "Solar Instruct", family = "solar", template = "mistral",
            userPrefix = "### User:\n", userSuffix = "\n\n### Assistant:\n",
            assistantPrefix = "", assistantSuffix = "\n",
            stopTokens = listOf("### User:", "</s>"),
            bos = "<s>", eos = "</s>",
        ),

        "default" to ModelFamily(
            name = "Generic Instruct", family = "default", template = "default",
            userPrefix = "### Instruction:\n", userSuffix = "\n### Response:\n",
            assistantPrefix = "", assistantSuffix = "\n",
            stopTokens = listOf("### Instruction:", "### Human:", "</s>"),
            bos = "", eos = "",
        ),
    )
}

