package org.nativelab.phonolab.data

import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.ceil
import kotlin.math.min

data class ChatMessage(
    val role: String,       // "user" or "assistant"
    val content: String,
    val timestamp: String = now(),
    val imageBase64: String? = null,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("role", role)
        put("content", content)
        put("timestamp", timestamp)
        if (imageBase64 != null) put("imageBase64", imageBase64)
    }

    companion object {
        fun fromJson(obj: JSONObject) = ChatMessage(
            role = obj.optString("role", "user"),
            content = obj.optString("content", ""),
            timestamp = obj.optString("timestamp", ""),
            imageBase64 = obj.optString("imageBase64", null),
        )
        fun now(): String = SimpleDateFormat("HH:mm", Locale.US).format(Date())
    }
}

/**
 * The single source of truth for conversation context sent to a model.
 *
 * Sessions keep their complete visible transcript. When it exceeds a model's
 * configured context window, the newest turns remain verbatim and a bounded,
 * durable memory represents the older part of the conversation.
 */
object ContextWindowManager {
    const val DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

    data class Usage(
        val inputTokens: Int,
        val contextLimit: Int,
        val reservedOutputTokens: Int,
        val compactedMessages: Int,
    )

    data class PreparedContext(
        val olderMemory: String?,
        val recentMessages: List<ChatMessage>,
        val usage: Usage,
    )

    /** A conservative, dependency-free token estimate suitable for budgeting. */
    fun estimateTokens(text: String): Int =
        if (text.isBlank()) 0 else ceil(text.length / 4.0).toInt().coerceAtLeast(1)

    fun usage(
        session: ChatSession,
        contextLimit: Int,
        maxOutputTokens: Int,
        systemPrompt: String = DEFAULT_SYSTEM_PROMPT,
    ): Usage = prepare(session, contextLimit, maxOutputTokens, systemPrompt, persistMemory = false).usage

    /**
     * Select the context for the next request and, when needed, persist its
     * rolling memory on the session so a reopened chat retains prior context.
     */
    fun prepare(
        session: ChatSession,
        contextLimit: Int,
        maxOutputTokens: Int,
        systemPrompt: String = DEFAULT_SYSTEM_PROMPT,
        persistMemory: Boolean = true,
    ): PreparedContext {
        val limit = contextLimit.coerceAtLeast(512)
        val reserve = maxOutputTokens.coerceIn(64, (limit / 2).coerceAtLeast(64))
        val inputBudget = (limit - reserve).coerceAtLeast(128)
        val transcript = session.messages.filter { it.content.isNotBlank() }
        val systemTokens = estimateTokens(systemPrompt) + MESSAGE_OVERHEAD

        // Reserve memory first. If no compaction is needed we reclaim it.
        val memoryTarget = min(MAX_MEMORY_TOKENS, (inputBudget / 3).coerceAtLeast(MIN_MEMORY_TOKENS))
        var recent = selectRecent(transcript, inputBudget - systemTokens - memoryTarget)
        var omitted = transcript.take((transcript.size - recent.size).coerceAtLeast(0))
        var memory: String? = null

        if (omitted.isNotEmpty()) {
            // The initial reservation is pessimistic. Retry with the actual
            // memory size before deciding that older messages must be compacted.
            val candidateMemory = compact(omitted, memoryTarget)
            recent = selectRecent(
                transcript,
                inputBudget - systemTokens - estimateTokens(candidateMemory) - MESSAGE_OVERHEAD,
            )
            omitted = transcript.take((transcript.size - recent.size).coerceAtLeast(0))
            if (omitted.isNotEmpty()) {
                memory = compact(omitted, memoryTarget)
                if (persistMemory) {
                    session.contextSummary = memory
                    session.compactedMessageCount = omitted.size
                }
            } else if (persistMemory) {
                session.contextSummary = ""
                session.compactedMessageCount = 0
            }
        } else if (persistMemory && session.contextSummary.isNotEmpty()) {
            session.contextSummary = ""
            session.compactedMessageCount = 0
        }

        val used = systemTokens + recent.sumOf { estimateMessage(it) } +
            (memory?.let { estimateTokens(it) + MESSAGE_OVERHEAD } ?: 0)
        return PreparedContext(
            olderMemory = memory,
            recentMessages = recent,
            usage = Usage(
                inputTokens = used.coerceAtMost(inputBudget),
                contextLimit = limit,
                reservedOutputTokens = reserve,
                compactedMessages = omitted.size,
            ),
        )
    }

    private fun selectRecent(messages: List<ChatMessage>, availableTokens: Int): List<ChatMessage> {
        if (messages.isEmpty() || availableTokens <= 0) return emptyList()
        var remaining = availableTokens
        val selected = ArrayList<ChatMessage>()
        for (message in messages.asReversed()) {
            val cost = estimateMessage(message)
            if (cost <= remaining) {
                selected.add(message)
                remaining -= cost
            } else if (selected.isEmpty()) {
                // Never discard the active user turn. Keep its end too, where
                // a request's actual question commonly appears.
                val chars = ((remaining - MESSAGE_OVERHEAD).coerceAtLeast(16) * 4)
                selected.add(message.copy(content = shorten(message.content, chars)))
                remaining = 0
            } else {
                break
            }
        }
        return selected.asReversed()
    }

    private fun estimateMessage(message: ChatMessage): Int =
        estimateTokens(message.content) + MESSAGE_OVERHEAD + if (message.imageBase64 != null) IMAGE_ALLOWANCE else 0

    private fun compact(messages: List<ChatMessage>, maxTokens: Int): String {
        val maxChars = (maxTokens * 4).coerceAtLeast(160)
        val lines = messages.map { message ->
            val speaker = if (message.role == "user") "User" else "Assistant"
            "$speaker: ${shorten(message.content.replace(Regex("\\s+"), " ").trim(), 360)}"
        }
        return shorten(lines.joinToString("\n"), maxChars)
    }

    private fun shorten(value: String, maxChars: Int): String {
        if (value.length <= maxChars) return value
        val front = (maxChars * 0.65).toInt().coerceAtLeast(1)
        val back = (maxChars - front - 3).coerceAtLeast(1)
        return value.take(front).trimEnd() + " … " + value.takeLast(back).trimStart()
    }

    private const val MESSAGE_OVERHEAD = 4
    private const val IMAGE_ALLOWANCE = 256
    private const val MIN_MEMORY_TOKENS = 96
    private const val MAX_MEMORY_TOKENS = 384
}

data class ChatSession(
    val id: String,
    var title: String,
    val created: String,  // "2026-06-13"
    val messages: MutableList<ChatMessage> = mutableListOf(),
    val logs: MutableList<String> = mutableListOf(),
    var contextSummary: String = "",
    var compactedMessageCount: Int = 0,
) {
    fun addMessage(role: String, content: String, imageBase64: String? = null): ChatMessage {
        val msg = ChatMessage(role, content, imageBase64 = imageBase64)
        messages.add(msg)
        // Auto-title from first user message
        if (title == "New Chat" && role == "user" && messages.count { it.role == "user" } == 1) {
            title = content.take(40).let { if (content.length > 40) "$it…" else it }
        }
        return msg
    }

    fun addLog(entry: String) {
        logs.add(entry)
        if (logs.size > MAX_LOGS) {
            val excess = logs.size - MAX_LOGS
            repeat(excess) { logs.removeAt(0) }
        }
    }

    /**
     * Build OpenAI-compatible messages array for /v1/chat/completions.
     * The server applies the model's chat template (e.g. SmolLM2's ChatML).
     */
    fun buildMessages(
        systemPrompt: String = "You are a helpful AI assistant.",
        contextLimit: Int = 2048,
        maxOutputTokens: Int = 384,
    ): JSONArray {
        val prepared = ContextWindowManager.prepare(this, contextLimit, maxOutputTokens, systemPrompt)
        return JSONArray().apply {
            put(JSONObject().apply {
                put("role", "system")
                put(
                    "content",
                    if (prepared.olderMemory.isNullOrBlank()) systemPrompt
                    else "$systemPrompt\n\nMemory from earlier in this conversation:\n${prepared.olderMemory}",
                )
            })
            prepared.recentMessages.forEach { m ->
                put(JSONObject().apply {
                    put("role", m.role)
                    if (m.imageBase64 != null) {
                        put("content", JSONArray().apply {
                            put(JSONObject().apply {
                                put("type", "text")
                                put("text", m.content)
                            })
                            put(JSONObject().apply {
                                put("type", "image_url")
                                put("image_url", JSONObject().apply {
                                    put("url", "data:image/jpeg;base64,${m.imageBase64}")
                                })
                            })
                        })
                    } else {
                        put("content", m.content)
                    }
                })
            }
        }
    }

    /**
     * Build the full request body for /v1/chat/completions.
     */
    fun buildRequestBody(
        systemPrompt: String = "You are a helpful AI assistant.",
        temperature: Float = 0.7f,
        maxTokens: Int = 384,
        contextLimit: Int = 2048,
    ): JSONObject = JSONObject().apply {
        put("model", "phonolab-active")
        put("messages", buildMessages(systemPrompt, contextLimit, maxTokens))
        put("temperature", temperature.toDouble())
        put("max_tokens", maxTokens)
        put("stream", true)
    }

    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id)
        put("title", title)
        put("created", created)
        put("messages", JSONArray().apply {
            messages.forEach { put(it.toJson()) }
        })
        put("logs", JSONArray().apply {
            logs.forEach { put(it) }
        })
        if (contextSummary.isNotEmpty()) put("contextSummary", contextSummary)
        put("compactedMessageCount", compactedMessageCount)
    }

    companion object {
        private const val MAX_LOGS = 500

        fun new(title: String = "New Chat"): ChatSession {
            val now = Date()
            return ChatSession(
                id = SimpleDateFormat("yyyy-MM-dd_HHmmss", Locale.US).format(now),
                title = title,
                created = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(now),
            )
        }

        fun fromJson(obj: JSONObject): ChatSession {
            val msgs = mutableListOf<ChatMessage>()
            val arr = obj.optJSONArray("messages")
            if (arr != null) {
                for (i in 0 until arr.length()) {
                    val msgObj = arr.optJSONObject(i) ?: continue
                    msgs.add(ChatMessage.fromJson(msgObj))
                }
            }
            val sessionLogs = mutableListOf<String>()
            val logArr = obj.optJSONArray("logs")
            if (logArr != null) {
                for (i in 0 until logArr.length()) {
                    sessionLogs.add(logArr.optString(i, ""))
                }
            }
            return ChatSession(
                id = obj.optString("id", ""),
                title = obj.optString("title", "Chat"),
                created = obj.optString("created", ""),
                messages = msgs,
                logs = sessionLogs,
                contextSummary = obj.optString("contextSummary", ""),
                compactedMessageCount = obj.optInt("compactedMessageCount", 0),
            )
        }
    }
}
