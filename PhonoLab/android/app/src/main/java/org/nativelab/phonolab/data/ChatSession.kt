package org.nativelab.phonolab.data

import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

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

data class ChatSession(
    val id: String,
    var title: String,
    val created: String,  // "2026-06-13"
    val messages: MutableList<ChatMessage> = mutableListOf(),
    val logs: MutableList<String> = mutableListOf(),
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
        maxChars: Int = 8000,
    ): JSONArray {
        val recent = mutableListOf<ChatMessage>()
        var used = systemPrompt.length
        for (m in messages.reversed()) {
            used += m.content.length
            if (used > maxChars) break
            recent.add(0, m)
        }
        return JSONArray().apply {
            put(JSONObject().apply {
                put("role", "system")
                put("content", systemPrompt)
            })
            recent.forEach { m ->
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
        maxChars: Int = 8000,
    ): JSONObject = JSONObject().apply {
        put("model", "phonolab-active")
        put("messages", buildMessages(systemPrompt, maxChars))
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
            )
        }
    }
}
