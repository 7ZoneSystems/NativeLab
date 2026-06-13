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
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("role", role)
        put("content", content)
        put("timestamp", timestamp)
    }

    companion object {
        fun fromJson(obj: JSONObject) = ChatMessage(
            role = obj.optString("role", "user"),
            content = obj.optString("content", ""),
            timestamp = obj.optString("timestamp", ""),
        )
        fun now(): String = SimpleDateFormat("HH:mm", Locale.US).format(Date())
    }
}

data class ChatSession(
    val id: String,
    var title: String,
    val created: String,  // "2026-06-13"
    val messages: MutableList<ChatMessage> = mutableListOf(),
) {
    fun addMessage(role: String, content: String): ChatMessage {
        val msg = ChatMessage(role, content)
        messages.add(msg)
        // Auto-title from first user message
        if (title == "New Chat" && role == "user" && messages.count { it.role == "user" } == 1) {
            title = content.take(40).let { if (content.length > 40) "$it…" else it }
        }
        return msg
    }

    fun buildPrompt(maxChars: Int = 8000): String {
        val recent = mutableListOf<ChatMessage>()
        var used = 0
        for (m in messages.reversed()) {
            used += m.content.length
            if (used > maxChars) break
            recent.add(0, m)
        }
        return buildString {
            for (m in recent) {
                when (m.role) {
                    "user" -> append("User: ${m.content}\n")
                    "assistant" -> append("Assistant: ${m.content}\n")
                }
            }
            append("Assistant: ")
        }
    }

    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id)
        put("title", title)
        put("created", created)
        put("messages", JSONArray().apply {
            messages.forEach { put(it.toJson()) }
        })
    }

    companion object {
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
                    msgs.add(ChatMessage.fromJson(arr.getJSONObject(i)))
                }
            }
            return ChatSession(
                id = obj.optString("id", ""),
                title = obj.optString("title", "Chat"),
                created = obj.optString("created", ""),
                messages = msgs,
            )
        }
    }
}
