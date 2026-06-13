package org.nativelab.phonolab.data

import org.json.JSONObject
import org.nativelab.phonolab.PhonoLabStore
import java.io.File

class SessionManager(private val store: PhonoLabStore) {

    private val sessionsDir = File(store.root, "sessions")

    init {
        sessionsDir.mkdirs()
    }

    fun loadAll(): List<ChatSession> {
        return sessionsDir.listFiles()
            ?.filter { it.extension == "json" }
            ?.mapNotNull { file ->
                try {
                    ChatSession.fromJson(JSONObject(file.readText()))
                } catch (_: Exception) {
                    null
                }
            }
            ?.sortedByDescending { it.id }
            ?: emptyList()
    }

    fun save(session: ChatSession) {
        val file = File(sessionsDir, "${session.id}.json")
        file.writeText(session.toJson().toString(2))
    }

    fun delete(sessionId: String) {
        val file = File(sessionsDir, "$sessionId.json")
        if (file.exists()) file.delete()
    }

    fun rename(sessionId: String, newTitle: String) {
        val session = loadAll().find { it.id == sessionId } ?: return
        session.title = newTitle
        save(session)
    }

    fun exportMarkdown(session: ChatSession): String {
        return buildString {
            appendLine("# ${session.title}")
            appendLine()
            appendLine("*${session.created}*")
            appendLine()
            appendLine("---")
            appendLine()
            for (m in session.messages) {
                val label = if (m.role == "user") "**You**" else "**Assistant**"
                appendLine("$label · ${m.timestamp}")
                appendLine()
                appendLine(m.content)
                appendLine()
                appendLine("---")
                appendLine()
            }
        }
    }
}
