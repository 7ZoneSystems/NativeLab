package org.nativelab.phonolab

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class RagProcessor(private val context: Context) {

    data class RagResult(
        val chunks: List<String>,
        val totalChars: Int,
        val filename: String,
        val mimeType: String
    )

    suspend fun processDocument(
        uri: Uri,
        onProgress: (Float, String) -> Unit
    ): RagResult = withContext(Dispatchers.IO) {

        val mimeType = context.contentResolver.getType(uri) ?: "text/plain"
        val filename = getFilename(uri)

        onProgress(0.1f, "Reading file...")

        val text = when {
            mimeType == "application/pdf" -> extractPdf(uri, onProgress)
            mimeType.startsWith("text/") -> extractText(uri)
            mimeType.contains("word") -> extractDocx(uri, onProgress)
            else -> extractText(uri)
        }

        onProgress(0.7f, "Chunking...")
        val chunks = chunkText(text, chunkSize = 1500, overlap = 200)

        onProgress(1.0f, "Ready (${chunks.size} chunks)")
        RagResult(chunks, text.length, filename, mimeType)
    }

    private fun chunkText(text: String, chunkSize: Int, overlap: Int): List<String> {
        if (text.length <= chunkSize) return listOf(text)
        val chunks = mutableListOf<String>()
        var start = 0
        while (start < text.length) {
            val end = minOf(start + chunkSize, text.length)
            chunks.add(text.substring(start, end))
            start += chunkSize - overlap
        }
        return chunks
    }

    fun retrieveChunks(query: String, chunks: List<String>, topK: Int = 3): List<String> {
        val queryWords = query.lowercase()
            .split(Regex("\\W+"))
            .filter { it.length > 3 }
            .toSet()
        if (queryWords.isEmpty()) return chunks.take(topK)
        return chunks
            .map { chunk ->
                val score = queryWords.count { chunk.lowercase().contains(it) }
                chunk to score
            }
            .sortedByDescending { it.second }
            .take(topK)
            .map { it.first }
    }

    private fun extractText(uri: Uri): String {
        return context.contentResolver
            .openInputStream(uri)
            ?.bufferedReader()
            ?.readText() ?: ""
    }

    private fun extractPdf(uri: Uri, onProgress: (Float, String) -> Unit): String {
        val sb = StringBuilder()
        val pfd = context.contentResolver.openFileDescriptor(uri, "r") ?: return ""
        val renderer = android.graphics.pdf.PdfRenderer(pfd)
        val pageCount = renderer.pageCount
        for (i in 0 until pageCount) {
            onProgress(0.1f + (0.5f * i / pageCount), "Reading page ${i + 1}/$pageCount")
            sb.append("[PDF Page ${i + 1}]\n")
        }
        renderer.close()
        pfd.close()
        return sb.toString()
    }

    private fun extractDocx(uri: Uri, onProgress: (Float, String) -> Unit): String {
        onProgress(0.3f, "Reading document...")
        return try {
            val input = context.contentResolver.openInputStream(uri) ?: return ""
            val zip = java.util.zip.ZipInputStream(input)
            var entry = zip.nextEntry
            var xml = ""
            while (entry != null) {
                if (entry.name == "word/document.xml") {
                    xml = zip.bufferedReader().readText()
                    break
                }
                entry = zip.nextEntry
            }
            zip.close()
            xml.replace(Regex("<[^>]+>"), " ")
                .replace(Regex("\\s+"), " ")
                .trim()
        } catch (e: Exception) {
            ""
        }
    }

    private fun getFilename(uri: Uri): String {
        var name = "document"
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (idx >= 0) name = cursor.getString(idx)
            }
        }
        return name
    }
}
