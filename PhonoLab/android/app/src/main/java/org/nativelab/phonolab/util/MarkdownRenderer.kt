package org.nativelab.phonolab.util

import android.content.Context
import androidx.core.content.ContextCompat
import org.json.JSONObject
import org.nativelab.phonolab.R

/**
 * Small, self-contained Markdown renderer for chat WebViews.
 *
 * Keeping this renderer in the Android project (rather than handing raw model
 * text to a TextView) gives streamed and restored messages the same treatment:
 * headings, lists, quotes, tables, inline code and fenced code blocks.
 */
object MarkdownRenderer {
    private val fence = Regex("^\\s*```\\s*([A-Za-z0-9_+.-]*)\\s*$")
    private val tableDivider = Regex("^\\s*\\|?\\s*:?-{3,}:?\\s*(?:\\|\\s*:?-{3,}:?\\s*)+\\|?\\s*$")
    private val unordered = Regex("^\\s*[-*+]\\s+(.+)$")
    private val ordered = Regex("^\\s*\\d+[.)]\\s+(.+)$")

    fun document(context: Context, markdown: String): String {
        val colors = Colors(
            text = color(context, R.color.ph_txt),
            subdued = color(context, R.color.ph_txt2),
            accent = color(context, R.color.ph_accent),
            surface = color(context, R.color.ph_surface),
            codeSurface = color(context, R.color.ph_bg2),
            border = color(context, R.color.ph_bdr),
        )
        return page(render(markdown, colors), colors)
    }

    /** JavaScript invocation used to update an already-loaded streamed bubble. */
    fun updateScript(context: Context, markdown: String): String {
        val colors = Colors(
            text = color(context, R.color.ph_txt),
            subdued = color(context, R.color.ph_txt2),
            accent = color(context, R.color.ph_accent),
            surface = color(context, R.color.ph_surface),
            codeSurface = color(context, R.color.ph_bg2),
            border = color(context, R.color.ph_bdr),
        )
        return "window.PhonolabChat.render(${JSONObject.quote(render(markdown, colors))});"
    }

    private fun render(markdown: String, c: Colors): String {
        val lines = markdown.replace("\r\n", "\n").replace('\r', '\n').split('\n')
        val out = StringBuilder()
        var i = 0

        while (i < lines.size) {
            val line = lines[i]
            val openingFence = fence.matchEntire(line)
            if (openingFence != null) {
                val language = openingFence.groupValues[1].ifBlank { "code" }
                val body = mutableListOf<String>()
                i++
                while (i < lines.size && fence.matchEntire(lines[i]) == null) {
                    body += lines[i]
                    i++
                }
                if (i < lines.size) i++ // consume closing fence
                out.append(codeBlock(language, body.joinToString("\n"), c))
                continue
            }

            if (line.isBlank()) {
                i++
                continue
            }

            // GitHub-style tables: header row followed by a divider row.
            if (i + 1 < lines.size && line.contains('|') && tableDivider.matches(lines[i + 1])) {
                val header = tableCells(line)
                val alignments = tableAlignments(lines[i + 1], header.size)
                i += 2
                val rows = mutableListOf<List<String>>()
                while (i < lines.size && lines[i].isNotBlank() && lines[i].contains('|')) {
                    rows += tableCells(lines[i])
                    i++
                }
                out.append(table(header, alignments, rows, c))
                continue
            }

            when {
                line.startsWith("### ") -> {
                    out.append("<h3>${inline(line.removePrefix("### "), c)}</h3>")
                    i++
                }
                line.startsWith("## ") -> {
                    out.append("<h2>${inline(line.removePrefix("## "), c)}</h2>")
                    i++
                }
                line.startsWith("# ") -> {
                    out.append("<h1>${inline(line.removePrefix("# "), c)}</h1>")
                    i++
                }
                line.matches(Regex("^\\s{0,3}([-*_])(?:\\s*\\1){2,}\\s*$")) -> {
                    out.append("<hr>")
                    i++
                }
                line.trimStart().startsWith(">") -> {
                    val quote = mutableListOf<String>()
                    while (i < lines.size && lines[i].trimStart().startsWith(">")) {
                        quote += lines[i].trimStart().removePrefix(">").trimStart()
                        i++
                    }
                    out.append("<blockquote>${quote.joinToString("<br>") { inline(it, c) }}</blockquote>")
                }
                unordered.matches(line) -> {
                    val items = mutableListOf<String>()
                    while (i < lines.size) {
                        val match = unordered.matchEntire(lines[i]) ?: break
                        items += "<li>${inline(match.groupValues[1], c)}</li>"
                        i++
                    }
                    out.append("<ul>${items.joinToString("")}</ul>")
                }
                ordered.matches(line) -> {
                    val items = mutableListOf<String>()
                    while (i < lines.size) {
                        val match = ordered.matchEntire(lines[i]) ?: break
                        items += "<li>${inline(match.groupValues[1], c)}</li>"
                        i++
                    }
                    out.append("<ol>${items.joinToString("")}</ol>")
                }
                else -> {
                    val paragraph = mutableListOf<String>()
                    while (i < lines.size && lines[i].isNotBlank()) {
                        if (paragraph.isNotEmpty() && (fence.matchEntire(lines[i]) != null ||
                            (i + 1 < lines.size && lines[i].contains('|') && tableDivider.matches(lines[i + 1])) ||
                            unordered.matches(lines[i]) || ordered.matches(lines[i]) ||
                            lines[i].trimStart().startsWith(">") || lines[i].startsWith("# ") ||
                            lines[i].startsWith("## ") || lines[i].startsWith("### "))) break
                        paragraph += lines[i]
                        i++
                    }
                    out.append("<p>${paragraph.joinToString("<br>") { inline(it, c) }}</p>")
                }
            }
        }
        return out.toString()
    }

    private fun codeBlock(language: String, body: String, c: Colors): String {
        val label = escape(language.uppercase())
        return """
            <section class="code-card">
              <div class="code-head"><span class="language">$label</span></div>
              <pre><code class="language-${escape(language.lowercase())}">${escape(body)}</code></pre>
            </section>
        """.trimIndent()
    }

    private fun table(header: List<String>, alignments: List<String>, rows: List<List<String>>, c: Colors): String {
        fun row(cells: List<String>, tag: String): String = buildString {
            append("<tr>")
            for (index in header.indices) {
                val value = cells.getOrElse(index) { "" }
                append("<$tag style=\"text-align:${alignments[index]}\">${inline(value, c)}</$tag>")
            }
            append("</tr>")
        }
        return buildString {
            append("<div class=\"table-wrap\"><table><thead>")
            append(row(header, "th"))
            append("</thead><tbody>")
            rows.forEach { append(row(it, "td")) }
            append("</tbody></table></div>")
        }
    }

    private fun tableCells(line: String): List<String> =
        line.trim().removePrefix("|").removeSuffix("|").split('|').map { it.trim() }

    private fun tableAlignments(divider: String, width: Int): List<String> =
        tableCells(divider).map { cell ->
            when {
                cell.startsWith(':') && cell.endsWith(':') -> "center"
                cell.endsWith(':') -> "right"
                else -> "left"
            }
        }.let { it + List((width - it.size).coerceAtLeast(0)) { "left" } }

    private fun inline(source: String, c: Colors): String {
        val protected = mutableListOf<String>()
        fun protect(html: String): String {
            val marker = "@@PHONOLAB_${protected.size}@@"
            protected += html
            return marker
        }

        var value = escape(source)
        value = Regex("`([^`\\n]+)`").replace(value) { match ->
            protect("<code class=\"inline-code\">${match.groupValues[1]}</code>")
        }
        value = Regex("\\[([^]]+)]\\((https?://[^\\s)]+)\\)").replace(value) { match ->
            "<a href=\"${match.groupValues[2]}\">${match.groupValues[1]}</a>"
        }
        value = Regex("\\*\\*\\*(.+?)\\*\\*\\*").replace(value, "<strong><em>$1</em></strong>")
        value = Regex("\\*\\*(.+?)\\*\\*").replace(value, "<strong>$1</strong>")
        value = Regex("~~(.+?)~~").replace(value, "<del>$1</del>")
        value = Regex("(?<!\\*)\\*([^*\\n]+)\\*(?!\\*)").replace(value, "<em>$1</em>")
        protected.forEachIndexed { index, html -> value = value.replace("@@PHONOLAB_${index}@@", html) }
        return value
    }

    private fun page(content: String, c: Colors): String = """
        <!doctype html><html><head>
        <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
        <style>
          *{box-sizing:border-box} html,body{margin:0;padding:0;background:transparent}
          body{color:${c.text};font-family:sans-serif;font-size:14px;line-height:1.5;overflow-wrap:anywhere}
          p{margin:0 0 8px} h1,h2,h3{color:${c.accent};line-height:1.25;margin:12px 0 6px}h1{font-size:20px}h2{font-size:17px}h3{font-size:15px}
          strong{font-weight:700} del{color:${c.subdued}} a{color:${c.accent};text-decoration:none}
          hr{border:0;border-top:1px solid ${c.border};margin:10px 0} ul,ol{margin:5px 0 9px;padding-left:22px} li{margin:2px 0}
          blockquote{border-left:3px solid ${c.accent};color:${c.subdued};margin:8px 0;padding:4px 0 4px 10px}
          .inline-code{background:${c.codeSurface};border:1px solid ${c.border};border-radius:4px;color:${c.accent};font-family:monospace;font-size:.9em;padding:1px 4px}
          .code-card{background:${c.codeSurface};border:1px solid ${c.border};border-radius:7px;margin:9px 0;overflow:hidden}
          .code-head{background:${c.surface};border-bottom:1px solid ${c.border};padding:5px 10px}.language{color:${c.accent};font-family:monospace;font-size:10px;font-weight:700;letter-spacing:.05em}
          pre{margin:0;overflow-x:auto;padding:10px 12px;white-space:pre;font-family:monospace;font-size:12px;line-height:1.55}pre code{color:${c.text};font-family:inherit}
          .table-wrap{border:1px solid ${c.border};border-radius:7px;margin:9px 0;overflow-x:auto}table{border-collapse:collapse;min-width:100%;width:max-content;font-size:12px}th{background:${c.codeSurface};color:${c.text};font-weight:700}th,td{border-bottom:1px solid ${c.border};padding:6px 8px;vertical-align:top}td{color:${c.subdued}}tbody tr:last-child td{border-bottom:0}
        </style></head><body><main id="content">$content</main>
        <script>window.PhonolabChat={render:function(html){document.getElementById('content').innerHTML=html;}};</script>
        </body></html>
    """.trimIndent()

    private fun escape(value: String): String = value
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")

    private fun color(context: Context, id: Int): String =
        String.format("#%06X", 0xFFFFFF and ContextCompat.getColor(context, id))

    private data class Colors(
        val text: String,
        val subdued: String,
        val accent: String,
        val surface: String,
        val codeSurface: String,
        val border: String,
    )
}
