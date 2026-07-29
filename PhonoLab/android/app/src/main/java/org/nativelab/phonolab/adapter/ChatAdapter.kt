package org.nativelab.phonolab.adapter

import android.annotation.SuppressLint
import android.graphics.Color
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import org.nativelab.phonolab.R
import org.nativelab.phonolab.data.ChatMessage
import org.nativelab.phonolab.util.MarkdownRenderer
import android.os.Handler
import android.os.Looper

class ChatAdapter : RecyclerView.Adapter<ChatAdapter.ViewHolder>() {

    companion object {
        private const val TYPE_USER = 0
        private const val TYPE_AST = 1
    }

    private val messages = mutableListOf<ChatMessage>()
    private val renderHandler = Handler(Looper.getMainLooper())
    private var pendingRenderedPosition = RecyclerView.NO_POSITION
    private val renderLastAssistantMessage = Runnable {
        val position = pendingRenderedPosition
        pendingRenderedPosition = RecyclerView.NO_POSITION
        if (position in messages.indices) notifyItemChanged(position)
    }

    fun addMessage(msg: ChatMessage) {
        messages.add(msg)
        notifyItemInserted(messages.size - 1)
    }

    /**
     * Append streamed token to the last message AND return it for session sync.
     * Returns the updated message, or null if adapter is empty.
     */
    fun appendToLast(text: String): ChatMessage? {
        if (messages.isEmpty()) return null
        val idx = messages.size - 1
        val last = messages[idx]
        val updated = last.copy(content = last.content + text)
        messages[idx] = updated
        // Reloading a WebView for every token visibly flashes.  Coalesce token
        // invalidations; the holder patches its already-loaded document below.
        pendingRenderedPosition = idx
        renderHandler.removeCallbacks(renderLastAssistantMessage)
        renderHandler.postDelayed(renderLastAssistantMessage, 50)
        return updated
    }

    fun clear() {
        renderHandler.removeCallbacks(renderLastAssistantMessage)
        pendingRenderedPosition = RecyclerView.NO_POSITION
        val size = messages.size
        messages.clear()
        notifyItemRangeRemoved(0, size)
    }

    fun setMessages(msgs: List<ChatMessage>) {
        renderHandler.removeCallbacks(renderLastAssistantMessage)
        pendingRenderedPosition = RecyclerView.NO_POSITION
        messages.clear()
        messages.addAll(msgs)
        notifyDataSetChanged()
    }

    override fun getItemViewType(position: Int): Int {
        return if (messages[position].role == "user") TYPE_USER else TYPE_AST
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val layout = if (viewType == TYPE_USER) R.layout.item_message_user else R.layout.item_message_ast
        val view = LayoutInflater.from(parent.context).inflate(layout, parent, false)
        return ViewHolder(view)
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val msg = messages[position]

        if (holder.webView != null && msg.role != "user") {
            holder.content.visibility = View.GONE
            holder.webView.visibility = View.VISIBLE
            holder.renderMarkdown(
                key = "$position:${msg.role}:${msg.timestamp}",
                markdown = msg.content,
            )
        } else {
            holder.content.text = msg.content
        }

        holder.time.text = msg.timestamp
    }

    override fun getItemCount() = messages.size

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val content: TextView = view.findViewById(R.id.msg_content)
        val time: TextView = view.findViewById(R.id.msg_time)
        val webView: WebView? = view.findViewById(R.id.msg_math_webview)
        private var boundKey: String? = null
        private var latestMarkdown = ""
        private var pageReady = false

        init {
            webView?.apply {
                setBackgroundColor(Color.TRANSPARENT)
                settings.javaScriptEnabled = true
                settings.domStorageEnabled = false
                settings.loadWithOverviewMode = false
                settings.useWideViewPort = false
                settings.builtInZoomControls = false
                settings.displayZoomControls = false
                isVerticalScrollBarEnabled = false
                isHorizontalScrollBarEnabled = false
                webViewClient = object : WebViewClient() {
                    override fun onPageFinished(view: WebView, url: String) {
                        pageReady = true
                        applyLatestMarkdown()
                    }
                }
            }
        }

        fun renderMarkdown(key: String, markdown: String) {
            val view = webView ?: return
            latestMarkdown = markdown
            if (boundKey != key) {
                boundKey = key
                pageReady = false
                view.loadDataWithBaseURL(
                    "file:///android_asset/",
                    MarkdownRenderer.document(view.context, markdown),
                    "text/html",
                    "UTF-8",
                    null,
                )
            } else if (pageReady) {
                applyLatestMarkdown()
            }
        }

        private fun applyLatestMarkdown() {
            val view = webView ?: return
            if (!pageReady) return
            view.evaluateJavascript(MarkdownRenderer.updateScript(view.context, latestMarkdown), null)
            // WebView content does not participate in wrap_content measurement.
            // Resize after the DOM has been patched without recreating its surface.
            view.postDelayed({ measureWebContent() }, 32)
            view.postDelayed({ measureWebContent() }, 160)
        }

        private fun measureWebContent() {
            val view = webView ?: return
            view.evaluateJavascript("Math.ceil(document.documentElement.scrollHeight)") { result ->
                val cssPixels = result.trim().trim('"').toIntOrNull() ?: return@evaluateJavascript
                val height = (cssPixels * view.resources.displayMetrics.density).toInt().coerceAtLeast(1)
                if (view.layoutParams.height != height) {
                    view.layoutParams = view.layoutParams.apply { this.height = height }
                }
            }
        }
    }
}
