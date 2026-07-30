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
        private const val STREAM_DEBOUNCE_MS = 60L
        private const val HEIGHT_SETTLE_MS = 150L
    }

    private val messages = mutableListOf<ChatMessage>()
    private val renderHandler = Handler(Looper.getMainLooper())
    private var pendingRenderedPosition = RecyclerView.NO_POSITION
    private var isStreaming = false
    private var streamingPosition = RecyclerView.NO_POSITION

    private val renderLastAssistantMessage = Runnable {
        val position = pendingRenderedPosition
        pendingRenderedPosition = RecyclerView.NO_POSITION
        if (position in messages.indices) notifyItemChanged(position)
    }

    fun setStreaming(streaming: Boolean) {
        isStreaming = streaming
        if (!streaming && streamingPosition != RecyclerView.NO_POSITION) {
            val pos = streamingPosition
            streamingPosition = RecyclerView.NO_POSITION
            if (pos in messages.indices) {
                pendingRenderedPosition = pos
                renderHandler.postDelayed({
                    val p = pendingRenderedPosition
                    pendingRenderedPosition = RecyclerView.NO_POSITION
                    if (p in messages.indices) notifyItemChanged(p)
                }, 100)
            }
        }
    }

    fun addMessage(msg: ChatMessage) {
        messages.add(msg)
        notifyItemInserted(messages.size - 1)
    }

    fun appendToLast(text: String): ChatMessage? {
        if (messages.isEmpty()) return null
        val idx = messages.size - 1
        val last = messages[idx]
        val updated = last.copy(content = last.content + text)
        messages[idx] = updated
        if (isStreaming) {
            streamingPosition = idx
            notifyItemChanged(idx)
        } else {
            pendingRenderedPosition = idx
            renderHandler.removeCallbacks(renderLastAssistantMessage)
            renderHandler.postDelayed(renderLastAssistantMessage, STREAM_DEBOUNCE_MS)
        }
        return updated
    }

    fun clear() {
        renderHandler.removeCallbacks(renderLastAssistantMessage)
        pendingRenderedPosition = RecyclerView.NO_POSITION
        streamingPosition = RecyclerView.NO_POSITION
        val size = messages.size
        messages.clear()
        notifyItemRangeRemoved(0, size)
    }

    fun setMessages(msgs: List<ChatMessage>) {
        renderHandler.removeCallbacks(renderLastAssistantMessage)
        pendingRenderedPosition = RecyclerView.NO_POSITION
        streamingPosition = RecyclerView.NO_POSITION
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
                streaming = isStreaming && position == streamingPosition,
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
        private var lastHeight = 0
        private var heightStable = false

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
                        applyLatestMarkdown(false)
                        view.evaluateJavascript("renderAllMath()", null)
                    }
                }
            }
        }

        fun renderMarkdown(key: String, markdown: String, streaming: Boolean) {
            val view = webView ?: return
            latestMarkdown = markdown
            if (boundKey != key) {
                boundKey = key
                pageReady = false
                lastHeight = 0
                heightStable = false
                view.loadDataWithBaseURL(
                    "file:///android_asset/",
                    MarkdownRenderer.document(view.context, markdown),
                    "text/html",
                    "UTF-8",
                    null,
                )
            } else if (pageReady) {
                applyLatestMarkdown(streaming)
            }
        }

        private fun applyLatestMarkdown(streaming: Boolean) {
            val view = webView ?: return
            if (!pageReady) return
            if (streaming) {
                view.evaluateJavascript(MarkdownRenderer.streamingUpdateScript(view.context, latestMarkdown), null)
            } else {
                view.evaluateJavascript(MarkdownRenderer.updateScript(view.context, latestMarkdown), null)
            }
            scheduleHeightMeasurement(streaming)
        }

        private fun scheduleHeightMeasurement(streaming: Boolean) {
            val view = webView ?: return
            view.postDelayed({ measureWebContent(streaming) }, 30)
            if (!streaming) {
                view.postDelayed({ measureWebContent(false) }, 100)
                view.postDelayed({ measureWebContent(false) }, 300)
                view.postDelayed({ measureWebContent(false) }, 600)
            }
        }

        private fun measureWebContent(streaming: Boolean) {
            val view = webView ?: return
            view.evaluateJavascript("Math.ceil(document.documentElement.scrollHeight)") { result ->
                val cssPixels = result.trim().trim('"').toIntOrNull() ?: return@evaluateJavascript
                val density = view.resources.displayMetrics.density
                val height = (cssPixels * density).toInt().coerceAtLeast(1)
                if (height == lastHeight) {
                    heightStable = true
                    return@evaluateJavascript
                }
                lastHeight = height
                if (streaming) {
                    if (height > (view.layoutParams.height ?: 0)) {
                        view.layoutParams = view.layoutParams.apply { this.height = height }
                    }
                } else {
                    if (view.layoutParams.height != height) {
                        view.layoutParams = view.layoutParams.apply { this.height = height }
                    }
                }
            }
        }

        fun resetForReuse() {
            boundKey = null
            latestMarkdown = ""
            pageReady = false
            lastHeight = 0
            heightStable = false
        }
    }
}
