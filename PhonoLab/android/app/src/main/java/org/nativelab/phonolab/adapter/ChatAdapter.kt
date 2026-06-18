package org.nativelab.phonolab.adapter

import android.annotation.SuppressLint
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import org.nativelab.phonolab.R
import org.nativelab.phonolab.data.ChatMessage
import org.nativelab.phonolab.util.MathHelper

class ChatAdapter : RecyclerView.Adapter<ChatAdapter.ViewHolder>() {

    companion object {
        private const val TYPE_USER = 0
        private const val TYPE_AST = 1
    }

    private val messages = mutableListOf<ChatMessage>()

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
        notifyItemChanged(idx)
        return updated
    }

    fun clear() {
        val size = messages.size
        messages.clear()
        notifyItemRangeRemoved(0, size)
    }

    fun setMessages(msgs: List<ChatMessage>) {
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
            val hasMath = MathHelper.hasMath(msg.content)
            if (hasMath && msg.content.isNotEmpty()) {
                // Show WebView with KaTeX rendering
                holder.content.visibility = View.GONE
                holder.webView.visibility = View.VISIBLE
                holder.webView.settings.javaScriptEnabled = true
                holder.webView.settings.domStorageEnabled = true
                holder.webView.settings.loadWithOverviewMode = true
                holder.webView.settings.useWideViewPort = true
                holder.webView.setBackgroundColor(0x00000000) // transparent

                // Wrap auto-detected math in $...$ for KaTeX
                val processedContent = MathHelper.wrapAutoMath(msg.content)

                val safeContent = processedContent
                    .replace("\\", "\\\\")
                    .replace("\"", "\\\"")
                    .replace("'", "\\'")
                    .replace("\n", "\\n")
                    .replace("\r", "")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")

                val html = """
                    <!DOCTYPE html>
                    <html><head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
                    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
                    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
                    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
                    <style>
                      * { margin:0; padding:0; }
                      body { font-family:sans-serif; font-size:14px; line-height:1.5; color:#ededf5; background:transparent; word-wrap:break-word; }
                      .katex { font-size:1.05em; }
                      .katex-display { margin:8px 0; overflow-x:auto; }
                      pre,code { background:rgba(255,255,255,0.06); border-radius:4px; padding:1px 4px; font-size:13px; }
                      pre { padding:8px; margin:6px 0; overflow-x:auto; }
                      pre code { background:none; padding:0; }
                      p { margin:4px 0; }
                    </style></head><body>
                    <div id="c">$safeContent</div>
                    <script>
                      var el=document.getElementById('c');
                      el.innerHTML=el.innerHTML.replace(/\n/g,'<br>');
                      renderMathInElement(el,{
                        delimiters:[
                          {left:'\$\$',right:'\$\$',display:true},
                          {left:'\$',right:'\$',display:false},
                          {left:'\\\\(',right:'\\\\)',display:false},
                          {left:'\\\\[',right:'\\\\]',display:true}
                        ],
                        throwOnError:false
                      });
                    </script>
                    </body></html>
                """.trimIndent()

                holder.webView.loadDataWithBaseURL(
                    "file:///android_asset/",
                    html,
                    "text/html",
                    "UTF-8",
                    null
                )
            } else {
                // Plain text — show TextView
                holder.content.visibility = View.VISIBLE
                holder.webView.visibility = View.GONE
                holder.content.text = msg.content
            }
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
    }
}
