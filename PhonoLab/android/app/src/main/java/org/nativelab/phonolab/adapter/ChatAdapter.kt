package org.nativelab.phonolab.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import org.nativelab.phonolab.R
import org.nativelab.phonolab.data.ChatMessage

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

    fun appendToLast(text: String) {
        if (messages.isEmpty()) return
        val last = messages.last()
        messages[messages.size - 1] = last.copy(content = last.content + text)
        notifyItemChanged(messages.size - 1)
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

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val msg = messages[position]
        holder.content.text = msg.content
        holder.time.text = msg.timestamp
    }

    override fun getItemCount() = messages.size

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val content: TextView = view.findViewById(R.id.msg_content)
        val time: TextView = view.findViewById(R.id.msg_time)
    }
}
