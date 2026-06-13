package org.nativelab.phonolab.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import org.nativelab.phonolab.R
import org.nativelab.phonolab.data.ChatSession

class SessionAdapter(
    private val onSessionClick: (ChatSession) -> Unit,
    private val onSessionLongClick: (ChatSession) -> Unit,
) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    companion object {
        private const val TYPE_HEADER = 0
        private const val TYPE_SESSION = 1
    }

    private val items = mutableListOf<Any>() // String (date) or ChatSession
    private var activeId = ""

    fun setSessions(sessions: List<ChatSession>, activeSessionId: String) {
        items.clear()
        activeId = activeSessionId

        // Group by date
        val grouped = sessions.groupBy { it.created }
        for (date in grouped.keys.sortedDescending()) {
            items.add(date)
            for (session in grouped[date]!!) {
                items.add(session)
            }
        }
        notifyDataSetChanged()
    }

    fun filter(query: String, sessions: List<ChatSession>) {
        val filtered = if (query.isEmpty()) sessions
        else sessions.filter { it.title.contains(query, ignoreCase = true) }
        setSessions(filtered, activeId)
    }

    override fun getItemViewType(position: Int): Int {
        return if (items[position] is String) TYPE_HEADER else TYPE_SESSION
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        return if (viewType == TYPE_HEADER) {
            val view = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_session_header, parent, false)
            HeaderViewHolder(view)
        } else {
            val view = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_session, parent, false)
            SessionViewHolder(view)
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        when (holder) {
            is HeaderViewHolder -> {
                holder.dateText.text = items[position] as String
            }
            is SessionViewHolder -> {
                val session = items[position] as ChatSession
                holder.title.text = session.title
                holder.date.text = session.created
                val isActive = session.id == activeId
                holder.itemView.alpha = if (isActive) 1.0f else 0.8f
                holder.title.setTextColor(
                    if (isActive) holder.itemView.context.getColor(R.color.ph_accent)
                    else holder.itemView.context.getColor(R.color.ph_txt)
                )
                holder.itemView.setOnClickListener { onSessionClick(session) }
                holder.itemView.setOnLongClickListener {
                    onSessionLongClick(session)
                    true
                }
            }
        }
    }

    override fun getItemCount() = items.size

    class HeaderViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val dateText: TextView = view.findViewById(R.id.header_date)
    }

    class SessionViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val title: TextView = view.findViewById(R.id.session_title)
        val date: TextView = view.findViewById(R.id.session_date)
    }
}
