package org.nativelab.phonolab.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import org.nativelab.phonolab.R
import org.nativelab.phonolab.data.ModelConfig
import java.io.File

class ModelAdapter(
    private val onSelect: (ModelConfig) -> Unit,
) : RecyclerView.Adapter<ModelAdapter.ViewHolder>() {

    private val models = mutableListOf<ModelConfig>()
    private var selectedPath = ""

    fun setModels(list: List<ModelConfig>, activePath: String = "") {
        models.clear()
        models.addAll(list)
        selectedPath = activePath
        notifyDataSetChanged()
    }

    fun selected(): ModelConfig? = models.find { it.path == selectedPath }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_model, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val model = models[position]
        val isSelected = model.path == selectedPath

        holder.name.text = model.name
        holder.info.text = buildString {
            if (model.repo.isNotEmpty()) append("${model.repo} · ")
            append("ctx:${model.ctx} · t:${model.threads}")
        }

        val file = File(model.path)
        holder.size.text = if (file.exists()) {
            val mb = file.length() / (1024 * 1024)
            "${mb} MB"
        } else ""

        holder.statusIcon.text = "●"
        holder.statusIcon.setTextColor(
            if (isSelected) holder.itemView.context.getColor(R.color.ph_ok)
            else holder.itemView.context.getColor(R.color.ph_txt3)
        )

        holder.itemView.setOnClickListener {
            selectedPath = model.path
            notifyDataSetChanged()
            onSelect(model)
        }
    }

    override fun getItemCount() = models.size

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val statusIcon: TextView = view.findViewById(R.id.model_status_icon)
        val name: TextView = view.findViewById(R.id.model_name)
        val info: TextView = view.findViewById(R.id.model_info)
        val size: TextView = view.findViewById(R.id.model_size)
    }
}
