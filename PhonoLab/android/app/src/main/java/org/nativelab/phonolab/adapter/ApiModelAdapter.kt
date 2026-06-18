package org.nativelab.phonolab.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import org.nativelab.phonolab.ApiModelConfig
import org.nativelab.phonolab.R

class ApiModelAdapter(
    private val onEdit: (ApiModelConfig) -> Unit,
    private val onDelete: (ApiModelConfig) -> Unit,
) : RecyclerView.Adapter<ApiModelAdapter.ViewHolder>() {

    private val models = mutableListOf<ApiModelConfig>()

    fun setModels(list: List<ApiModelConfig>) {
        models.clear()
        models.addAll(list)
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_api_model, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val model = models[position]
        holder.provider.text = model.provider
        holder.name.text = model.modelName
        holder.editBtn.setOnClickListener { onEdit(model) }
        holder.deleteBtn.setOnClickListener { onDelete(model) }
    }

    override fun getItemCount() = models.size

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val provider: TextView = view.findViewById(R.id.api_model_provider)
        val name: TextView = view.findViewById(R.id.api_model_name)
        val editBtn: ImageButton = view.findViewById(R.id.btn_edit_api_model)
        val deleteBtn: ImageButton = view.findViewById(R.id.btn_delete_api_model)
    }
}
