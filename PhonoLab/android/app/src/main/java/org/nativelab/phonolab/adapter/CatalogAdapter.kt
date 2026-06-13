package org.nativelab.phonolab.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import org.nativelab.phonolab.ModelCandidate
import org.nativelab.phonolab.ModelCatalog
import org.nativelab.phonolab.R

class CatalogAdapter(
    private val onDownload: (ModelCandidate) -> Unit,
) : RecyclerView.Adapter<CatalogAdapter.ViewHolder>() {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_catalog_model, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val item = ModelCatalog.items[position]
        holder.name.text = item.label
        holder.repo.text = "${item.repo}\nMin RAM: ${item.minRamMb} MB · Quants: ${item.quantPreferences.joinToString(", ")}"
        holder.downloadBtn.setOnClickListener { onDownload(item) }
    }

    override fun getItemCount() = ModelCatalog.items.size

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val name: TextView = view.findViewById(R.id.catalog_name)
        val repo: TextView = view.findViewById(R.id.catalog_repo)
        val downloadBtn: MaterialButton = view.findViewById(R.id.btn_download_catalog)
    }
}
