package org.nativelab.phonolab.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import org.nativelab.phonolab.ApiModelConfig
import org.nativelab.phonolab.LlamaRuntime
import org.nativelab.phonolab.PhonoLabApp
import org.nativelab.phonolab.PhonoLabStore
import org.nativelab.phonolab.R
import org.nativelab.phonolab.adapter.ApiModelAdapter
import org.nativelab.phonolab.adapter.ModelAdapter
import org.nativelab.phonolab.data.ModelConfig
import org.nativelab.phonolab.data.ModelManager
import java.util.UUID

class ModelsFragment : Fragment() {

    private lateinit var store: PhonoLabStore
    private lateinit var modelManager: ModelManager
    private lateinit var modelAdapter: ModelAdapter
    private lateinit var apiModelAdapter: ApiModelAdapter
    private lateinit var runtime: LlamaRuntime

    // Param fields
    private lateinit var cfgName: TextView
    private lateinit var cfgCtx: EditText
    private lateinit var cfgThreads: EditText
    private lateinit var cfgTemp: EditText
    private lateinit var cfgTopP: EditText
    private lateinit var cfgTopK: EditText
    private lateinit var cfgRep: EditText
    private lateinit var cfgMaxTokens: EditText
    private lateinit var cfgParamWarn: TextView

    private var selectedModel: ModelConfig? = null

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        return inflater.inflate(R.layout.fragment_models, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val app = requireActivity().application as PhonoLabApp
        store = app.store
        modelManager = app.modelManager
        runtime = app.runtime

        // Model list
        val rvModels = view.findViewById<RecyclerView>(R.id.rv_models)
        modelAdapter = ModelAdapter { model -> onModelSelected(model) }
        rvModels.apply {
            layoutManager = LinearLayoutManager(context)
            adapter = modelAdapter
        }

        // Param fields
        cfgName = view.findViewById(R.id.cfg_name)
        cfgCtx = view.findViewById(R.id.cfg_ctx)
        cfgThreads = view.findViewById(R.id.cfg_threads)
        cfgTemp = view.findViewById(R.id.cfg_temp)
        cfgTopP = view.findViewById(R.id.cfg_topp)
        cfgTopK = view.findViewById(R.id.cfg_topk)
        cfgRep = view.findViewById(R.id.cfg_rep)
        cfgMaxTokens = view.findViewById(R.id.cfg_max_tokens)
        cfgParamWarn = view.findViewById(R.id.cfg_param_warn)

        // Buttons
        view.findViewById<MaterialButton>(R.id.btn_browse_gguf).setOnClickListener { browseGguf() }
        view.findViewById<MaterialButton>(R.id.btn_load_selected).setOnClickListener { loadSelected() }
        view.findViewById<MaterialButton>(R.id.btn_remove_model).setOnClickListener { removeSelected() }
        view.findViewById<MaterialButton>(R.id.btn_save_params).setOnClickListener { saveParams() }

        // API Models
        val rvApiModels = view.findViewById<RecyclerView>(R.id.rv_api_models)
        apiModelAdapter = ApiModelAdapter(
            onEdit = { model -> showApiModelDialog(model) },
            onDelete = { model -> deleteApiModel(model) },
        )
        rvApiModels.apply {
            layoutManager = LinearLayoutManager(context)
            adapter = apiModelAdapter
        }
        view.findViewById<MaterialButton>(R.id.btn_add_api_model).setOnClickListener {
            showApiModelDialog(null)
        }

        refreshModelList()
    }

    private fun refreshModelList() {
        modelManager.syncDiscovery()
        val activePath = runtime.loadedModelPath()
        modelAdapter.setModels(modelManager.all())
        refreshApiModels()
    }

    private fun refreshApiModels() {
        val models = store.getApiModels()
        apiModelAdapter.setModels(models)
        view?.findViewById<View>(R.id.api_models_empty)?.visibility =
            if (models.isEmpty()) View.VISIBLE else View.GONE
    }

    private fun showApiModelDialog(existing: ApiModelConfig?) {
        val ctx = context ?: return
        val dialogView = LayoutInflater.from(ctx).inflate(R.layout.dialog_api_model, null)
        val etProvider = dialogView.findViewById<EditText>(R.id.et_api_provider)
        val etModelName = dialogView.findViewById<EditText>(R.id.et_api_model_name)
        val etBaseUrl = dialogView.findViewById<EditText>(R.id.et_api_base_url)
        val etApiKey = dialogView.findViewById<EditText>(R.id.et_api_key)

        if (existing != null) {
            etProvider.setText(existing.provider)
            etModelName.setText(existing.modelName)
            etBaseUrl.setText(existing.baseUrl)
            etApiKey.setText(existing.apiKey)
        }

        AlertDialog.Builder(ctx)
            .setTitle(if (existing != null) "Edit API Model" else "Add API Model")
            .setView(dialogView)
            .setPositiveButton("Save") { _, _ ->
                val provider = etProvider.text.toString().trim()
                val modelName = etModelName.text.toString().trim()
                val baseUrl = etBaseUrl.text.toString().trim()
                val apiKey = etApiKey.text.toString().trim()
                if (provider.isEmpty() || modelName.isEmpty() || baseUrl.isEmpty()) {
                    Toast.makeText(ctx, "Provider, model name, and base URL are required.", Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }
                val models = store.getApiModels().toMutableList()
                val newModel = ApiModelConfig(
                    id = existing?.id ?: UUID.randomUUID().toString(),
                    provider = provider,
                    modelName = modelName,
                    baseUrl = baseUrl,
                    apiKey = apiKey,
                )
                val idx = models.indexOfFirst { it.id == existing?.id }
                if (idx >= 0) models[idx] = newModel else models.add(newModel)
                store.saveApiModels(models)
                refreshApiModels()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun deleteApiModel(model: ApiModelConfig) {
        val models = store.getApiModels().toMutableList()
        models.removeAll { it.id == model.id }
        store.saveApiModels(models)
        refreshApiModels()
    }

    private fun onModelSelected(model: ModelConfig) {
        selectedModel = model
        cfgName.text = model.name
        cfgCtx.setText(model.ctx.toString())
        cfgThreads.setText(model.threads.toString())
        cfgTemp.setText(model.temperature.toString())
        cfgTopP.setText(model.topP.toString())
        cfgTopK.setText(model.topK.toString())
        cfgRep.setText(model.repeatPenalty.toString())
        cfgMaxTokens.setText(model.maxTokens.toString())
        checkWarnings()
    }

    private fun browseGguf() {
        // On Android, we can't easily browse files without SAF
        // Instead, show a message about placing files in the models directory
        Toast.makeText(context,
            "Place .gguf files in:\n${store.modelsDir.absolutePath}",
            Toast.LENGTH_LONG
        ).show()
        refreshModelList()
    }

    private fun loadSelected() {
        val model = modelAdapter.selected()
        if (model == null) {
            Toast.makeText(context, "Select a model first.", Toast.LENGTH_SHORT).show()
            return
        }
        try {
            runtime.load(java.io.File(model.path))
            Toast.makeText(context, "Loaded: ${model.name}", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(context, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun removeSelected() {
        val model = modelAdapter.selected() ?: return
        modelManager.remove(model.path)
        refreshModelList()
        Toast.makeText(context, "Removed: ${model.name}", Toast.LENGTH_SHORT).show()
    }

    private fun saveParams() {
        val model = selectedModel
        if (model == null) {
            Toast.makeText(context, "Select a model first.", Toast.LENGTH_SHORT).show()
            return
        }
        try {
            val ctx = cfgCtx.text.toString().toInt()
            val threads = cfgThreads.text.toString().toInt()
            val temp = cfgTemp.text.toString().toFloat()
            val topP = cfgTopP.text.toString().toFloat()
            val topK = cfgTopK.text.toString().toInt()
            val rep = cfgRep.text.toString().toFloat()
            val maxTokens = cfgMaxTokens.text.toString().toInt()

            // Validate
            val warnings = mutableListOf<String>()
            if (ctx > 24576) warnings.add("Context $ctx is very high")
            if (temp > 2.0f) warnings.add("Temperature > 2.0")
            if (topP <= 0f || topP > 1f) warnings.add("Top-P must be 0..1")
            if (topK < 0) warnings.add("Top-K cannot be negative")
            if (rep <= 0f) warnings.add("Repeat Penalty must be positive")
            if (maxTokens < 0) warnings.add("Max Tokens cannot be negative")

            if (warnings.isNotEmpty()) {
                cfgParamWarn.text = warnings.joinToString("\n")
                cfgParamWarn.visibility = View.VISIBLE
                return
            }

            modelManager.update(model.path) { cfg ->
                cfg.ctx = ctx
                cfg.threads = threads
                cfg.temperature = temp
                cfg.topP = topP
                cfg.topK = topK
                cfg.repeatPenalty = rep
                cfg.maxTokens = maxTokens
            }
            cfgParamWarn.visibility = View.GONE
            Toast.makeText(context, "Parameters saved for ${model.name}", Toast.LENGTH_SHORT).show()
            refreshModelList()
        } catch (e: NumberFormatException) {
            Toast.makeText(context, "Invalid parameter value.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun checkWarnings() {
        val warnings = mutableListOf<String>()
        try {
            val ctx = cfgCtx.text.toString().toInt()
            if (ctx > 16384) warnings.add("Context $ctx is high for mobile")
        } catch (_: NumberFormatException) {}
        try {
            val temp = cfgTemp.text.toString().toFloat()
            if (temp > 1.5f) warnings.add("Temperature > 1.5")
        } catch (_: NumberFormatException) {}

        if (warnings.isNotEmpty()) {
            cfgParamWarn.text = warnings.joinToString("\n")
            cfgParamWarn.visibility = View.VISIBLE
        } else {
            cfgParamWarn.visibility = View.GONE
        }
    }
}
