package org.nativelab.phonolab.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import org.nativelab.phonolab.LlamaRuntime
import org.nativelab.phonolab.PhonoLabStore
import org.nativelab.phonolab.R
import org.nativelab.phonolab.adapter.ModelAdapter
import org.nativelab.phonolab.data.ModelConfig
import org.nativelab.phonolab.data.ModelManager

class ModelsFragment : Fragment() {

    private lateinit var store: PhonoLabStore
    private lateinit var modelManager: ModelManager
    private lateinit var modelAdapter: ModelAdapter
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

        store = PhonoLabStore(requireContext())
        modelManager = ModelManager(store)
        runtime = LlamaRuntime(requireContext(), store)

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

        refreshModelList()
    }

    private fun refreshModelList() {
        modelManager.syncDiscovery()
        val activePath = runtime.loadedModelPath()
        modelAdapter.setModels(modelManager.all())
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
