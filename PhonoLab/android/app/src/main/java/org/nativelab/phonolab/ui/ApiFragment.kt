package org.nativelab.phonolab.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import org.nativelab.phonolab.LlamaRuntime
import org.nativelab.phonolab.PhonoLabApp
import org.nativelab.phonolab.PhonoLabStore
import org.nativelab.phonolab.R
import org.nativelab.phonolab.api.ApiConfig
import org.nativelab.phonolab.api.PhonoLabApiServer
import org.nativelab.phonolab.data.ModelManager
import org.nativelab.phonolab.theme.ThemeManager

class ApiFragment : Fragment() {

    private lateinit var store: PhonoLabStore
    private lateinit var runtime: LlamaRuntime
    private lateinit var modelManager: ModelManager
    private lateinit var config: ApiConfig
    private var server: PhonoLabApiServer? = null

    private lateinit var statusIcon: TextView
    private lateinit var statusText: TextView
    private lateinit var localUrl: TextView
    private lateinit var lanUrl: TextView
    private lateinit var localKey: TextView
    private lateinit var lanKey: TextView
    private lateinit var portInput: EditText
    private lateinit var protocolSpinner: Spinner
    private lateinit var requestLog: TextView
    private lateinit var btnStart: MaterialButton
    private lateinit var btnStop: MaterialButton

    private val main = Handler(Looper.getMainLooper())
    private val logUpdater = Runnable { updateLog() }

    private fun runOnUi(block: () -> Unit) {
        main.post { if (isAdded) block() }
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        return inflater.inflate(R.layout.fragment_api, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val app = requireActivity().application as PhonoLabApp
        store = app.store
        runtime = app.runtime
        modelManager = app.modelManager
        config = ApiConfig.load(requireContext())

        // Views
        statusIcon = view.findViewById(R.id.api_status_icon)
        statusText = view.findViewById(R.id.api_status_text)
        localUrl = view.findViewById(R.id.local_url)
        lanUrl = view.findViewById(R.id.lan_url)
        localKey = view.findViewById(R.id.local_key)
        lanKey = view.findViewById(R.id.lan_key)
        portInput = view.findViewById(R.id.port_input)
        protocolSpinner = view.findViewById(R.id.protocol_spinner)
        requestLog = view.findViewById(R.id.request_log)
        btnStart = view.findViewById(R.id.btn_start_server)
        btnStop = view.findViewById(R.id.btn_stop_server)

        // Protocol spinner
        val protocols = listOf("both", "openai", "anthropic")
        protocolSpinner.adapter = ArrayAdapter(requireContext(), R.layout.ph_spinner_item, protocols).apply {
            setDropDownViewResource(R.layout.ph_spinner_dropdown_item)
        }
        protocolSpinner.setSelection(protocols.indexOf(config.protocol).coerceAtLeast(0))

        // Load config
        portInput.setText(config.port.toString())
        updateDisplay()

        // Copy buttons
        view.findViewById<MaterialButton>(R.id.btn_copy_local).setOnClickListener { copyToClipboard("Local URL", localUrl.text.toString()) }
        view.findViewById<MaterialButton>(R.id.btn_copy_lan).setOnClickListener { copyToClipboard("LAN URL", lanUrl.text.toString()) }
        view.findViewById<MaterialButton>(R.id.btn_copy_local_key).setOnClickListener { copyToClipboard("Local API Key", config.localApiKey) }
        view.findViewById<MaterialButton>(R.id.btn_copy_lan_key).setOnClickListener { copyToClipboard("LAN API Key", config.lanApiKey) }

        // Start/Stop
        btnStart.setOnClickListener { startServer() }
        btnStop.setOnClickListener { stopServer() }

        // Log updater
        main.postDelayed(logUpdater, 2000)
    }

    private fun updateDisplay() {
        val port = portInput.text.toString().toIntOrNull() ?: 8787
        localUrl.text = "http://127.0.0.1:$port/v1"
        lanUrl.text = "http://${ApiConfig.detectLanIp()}:$port/v1"
        localKey.text = config.localApiKey
        lanKey.text = config.lanApiKey
        updateServerStatus()
    }

    private fun updateServerStatus() {
        val isRunning = server?.isRunning == true
        val color = if (isRunning) {
            ThemeManager.palette()["ok"] ?: "#1cb88a"
        } else {
            ThemeManager.palette()["txt3"] ?: "#48485e"
        }
        statusIcon.setTextColor(android.graphics.Color.parseColor(color))
        statusText.text = if (isRunning) "Server running on port ${config.port}" else "Server stopped"
        btnStart.isEnabled = !isRunning
        btnStop.isEnabled = isRunning
    }

    private fun startServer() {
        // Save config
        val port = portInput.text.toString().toIntOrNull() ?: 8787
        val protocol = protocolSpinner.selectedItem?.toString() ?: "both"
        config = config.copy(port = port, protocol = protocol)
        config.save(requireContext().getSharedPreferences("phonolab_api_server", Context.MODE_PRIVATE))

        updateDisplay()

        // Capture at creation time to avoid stale references if fragment is recreated
        val capturedRuntime = runtime
        val capturedModelManager = modelManager

        // Create server
        server = PhonoLabApiServer(
            config = config,
            context = requireContext().applicationContext,
            onLog = { msg -> runOnUi { appendLog(msg) } },
            generateFn = { prompt, nPredict, temperature, topP, topK, repeatPenalty ->
                if (!capturedRuntime.isModelLoaded()) {
                    val modelFile = capturedRuntime.loadedModelPath()?.let { java.io.File(it) }
                    if (modelFile != null && modelFile.exists()) {
                        capturedRuntime.load(modelFile)
                    }
                }
                capturedRuntime.generate(prompt) { }
            },
            streamGenerateFn = { prompt, nPredict, temperature, topP, topK, repeatPenalty, onToken ->
                if (!capturedRuntime.isModelLoaded()) {
                    val modelFile = capturedRuntime.loadedModelPath()?.let { java.io.File(it) }
                    if (modelFile != null && modelFile.exists()) {
                        capturedRuntime.load(modelFile)
                    }
                }
                capturedRuntime.generate(prompt, onToken)
            },
            runtimeInfo = {
                mapOf(
                    "loaded" to capturedRuntime.isModelLoaded(),
                    "model" to (capturedRuntime.loadedModelPath()?.substringAfterLast("/")?.removeSuffix(".gguf") ?: "none"),
                    "model_path" to (capturedRuntime.loadedModelPath() ?: ""),
                    "state" to when {
                        capturedRuntime.isModelLoaded() -> "loaded"
                        capturedRuntime.isServerRunning() -> "server_running"
                        else -> "idle"
                    },
                    "ctx" to (capturedRuntime.loadedConfig?.ctx ?: 2048),
                )
            },
            modelList = {
                capturedModelManager.syncDiscovery()
                capturedModelManager.all().map { m ->
                    mapOf("id" to m.name, "path" to m.path)
                }
            },
            loadModelFn = { modelPath ->
                val model = java.io.File(modelPath)
                capturedRuntime.load(model)
            },
            updateConfigFn = { temperature, topP, topK, repeatPenalty, maxTokens, ctx ->
                capturedRuntime.loadedConfig?.let { cfg ->
                    capturedRuntime.updateConfig(temperature, topP, topK, repeatPenalty, maxTokens, ctx)
                }
            },
            getVisionModelFn = {
                val path = capturedRuntime.loadedModelPath() ?: ""
                if (path.isNotEmpty()) {
                    org.nativelab.phonolab.detectVisionModel(path.substringAfterLast("/")).isVision
                } else false
            },
        )

        val result = server?.start() ?: "Failed"
        Toast.makeText(context, result, Toast.LENGTH_SHORT).show()
        updateServerStatus()
    }

    private fun stopServer() {
        server?.stop()
        server = null
        updateServerStatus()
        Toast.makeText(context, "Server stopped", Toast.LENGTH_SHORT).show()
    }

    private fun appendLog(msg: String) {
        requestLog.text = msg + "\n" + requestLog.text.toString()
        // Keep log manageable
        val lines = requestLog.text.toString().lines()
        if (lines.size > 100) {
            requestLog.text = lines.take(100).joinToString("\n")
        }
    }

    private fun updateLog() {
        val logs = server?.getRecentLogs() ?: emptyList()
        if (logs.isNotEmpty() && isAdded) {
            requestLog.text = logs.reversed().joinToString("\n")
        }
        if (isAdded) main.postDelayed(logUpdater, 3000)
    }

    private fun copyToClipboard(label: String, text: String) {
        val clipboard = requireContext().getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText(label, text))
        Toast.makeText(context, "$label copied", Toast.LENGTH_SHORT).show()
    }

    override fun onDestroyView() {
        main.removeCallbacks(logUpdater)
        // Don't stop server on view destroy - keep it running in background
        super.onDestroyView()
    }
}
