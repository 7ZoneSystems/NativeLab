package org.nativelab.phonolab.ui

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import org.nativelab.phonolab.*
import org.nativelab.phonolab.adapter.ChatAdapter
import org.nativelab.phonolab.data.*
import org.nativelab.phonolab.theme.ThemeManager
import java.io.File
import java.util.concurrent.Executors

class ChatFragment : Fragment() {

    private lateinit var store: PhonoLabStore
    private lateinit var runtime: LlamaRuntime
    private lateinit var sessionManager: SessionManager
    private lateinit var modelManager: ModelManager

    private lateinit var chatAdapter: ChatAdapter
    private lateinit var rvChat: RecyclerView
    private lateinit var promptInput: EditText
    private lateinit var btnSend: ImageButton
    private lateinit var statusIcon: TextView
    private lateinit var statusText: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var modelSpinner: Spinner

    private var activeSession: ChatSession? = null
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        return inflater.inflate(R.layout.fragment_chat, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        store = PhonoLabStore(requireContext())
        runtime = LlamaRuntime(requireContext(), store)
        sessionManager = SessionManager(store)
        modelManager = ModelManager(store)

        // Views
        rvChat = view.findViewById(R.id.rv_chat)
        promptInput = view.findViewById(R.id.prompt_input)
        btnSend = view.findViewById(R.id.btn_send)
        statusIcon = view.findViewById(R.id.status_icon)
        statusText = view.findViewById(R.id.status_text)
        progressBar = view.findViewById(R.id.progress_bar)
        modelSpinner = view.findViewById(R.id.model_spinner)

        // Chat RecyclerView
        chatAdapter = ChatAdapter()
        rvChat.apply {
            layoutManager = LinearLayoutManager(context).apply { stackFromEnd = true }
            adapter = chatAdapter
        }

        // Buttons
        view.findViewById<MaterialButton>(R.id.btn_setup).setOnClickListener { setupLlama() }
        view.findViewById<MaterialButton>(R.id.btn_download_model).setOnClickListener { downloadModel() }
        view.findViewById<MaterialButton>(R.id.btn_load_model).setOnClickListener { loadModel() }
        btnSend.setOnClickListener { sendPrompt() }

        // Show initial runtime status
        refreshRuntimeStatus()

        // Auto-setup on first launch
        if (store.isFirstLaunch()) {
            store.markLaunched()
            autoSetup()
        }

        refreshModels()
    }

    /** Check runtime and show real status. */
    private fun refreshRuntimeStatus() {
        val rt = runtime.runtimeStatus()
        if (runtime.isServerRunning()) {
            val model = runtime.loadedModelPath()?.substringAfterLast("/") ?: "none"
            setStatus("ok", "Server running · $model")
        } else if (rt.ready) {
            val verText = if (rt.version.isNotEmpty()) " · ${rt.version.lines().first()}" else ""
            setStatus("ok", "Runtime ready${verText} · Load a model to start")
        } else {
            setStatus("warn", "llama-server not installed · Tap Setup")
        }
    }

    fun newChat() {
        activeSession = ChatSession.new()
        sessionManager.save(activeSession!!)
        chatAdapter.clear()
    }

    fun loadSession(session: ChatSession) {
        activeSession = session
        chatAdapter.setMessages(session.messages)
        scrollToBottom()
    }

    fun getCurrentSession(): ChatSession? = activeSession

    private fun refreshModels() {
        modelManager.syncDiscovery()
        val local = modelManager.all().map { it.path }
        val catalogValues = ModelCatalog.items.map { "catalog:${it.key}" }
        val values = if (local.isEmpty()) catalogValues else local + catalogValues
        val adapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_item, values)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        modelSpinner.adapter = adapter
    }

    private fun autoSetup() {
        if (runtime.findServer() != null) {
            refreshRuntimeStatus()
            return
        }
        setStatus("loading", "First launch: installing llama-server…")
        worker.execute {
            try {
                val server = runtime.autoInstallBinary { done, total, label ->
                    main.post {
                        val pct = if (total > 0) ((done * 100) / total).toInt().coerceIn(0, 100) else 0
                        progressBar.progress = pct
                        progressBar.visibility = View.VISIBLE
                        setStatus("loading", "Installing $label: $pct%")
                    }
                }
                main.post {
                    progressBar.visibility = View.GONE
                    if (server != null) {
                        refreshRuntimeStatus()
                    } else {
                        setStatus("err", "Install returned no binary.")
                    }
                }
            } catch (e: Exception) {
                main.post {
                    progressBar.visibility = View.GONE
                    setStatus("err", "${e.javaClass.simpleName}: ${e.message}")
                }
            }
        }
    }

    private fun setupLlama() {
        setStatus("loading", "Installing llama-server…")
        worker.execute {
            try {
                val server = runtime.autoInstallBinary { done, total, label ->
                    main.post {
                        val pct = if (total > 0) ((done * 100) / total).toInt().coerceIn(0, 100) else 0
                        progressBar.progress = pct
                        progressBar.visibility = View.VISIBLE
                        setStatus("loading", "Installing $label: $pct%")
                    }
                }
                main.post {
                    progressBar.visibility = View.GONE
                    if (server != null) {
                        refreshRuntimeStatus()
                    } else {
                        setStatus("err", "Install returned no binary.")
                    }
                }
            } catch (e: Exception) {
                main.post {
                    progressBar.visibility = View.GONE
                    setStatus("err", "${e.javaClass.simpleName}: ${e.message}")
                }
            }
        }
    }

    private fun downloadModel() {
        val selected = modelSpinner.selectedItem?.toString() ?: ""
        val candidate = if (selected.startsWith("catalog:")) {
            val key = selected.removePrefix("catalog:")
            ModelCatalog.items.firstOrNull { it.key == key }
                ?: ModelCatalog.chooseForDevice(totalRamMb())
        } else {
            ModelCatalog.chooseForDevice(totalRamMb())
        }

        setStatus("downloading", "Downloading ${candidate.label}…")
        worker.execute {
            try {
                val downloader = SafeDownloader(store)
                val model = downloader.downloadModel(candidate) { done, total, label ->
                    main.post {
                        val pct = if (total > 0) ((done * 100) / total).toInt().coerceIn(0, 100) else 0
                        progressBar.progress = pct
                        progressBar.visibility = View.VISIBLE
                        setStatus("downloading", "$label: $pct%")
                    }
                }
                modelManager.add(model, repo = candidate.repo)
                main.post {
                    progressBar.visibility = View.GONE
                    setStatus("ok", "Downloaded ${model.name} (${model.length() / (1024 * 1024)} MB)")
                    refreshModels()
                }
            } catch (e: Exception) {
                main.post {
                    progressBar.visibility = View.GONE
                    setStatus("err", "Download failed: ${e.message}")
                }
            }
        }
    }

    private fun loadModel() {
        val selected = modelSpinner.selectedItem?.toString() ?: ""
        if (selected.startsWith("catalog:")) {
            setStatus("warn", "Download the model first.")
            return
        }
        if (selected.isEmpty()) {
            setStatus("warn", "No model selected.")
            return
        }

        val model = File(selected)
        val sizeMb = model.length() / (1024 * 1024)
        setStatus("loading", "Starting server with ${model.name} ($sizeMb MB)…")
        progressBar.visibility = View.VISIBLE
        progressBar.isIndeterminate = true

        worker.execute {
            try {
                runtime.load(model)
                main.post {
                    progressBar.visibility = View.GONE
                    progressBar.isIndeterminate = false
                    val rtStatus = runtime.runtimeStatus()
                    setStatus("ok", "Server running · ${model.name} ($sizeMb MB) · ${rtStatus.version.lines().firstOrNull() ?: ""}")
                }
            } catch (e: Exception) {
                main.post {
                    progressBar.visibility = View.GONE
                    progressBar.isIndeterminate = false
                    setStatus("err", "Load failed: ${e.message}")
                }
            }
        }
    }

    private fun sendPrompt() {
        val text = promptInput.text.toString().trim()
        if (text.isEmpty()) return
        promptInput.setText("")

        // Ensure session exists
        if (activeSession == null) {
            activeSession = ChatSession.new()
        }

        activeSession!!.addMessage("user", text)
        chatAdapter.addMessage(ChatMessage("user", text))
        chatAdapter.addMessage(ChatMessage("assistant", ""))
        scrollToBottom()

        setStatus("generating", "Generating…")
        worker.execute {
            try {
                // Auto-load if not loaded
                if (!runtime.isModelLoaded()) {
                    val selected = modelSpinner.selectedItem?.toString() ?: ""
                    if (selected.isNotEmpty() && !selected.startsWith("catalog:")) {
                        val model = File(selected)
                        runtime.load(model)
                    } else {
                        throw IllegalStateException("No model loaded. Select and load a model first.")
                    }
                }

                runtime.generate(text) { token ->
                    main.post {
                        chatAdapter.appendToLast(token)
                        scrollToBottom()
                    }
                }

                // Save session
                sessionManager.save(activeSession!!)
                main.post {
                    setStatus("ok", "Ready")
                    (activity as? MainActivity)?.refreshSessionList()
                }
            } catch (e: Exception) {
                main.post {
                    setStatus("err", "Error: ${e.message}")
                    chatAdapter.addMessage(ChatMessage("assistant", "Error: ${e.message}"))
                    scrollToBottom()
                }
            }
        }
    }

    private fun setStatus(state: String, text: String) {
        val color = when (state) {
            "ok" -> ThemeManager.palette()["ok"] ?: "#1cb88a"
            "warn", "downloading" -> ThemeManager.palette()["warn"] ?: "#e8971a"
            "err" -> ThemeManager.palette()["err"] ?: "#e84848"
            "loading", "generating" -> ThemeManager.palette()["accent"] ?: "#55C2A4"
            else -> ThemeManager.palette()["txt3"] ?: "#48485e"
        }
        try {
            statusIcon.setTextColor(android.graphics.Color.parseColor(color))
            statusText.text = text
        } catch (_: Exception) { }
    }

    private fun scrollToBottom() {
        if (chatAdapter.itemCount > 0) {
            rvChat.smoothScrollToPosition(chatAdapter.itemCount - 1)
        }
    }

    private fun totalRamMb(): Int {
        val rt = Runtime.getRuntime()
        return ((rt.maxMemory() / (1024L * 1024L)).coerceAtLeast(2048L)).toInt()
    }

    override fun onDestroyView() {
        // Shutdown worker thread cleanly
        worker.shutdown()
        super.onDestroyView()
    }
}
