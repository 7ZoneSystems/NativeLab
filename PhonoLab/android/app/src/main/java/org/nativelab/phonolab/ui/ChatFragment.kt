package org.nativelab.phonolab.ui

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.OpenableColumns
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.nativelab.phonolab.*
import org.nativelab.phonolab.adapter.ChatAdapter
import org.nativelab.phonolab.data.*
import org.nativelab.phonolab.theme.ThemeManager
import org.nativelab.phonolab.util.UiHelpers
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executors

class ChatFragment : Fragment() {

    interface Host {
        fun onSessionChanged()
    }

    private lateinit var store: PhonoLabStore
    private lateinit var runtime: LlamaRuntime
    private lateinit var sessionManager: SessionManager
    private lateinit var modelManager: ModelManager

    private lateinit var chatAdapter: ChatAdapter
    private lateinit var rvChat: RecyclerView
    private lateinit var promptInput: EditText
    private lateinit var btnSend: ImageButton
    private lateinit var btnModelPicker: TextView
    private lateinit var generatingBanner: View
    private lateinit var generatingText: TextView
    private lateinit var btnStop: ImageButton
    private lateinit var downloadBar: View
    private lateinit var downloadLabel: TextView
    private lateinit var downloadProgress: ProgressBar

    private var activeSession: ChatSession? = null
    private var selectedModelPath: String = ""
    private val sessionLogs = mutableListOf<String>()
    private var userScrolledUp = false
    private var isGenerating = false
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private val timeFmt = object : ThreadLocal<SimpleDateFormat>() {
        override fun initialValue() = SimpleDateFormat("HH:mm:ss", Locale.US)
    }

    private lateinit var btnAttach: ImageButton
    private lateinit var ragBar: View
    private lateinit var tvRagStatus: TextView
    private lateinit var progressRag: ProgressBar
    private lateinit var attachmentChip: View
    private lateinit var tvAttachmentName: TextView
    private lateinit var btnClearAttachment: ImageButton

    private var attachedImageUri: Uri? = null
    private var attachedDocumentUri: Uri? = null
    private var attachedDocumentText: String? = null
    private var ragProcessor: RagProcessor? = null
    private var pendingRagResult: RagProcessor.RagResult? = null

    private lateinit var imagePickerLauncher: ActivityResultLauncher<Intent>
    private lateinit var documentPickerLauncher: ActivityResultLauncher<Array<String>>

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        return inflater.inflate(R.layout.fragment_chat, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Register pickers
        imagePickerLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                val uri = result.data?.data
                if (uri != null) {
                    attachedImageUri = uri
                    attachedDocumentUri = null
                    attachedDocumentText = null
                    showAttachmentPreview(uri, isImage = true)
                }
            }
        }

        documentPickerLauncher = registerForActivityResult(
            ActivityResultContracts.OpenDocument()
        ) { uri ->
            if (uri != null) {
                attachedDocumentUri = uri
                attachedImageUri = null
                attachedDocumentText = null
                pendingRagResult = null
                showAttachmentPreview(uri, isImage = false)
                processDocumentWithRag(uri)
            }
        }

        // Use singletons from PhonoLabApp (survive theme switch / recreate)
        val app = requireActivity().application as PhonoLabApp
        store = app.store
        sessionManager = app.sessionManager
        modelManager = app.modelManager
        runtime = app.runtime

        rvChat = view.findViewById(R.id.rv_chat)
        promptInput = view.findViewById(R.id.prompt_input)
        btnSend = view.findViewById(R.id.btn_send)
        btnModelPicker = view.findViewById(R.id.btn_model_picker)
        generatingBanner = view.findViewById(R.id.generating_banner)
        generatingText = view.findViewById(R.id.generating_text)
        btnStop = view.findViewById(R.id.btn_stop)
        downloadBar = view.findViewById(R.id.download_bar)
        downloadLabel = view.findViewById(R.id.download_label)
        downloadProgress = view.findViewById(R.id.download_progress)

        ragProcessor = RagProcessor(requireContext())
        ragBar = view.findViewById(R.id.rag_bar)
        tvRagStatus = view.findViewById(R.id.tv_rag_status)
        progressRag = view.findViewById(R.id.progress_rag)
        attachmentChip = view.findViewById(R.id.attachment_chip)
        tvAttachmentName = view.findViewById(R.id.tv_attachment_name)
        btnClearAttachment = view.findViewById(R.id.btn_clear_attachment)
        btnClearAttachment.setOnClickListener { clearAttachment() }

        chatAdapter = ChatAdapter()
        val lm = LinearLayoutManager(context).apply { stackFromEnd = true }
        rvChat.layoutManager = lm
        rvChat.adapter = chatAdapter

        // Detect when user scrolls up manually — pause auto-scroll
        rvChat.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrollStateChanged(rv: RecyclerView, newState: Int) {
                if (newState == RecyclerView.SCROLL_STATE_DRAGGING) {
                    // User is dragging — check if they scrolled away from bottom
                    val lastVisible = lm.findLastCompletelyVisibleItemPosition()
                    val total = chatAdapter.itemCount
                    userScrolledUp = total > 0 && lastVisible < total - 1
                }
            }

            override fun onScrolled(rv: RecyclerView, dx: Int, dy: Int) {
                // Re-enable auto-scroll if user scrolls back to bottom
                if (userScrolledUp) {
                    val lastVisible = lm.findLastCompletelyVisibleItemPosition()
                    val total = chatAdapter.itemCount
                    if (total > 0 && lastVisible >= total - 1) {
                        userScrolledUp = false
                    }
                }
            }
        })

        btnModelPicker.setOnClickListener { showModelPicker() }
        btnSend.setOnClickListener { sendPrompt() }
        btnStop.setOnClickListener { stopGeneration() }

        btnAttach = view.findViewById(R.id.btn_attach)
        btnAttach.setOnClickListener { showAttachmentSheet() }

        refreshModels()
        refreshRuntimeStatus()
    }

    // ── Logging ──────────────────────────────────────────────────────

    private fun logSession(msg: String) {
        val entry = "${timeFmt.get()!!.format(Date())} $msg"
        sessionLogs.add(entry)
        if (sessionLogs.size > 500) sessionLogs.removeAt(0)
        activeSession?.addLog(entry)
    }

    private fun runOnUi(block: () -> Unit) {
        main.post { if (isAdded) block() }
    }

    fun showLogs() {
        val logs = activeSession?.logs ?: sessionLogs
        val text = if (logs.isEmpty()) "(No logs for this session)" else logs.joinToString("\n")
        AlertDialog.Builder(requireContext())
            .setTitle("Session Logs")
            .setMessage(text)
            .setPositiveButton("Close", null)
            .show()
    }

    // ── Model Picker ─────────────────────────────────────────────────
    //
    // Shows ALL models (downloaded + catalog) in one list.
    //   ✓ = downloaded → click to load
    //   ⬇ = not downloaded → click to download then load

    private fun showModelPicker() {
        val localModels = modelManager.all()

        data class PickerItem(val label: String, val key: String, val downloaded: Boolean)

        val items = mutableListOf<PickerItem>()

        // Catalog models — mark which are downloaded
        for (cat in ModelCatalog.items) {
            val localMatch = localModels.find { m ->
                m.path.contains(cat.key, ignoreCase = true) || m.repo == cat.repo
            }
            if (localMatch != null) {
                items.add(PickerItem("✓  ${cat.label}", localMatch.path, true))
            } else {
                items.add(PickerItem("⬇  ${cat.label}", "catalog:${cat.key}", false))
            }
        }

        // Local models not in catalog
        for (m in localModels) {
            val inCatalog = ModelCatalog.items.any { cat ->
                m.path.contains(cat.key, ignoreCase = true) || m.repo == cat.repo
            }
            if (!inCatalog) {
                val name = m.name.removeSuffix(".gguf").ifEmpty { m.path.substringAfterLast("/") }
                items.add(PickerItem("✓  $name", m.path, true))
            }
        }

        // Themed dialog with custom adapter
        val ctx = requireContext()
        val adapter = object : ArrayAdapter<PickerItem>(ctx, R.layout.ph_spinner_dropdown_item, items) {
            override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
                val v = super.getView(position, convertView, parent)
                (v as? TextView)?.text = getItem(position)?.label ?: ""
                return v
            }
        }

        AlertDialog.Builder(ctx)
            .setTitle("Select Model")
            .setAdapter(adapter) { _, which ->
                val item = items[which]
                if (item.downloaded) {
                    loadModelFromPath(item.key)
                } else {
                    val catalogKey = item.key.removePrefix("catalog:")
                    val candidate = ModelCatalog.items.find { it.key == catalogKey }
                    if (candidate != null) downloadAndLoad(candidate)
                }
            }
            .show()
    }

    private fun refreshModels() {
        modelManager.syncDiscovery()
        val models = modelManager.all()
        val displayName = if (models.isNotEmpty()) {
            val active = selectedModelPath.ifEmpty { runtime.loadedModelPath() ?: "" }
            val match = models.find { it.path == active }
            (match?.name ?: models.first().name).removeSuffix(".gguf").take(18)
        } else {
            "Model"
        }
        btnModelPicker.text = "$displayName ▾"
    }

    // ── Runtime Status ───────────────────────────────────────────────

    private fun refreshRuntimeStatus() {
        if (runtime.isServerRunning()) {
            val model = runtime.loadedModelPath()?.substringAfterLast("/")?.removeSuffix(".gguf") ?: "none"
            logSession("Server running: $model")
        } else {
            val rt = runtime.runtimeStatus()
            if (rt.ready) logSession("Runtime ready") else logSession("Runtime not installed")
        }
    }

    // ── Session Management ───────────────────────────────────────────

    fun newChat() {
        activeSession = ChatSession.new()
        sessionLogs.clear()
        activeSession?.let { sessionManager.save(it) }
        chatAdapter.clear()
        logSession("New chat started")
    }

    fun loadSession(session: ChatSession) {
        activeSession = session
        sessionLogs.clear()
        sessionLogs.addAll(session.logs)
        chatAdapter.setMessages(session.messages)
        logSession("Loaded session: ${session.title}")
        scrollToBottom()
    }

    fun getCurrentSession(): ChatSession? = activeSession

    // ── Download + Load ──────────────────────────────────────────────

    private fun downloadAndLoad(candidate: ModelCandidate) {
        logSession("Downloading ${candidate.label}…")

        // Show green progress bar in chat area
        runOnUi {
            downloadLabel.text = "Downloading ${candidate.label}…"
            downloadProgress.progress = 0
            downloadBar.visibility = View.VISIBLE
            promptInput.isEnabled = false
            btnSend.isEnabled = false
        }

        worker.execute {
            try {
                val downloader = SafeDownloader(store)
                val model = downloader.downloadModel(candidate) { done, total, label ->
                    val pct = UiHelpers.calcPercent(done, total)
                    runOnUi {
                        downloadProgress.progress = pct
                        downloadLabel.text = "$label: $pct% (${done / (1024 * 1024)} MB)"
                    }
                }
                modelManager.add(model, repo = candidate.repo)
                logSession("Downloaded: ${model.name} (${model.length() / (1024 * 1024)} MB)")

                runOnUi {
                    downloadBar.visibility = View.GONE
                    refreshModels()
                }

                // Auto-load the downloaded model
                loadModelFromPath(model.absolutePath)

            } catch (e: Exception) {
                logSession("ERROR download: ${e.message}")
                runOnUi {
                    downloadBar.visibility = View.GONE
                    promptInput.isEnabled = true
                    btnSend.isEnabled = true
                    Toast.makeText(context, "Download failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun loadModelFromPath(path: String) {
        val model = File(path)
        if (!model.exists()) {
            Toast.makeText(context, "Model not found: ${model.name}", Toast.LENGTH_SHORT).show()
            logSession("ERROR: Model not found: $path")
            return
        }
        val sizeMb = model.length() / (1024 * 1024)
        selectedModelPath = path
        logSession("Loading model: ${model.name} ($sizeMb MB)")
        setStatus("loading", "Loading ${model.name}…")

        worker.execute {
            val error = runtime.load(model)
            if (error == null) {
                logSession("Model loaded: ${model.name}")
                runOnUi {
                    generatingBanner.visibility = View.GONE
                    refreshModels()
                    setStatus("ok", "Ready · ${model.name}")
                }
            } else {
                logSession("ERROR load: $error")
                runOnUi {
                    generatingBanner.visibility = View.GONE
                    setStatus("err", error)
                }
            }
        }
    }

    // ── Generate ─────────────────────────────────────────────────────

    /** Find a model to auto-load: selected > last used > first available. */
    private fun findAutoLoadModel(): String? {
        // 1. Already selected
        if (selectedModelPath.isNotEmpty() && File(selectedModelPath).exists()) {
            return selectedModelPath
        }
        // 2. Server still running with a model
        val serverModel = runtime.loadedModelPath()
        if (serverModel != null && File(serverModel).exists()) return serverModel
        // 3. First available downloaded model
        val models = modelManager.all()
        return models.firstOrNull()?.path?.let { if (File(it).exists()) it else null }
    }

    private fun sendPrompt() {
        var text = promptInput.text.toString().trim()
        if (text.isEmpty() && attachedImageUri == null && attachedDocumentUri == null) return

        // Inject RAG context if available
        val rag = pendingRagResult
        if (rag != null) {
            val query = text.ifEmpty { "Summarize this document" }
            val chunks = ragProcessor?.retrieveChunks(query, rag.chunks) ?: rag.chunks.take(3)
            val context = "Relevant document excerpts (${rag.filename}):\n" +
                chunks.joinToString("\n---\n")
            text = "$context\n\nUser question: $text"
        } else {
            // Fallback: raw document text
            val docText = attachedDocumentText
            if (docText != null) {
                text = "[Document content]\n$docText\n[End document]\n\n$text"
            }
        }

        // Handle image attachment
        val imageUri = attachedImageUri
        if (imageUri != null) {
            text = "[Image attached: ${getFileName(imageUri)}]\n\n$text"
        }

        promptInput.setText("")
        clearAttachment()

        if (activeSession == null) {
            activeSession = ChatSession.new()
            sessionLogs.clear()
        }

        val session = activeSession ?: return
        session.addMessage("user", text)
        chatAdapter.addMessage(ChatMessage("user", text))
        session.addMessage("assistant", "")
        chatAdapter.addMessage(ChatMessage("assistant", ""))
        scrollToBottom()

        logSession("User: ${text.take(80)}${if (text.length > 80) "…" else ""}")
        isGenerating = true
        userScrolledUp = false
        setStatus("generating", "Generating…")

        worker.execute {
            try {
                if (!runtime.isModelLoaded()) {
                    val modelPath = findAutoLoadModel()
                    if (modelPath != null) {
                        val model = File(modelPath)
                        logSession("Auto-loading: ${model.name}")
                        runOnUi { setStatus("loading", "Loading ${model.name}…") }
                        val loadError = runtime.load(model)
                        if (loadError != null) {
                            logSession("ERROR load: $loadError")
                            runOnUi {
                                isGenerating = false
                                setStatus("err", loadError)
                                showError(loadError)
                            }
                            return@execute
                        }
                        selectedModelPath = modelPath
                        runOnUi { refreshModels() }
                    } else {
                        runOnUi {
                            isGenerating = false
                            setStatus("warn", "No model available")
                            showError("No model available. Download one from the model picker.")
                        }
                        return@execute
                    }
                }

                val t0 = System.currentTimeMillis()
                val result = runtime.generate(text) { token ->
                    runOnUi {
                        val updated = chatAdapter.appendToLast(token)
                        if (updated != null) {
                            activeSession?.let { sess ->
                                val idx = sess.messages.indexOfLast { it.role == "assistant" }
                                if (idx >= 0) sess.messages[idx] = updated
                            }
                        }
                    }
                }

                if (result == "[ABORTED]") {
                    logSession("Generation aborted")
                    activeSession?.let { sessionManager.save(it) }
                    runOnUi {
                        isGenerating = false
                        setStatus("warn", "Stopped")
                    }
                    return@execute
                }

                if (result.startsWith("[ERROR]")) {
                    val errMsg = result.removePrefix("[ERROR] ").trim()
                    logSession("ERROR: $errMsg")
                    runOnUi {
                        isGenerating = false
                        setStatus("err", errMsg)
                        showError(errMsg)
                    }
                    return@execute
                }

                logSession("Generated in ${System.currentTimeMillis() - t0}ms")
                activeSession?.let { sessionManager.save(it) }
                runOnUi {
                    isGenerating = false
                    setStatus("ok", "Ready")
                    (activity as? Host)?.onSessionChanged()
                }
            } catch (e: Exception) {
                Log.e("ChatFragment", "sendPrompt failed", e)
                logSession("CRASH: ${e.message}")
                try { runtime.unload() } catch (_: Exception) {}
                runOnUi {
                    isGenerating = false
                    setStatus("err", "Crash: ${e.message}")
                    showError("Generation crashed: ${e.message}")
                }
            }
        }
    }

    private fun stopGeneration() {
        logSession("Stopped by user")
        isGenerating = false
        // Abort generation without killing server — model stays loaded
        runtime.abort()
        setStatus("warn", "Stopped")
    }

    // ── Status ───────────────────────────────────────────────────────

    private fun setStatus(state: String, text: String) {
        val showBanner = state == "generating" || state == "loading"
        if (showBanner) {
            generatingText.text = text
            generatingBanner.visibility = View.VISIBLE
            promptInput.isEnabled = false
            btnSend.isEnabled = false
        } else {
            generatingBanner.visibility = View.GONE
            promptInput.isEnabled = true
            btnSend.isEnabled = true
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────

    /** Show an error message in the chat as an assistant message. */
    private fun showError(message: String) {
        val msg = ChatMessage("assistant", "⚠️ $message")
        chatAdapter.addMessage(msg)
        activeSession?.messages?.add(msg)
        activeSession?.let { sessionManager.save(it) }
        scrollToBottom()
    }

    private fun scrollToBottom() {
        if (userScrolledUp || chatAdapter.itemCount == 0) return
        // Use instant scroll during generation to avoid jitter
        if (isGenerating) {
            rvChat.scrollToPosition(chatAdapter.itemCount - 1)
        } else {
            rvChat.smoothScrollToPosition(chatAdapter.itemCount - 1)
        }
    }

    // ── Attachments ─────────────────────────────────────────────────

    private fun showAttachmentSheet() {
        val sheet = AttachmentBottomSheet()
        sheet.callback = object : AttachmentBottomSheet.AttachmentCallback {
            override fun onImageSelected() {
                val intent = Intent(Intent.ACTION_PICK).apply {
                    type = "image/*"
                }
                imagePickerLauncher.launch(intent)
            }
            override fun onDocumentSelected() {
                documentPickerLauncher.launch(arrayOf(
                    "application/pdf",
                    "text/plain",
                    "application/msword",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ))
            }
        }
        sheet.show(childFragmentManager, AttachmentBottomSheet.TAG)
    }

    private fun showAttachmentPreview(uri: Uri, isImage: Boolean) {
        val name = getFileName(uri)
        tvAttachmentName.text = if (isImage) "🖼 $name" else "📄 $name"
        attachmentChip.visibility = View.VISIBLE
        promptInput.hint = "Ask about ${name}…"
    }

    private fun processDocumentWithRag(uri: Uri) {
        val processor = ragProcessor ?: return
        ragBar.visibility = View.VISIBLE
        progressRag.progress = 0
        tvRagStatus.text = "Processing document…"

        lifecycleScope.launch {
            try {
                val result = processor.processDocument(uri) { progress, status ->
                    runOnUi {
                        progressRag.progress = (progress * 100).toInt()
                        tvRagStatus.text = status
                    }
                }
                pendingRagResult = result
                attachedDocumentText = null
                logSession("Document processed: ${result.filename} (${result.chunks.size} chunks, ${result.totalChars} chars)")
                withContext(Dispatchers.Main) {
                    ragBar.visibility = View.GONE
                    showAttachmentPreview(uri, isImage = false)
                }
            } catch (e: Exception) {
                logSession("RAG error: ${e.message}")
                withContext(Dispatchers.Main) {
                    if (isAdded) {
                        ragBar.visibility = View.GONE
                        Toast.makeText(context, "Document error: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }
    }

    private fun getFileName(uri: Uri): String {
        val ctx = context ?: return "file"
        return try {
            ctx.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                cursor.moveToFirst()
                cursor.getString(nameIndex)
            } ?: uri.lastPathSegment ?: "file"
        } catch (_: Exception) {
            uri.lastPathSegment ?: "file"
        }
    }

    private fun clearAttachment() {
        attachedImageUri = null
        attachedDocumentUri = null
        attachedDocumentText = null
        pendingRagResult = null
        attachmentChip.visibility = View.GONE
        promptInput.hint = "Message PhonoLab…"
    }

    private fun totalRamMb(): Int {
        return ((Runtime.getRuntime().maxMemory() / (1024L * 1024L)).coerceAtLeast(2048L)).toInt()
    }

    override fun onDestroyView() {
        // Don't kill worker during generation — server keeps running via PhonoLabApp
        if (!isGenerating) {
            worker.shutdown()
        }
        super.onDestroyView()
    }
}
