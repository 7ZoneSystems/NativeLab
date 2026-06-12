package org.nativelab.phonolab

import android.app.Activity
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Spannable
import android.text.SpannableString
import android.text.style.ForegroundColorSpan
import android.text.style.StyleSpan
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import java.io.File
import java.util.concurrent.Executors

class MainActivity : Activity() {
    private lateinit var store: PhonoLabStore
    private lateinit var downloader: SafeDownloader
    private lateinit var runtime: LlamaRuntime
    private lateinit var statusIcon: TextView
    private lateinit var statusText: TextView
    private lateinit var outputText: TextView
    private lateinit var promptInput: EditText
    private lateinit var modelSpinner: Spinner
    private lateinit var progressBar: ProgressBar
    private lateinit var sendBtn: Button
    private lateinit var setupBtn: Button
    private lateinit var downloadBtn: Button
    private lateinit var loadBtn: Button

    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private val catalogLabels = ModelCatalog.items.map { "Download: ${it.label}" }

    // Theme colors (NativeLab dark palette)
    private val cBg0 = Color.parseColor("#09090d")
    private val cBg1 = Color.parseColor("#0f0f15")
    private val cBg2 = Color.parseColor("#141420")
    private val cSurface = Color.parseColor("#1e1e2e")
    private val cSurface2 = Color.parseColor("#252538")
    private val cAccent = Color.parseColor("#55C2A4")
    private val cAccentDim = Color.parseColor("#1a3d33")
    private val cTxt = Color.parseColor("#ededf5")
    private val cTxt2 = Color.parseColor("#7a7a9a")
    private val cTxt3 = Color.parseColor("#48485e")
    private val cBdr = Color.parseColor("#232335")
    private val cBdr2 = Color.parseColor("#2d2d45")
    private val cOk = Color.parseColor("#1cb88a")
    private val cWarn = Color.parseColor("#e8971a")
    private val cErr = Color.parseColor("#e84848")
    private val cBubbleUser = Color.parseColor("#0e0c26")
    private val cBubbleAst = Color.parseColor("#0c0c14")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Immersive dark status bar
        window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS)
        window.statusBarColor = cBg0
        window.navigationBarColor = cBg0

        store = PhonoLabStore(this)
        downloader = SafeDownloader(store)
        runtime = LlamaRuntime(this, store)
        setContentView(buildUi())
        refreshModels()

        // Auto-setup on first launch
        if (store.isFirstLaunch()) {
            store.markLaunched()
            autoSetup()
        } else {
            setStatus("idle", getString(R.string.status_initial))
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // UI BUILD
    // ═══════════════════════════════════════════════════════════════════

    private fun buildUi(): ScrollView {
        val scroll = ScrollView(this).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            )
            setBackgroundColor(cBg0)
            scrollBarSize = dp(3)
            isVerticalScrollBarEnabled = true
        }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(12), dp(16), dp(16))
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            )
        }

        // ── Status Bar Card ─────────────────────────────────────────
        root.addView(buildStatusCard())

        root.addView(spacer(10))

        // ── Model Section ───────────────────────────────────────────
        root.addView(sectionLabel(getString(R.string.section_model)))
        root.addView(spacer(4))
        root.addView(buildModelCard())

        root.addView(spacer(10))

        // ── Actions Row ─────────────────────────────────────────────
        root.addView(sectionLabel(getString(R.string.section_runtime)))
        root.addView(spacer(4))
        root.addView(buildActionsCard())

        root.addView(spacer(10))

        // ── Progress ────────────────────────────────────────────────
        progressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progress = 0
            progressDrawable = getDrawable(R.drawable.ph_progress_drawable)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(4),
            )
        }
        root.addView(progressBar)

        root.addView(spacer(10))

        // ── Chat Section ────────────────────────────────────────────
        root.addView(sectionLabel(getString(R.string.section_chat)))
        root.addView(spacer(4))
        root.addView(buildChatCard())

        root.addView(spacer(10))

        // ── Input Bar ───────────────────────────────────────────────
        root.addView(buildInputBar())

        scroll.addView(root)
        return scroll
    }

    private fun buildStatusCard(): LinearLayout {
        val card = cardContainer()

        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        statusIcon = TextView(this).apply {
            text = "●"
            textSize = 14f
            setTextColor(cTxt3)
        }
        row.addView(statusIcon, LinearLayout.LayoutParams(dp(20), dp(20)))

        statusText = TextView(this).apply {
            text = getString(R.string.status_ready)
            textSize = 13f
            setTextColor(cTxt2)
            setPadding(dp(8), 0, 0, 0)
        }
        row.addView(statusText, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))

        card.addView(row)
        return card
    }

    private fun buildModelCard(): LinearLayout {
        val card = cardContainer()

        // Model spinner
        modelSpinner = Spinner(this).apply {
            background = getDrawable(R.drawable.ph_spinner_bg)
            setPadding(dp(12), dp(10), dp(12), dp(10))
        }
        card.addView(modelSpinner, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            dp(44),
        ))

        return card
    }

    private fun buildActionsCard(): LinearLayout {
        val card = cardContainer()

        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }

        setupBtn = styledButton(getString(R.string.btn_setup), true).apply {
            setOnClickListener { setupLlama() }
        }
        downloadBtn = styledButton(getString(R.string.btn_download), false).apply {
            setOnClickListener { downloadSelectedModel() }
        }
        loadBtn = styledButton(getString(R.string.btn_load), false).apply {
            setOnClickListener { loadSelectedModel() }
        }

        val lp = LinearLayout.LayoutParams(0, dp(42), 1f)
        lp.marginEnd = dp(6)
        row.addView(setupBtn, lp)

        val lp2 = LinearLayout.LayoutParams(0, dp(42), 1f)
        lp2.marginStart = dp(3)
        lp2.marginEnd = dp(3)
        row.addView(downloadBtn, lp2)

        val lp3 = LinearLayout.LayoutParams(0, dp(42), 1f)
        lp3.marginStart = dp(6)
        row.addView(loadBtn, lp3)

        card.addView(row)
        return card
    }

    private fun buildChatCard(): FrameLayout {
        val frame = FrameLayout(this).apply {
            background = getDrawable(R.drawable.ph_card_bg)
            minimumHeight = dp(200)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
            ).apply { weight = 0f }
        }

        outputText = TextView(this).apply {
            textSize = 14f
            setTextColor(cTxt)
            setPadding(dp(14), dp(12), dp(14), dp(12))
            setTextIsSelectable(true)
            setLineSpacing(0f, 1.4f)
        }

        val scroll = ScrollView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(280),
            )
        }
        scroll.addView(outputText)
        frame.addView(scroll)
        return frame
    }

    private fun buildInputBar(): LinearLayout {
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = getDrawable(R.drawable.ph_card_bg)
            setPadding(dp(10), dp(8), dp(10), dp(8))
        }

        promptInput = EditText(this).apply {
            hint = getString(R.string.hint_prompt)
            setHintTextColor(cTxt3)
            setTextColor(cTxt)
            textSize = 14f
            background = getDrawable(R.drawable.ph_input_bg)
            setPadding(dp(12), dp(10), dp(12), dp(10))
            minLines = 1
            maxLines = 4
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        bar.addView(promptInput)

        bar.addView(spacer(8))

        sendBtn = styledButton(getString(R.string.btn_send), true).apply {
            layoutParams = LinearLayout.LayoutParams(dp(72), dp(42))
            setOnClickListener { sendPrompt() }
        }
        bar.addView(sendBtn)

        return bar
    }

    // ═══════════════════════════════════════════════════════════════════
    // WIDGET HELPERS
    // ═══════════════════════════════════════════════════════════════════

    private fun cardContainer(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = getDrawable(R.drawable.ph_card_bg)
            setPadding(dp(14), dp(12), dp(14), dp(12))
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            )
        }
    }

    private fun styledButton(text: String, primary: Boolean): Button {
        return Button(this).apply {
            this.text = text
            textSize = 13f
            isAllCaps = false
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            if (primary) {
                background = getDrawable(R.drawable.ph_btn_primary)
                setTextColor(Color.WHITE)
            } else {
                background = getDrawable(R.drawable.ph_btn_secondary)
                setTextColor(cTxt)
            }
            stateListAnimator = null
            elevation = 0f
        }
    }

    private fun sectionLabel(text: String): TextView {
        return TextView(this).apply {
            this.text = text
            textSize = 10f
            setTextColor(cTxt3)
            typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
            letterSpacing = 0.08f
            setPadding(dp(2), 0, 0, 0)
        }
    }

    private fun spacer(heightDp: Int): View {
        return View(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(heightDp),
            )
        }
    }

    private fun dp(value: Int): Int {
        return (value * resources.displayMetrics.density).toInt()
    }

    // ═══════════════════════════════════════════════════════════════════
    // STATUS & CHAT
    // ═══════════════════════════════════════════════════════════════════

    private fun setStatus(state: String, text: String) {
        val color = when (state) {
            "ok" -> cOk
            "warn", "downloading" -> cWarn
            "err" -> cErr
            "loading", "generating" -> cAccent
            else -> cTxt3
        }
        statusIcon.setTextColor(color)
        statusText.text = text
    }

    private fun appendChat(text: String) {
        outputText.append(text)
        // Auto-scroll to bottom
        val parent = outputText.parent?.parent as? ScrollView
        parent?.post { parent.fullScroll(View.FOCUS_DOWN) }
    }

    private fun appendChatBubble(prefix: String, text: String, isUser: Boolean) {
        val color = if (isUser) cAccent else cTxt2
        val spannable = SpannableString("$prefix$text\n")
        spannable.setSpan(
            ForegroundColorSpan(color),
            0, prefix.length,
            Spannable.SPAN_EXCLUSIVE_EXCLUSIVE,
        )
        spannable.setSpan(
            StyleSpan(Typeface.BOLD),
            0, prefix.length,
            Spannable.SPAN_EXCLUSIVE_EXCLUSIVE,
        )
        spannable.setSpan(
            ForegroundColorSpan(cTxt),
            prefix.length, prefix.length + text.length,
            Spannable.SPAN_EXCLUSIVE_EXCLUSIVE,
        )
        outputText.append(spannable)
        val parent = outputText.parent?.parent as? ScrollView
        parent?.post { parent.fullScroll(View.FOCUS_DOWN) }
    }

    // ═══════════════════════════════════════════════════════════════════
    // ACTIONS
    // ═══════════════════════════════════════════════════════════════════

    private fun autoSetup() {
        if (runtime.findCli() != null) {
            setStatus("ok", getString(R.string.status_setup_done))
            store.markBinaryInstalled()
            return
        }

        setStatus("loading", getString(R.string.status_auto_setup))

        worker.execute {
            try {
                // Try auto binary install
                val cli = runtime.autoInstallBinary { done, total, label ->
                    ui {
                        val pct = if (total > 0) ((done * 100) / total).toInt().coerceIn(0, 100) else 0
                        progressBar.progress = pct
                        setStatus("loading", "Installing $label: $pct%")
                    }
                }
                ui {
                    progressBar.progress = 0
                    if (cli != null) {
                        setStatus("ok", getString(R.string.status_setup_done))
                    } else {
                        // Fall back to source pull
                        setStatus("warn", "Binary download failed. Pulling source instead…")
                        pullSource()
                    }
                }
            } catch (e: Exception) {
                ui {
                    progressBar.progress = 0
                    setStatus("warn", "Auto-setup failed. Tap Setup to retry.")
                }
            }
        }
    }

    private fun setupLlama() = runJob(getString(R.string.status_setup_running)) {
        // First try auto binary
        val cli = runtime.autoInstallBinary { done, total, label ->
            ui {
                val pct = if (total > 0) ((done * 100) / total).toInt().coerceIn(0, 100) else 0
                progressBar.progress = pct
                setStatus("loading", "Installing $label: $pct%")
            }
        }
        if (cli != null) {
            ui {
                progressBar.progress = 0
                setStatus("ok", getString(R.string.status_setup_done))
            }
            return@runJob
        }

        // Fall back to source pull
        pullSource()
    }

    private fun pullSource() {
        val source = runtime.pullLlamaSource { done, total, label ->
            ui {
                val pct = if (total > 0) ((done * 100) / total).toInt().coerceIn(0, 100) else 0
                progressBar.progress = pct
                setStatus("downloading", "Source: $pct%")
            }
        }
        ui {
            progressBar.progress = 0
            setStatus("ok", "Source ready: ${source.name}. Bundle llama-cli for mobile execution.")
        }
    }

    private fun downloadSelectedModel() = runJob(getString(R.string.status_downloading)) {
        val selected = selectedCandidate()
        val model = downloader.downloadModel(selected) { done, total, label ->
            ui {
                val pct = if (total > 0) ((done * 100) / total).toInt().coerceIn(0, 100) else 0
                progressBar.progress = pct
                setStatus("downloading", "$label: $pct%")
            }
        }
        ui {
            progressBar.progress = 0
            setStatus("ok", "Downloaded ${model.name}")
            refreshModels()
        }
    }

    private fun loadSelectedModel() = runJob(getString(R.string.status_loading)) {
        val selected = selectedText()
        require(!selected.startsWith("Download:")) {
            getString(R.string.error_download_first)
        }
        val model = File(selected)
        runtime.load(model)
        ui { setStatus("ok", "Loaded ${model.name}") }
    }

    private fun sendPrompt() {
        val text = promptInput.text.toString().trim()
        if (text.isEmpty()) return
        promptInput.setText("")
        appendChatBubble("You: ", text, true)
        runJob(getString(R.string.status_generating)) {
            if (!runtime.isModelLoaded()) {
                val selected = selectedText()
                if (selected.isNotEmpty() && !selected.startsWith("Download:")) {
                    runtime.load(File(selected))
                }
            }
            runtime.generate(text) { token -> ui { appendChat(token) } }
            ui {
                appendChat("\n")
                setStatus("ok", getString(R.string.status_ready))
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // HELPERS
    // ═══════════════════════════════════════════════════════════════════

    private fun selectedCandidate(): ModelCandidate {
        val selected = selectedText()
        val label = selected.removePrefix("Download: ").trim()
        return ModelCatalog.items.firstOrNull { it.label == label }
            ?: ModelCatalog.chooseForDevice(totalRamMb())
    }

    private fun selectedText(): String {
        return modelSpinner.selectedItem?.toString().orEmpty()
    }

    private fun refreshModels() {
        val local = store.modelFiles().map { it.absolutePath }
        val values = if (local.isEmpty()) catalogLabels else local + catalogLabels
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, values)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        modelSpinner.adapter = adapter
    }

    private fun runJob(label: String, block: () -> Unit) {
        setStatus("loading", label)
        worker.execute {
            try {
                block()
            } catch (exc: Exception) {
                ui {
                    progressBar.progress = 0
                    setStatus("err", getString(R.string.status_error))
                    appendChatBubble("Error: ", exc.message ?: exc.javaClass.simpleName, false)
                }
            }
        }
    }

    private fun totalRamMb(): Int {
        val runtime = Runtime.getRuntime()
        return ((runtime.maxMemory() / (1024L * 1024L)).coerceAtLeast(2048L)).toInt()
    }

    private fun ui(block: () -> Unit) {
        main.post(block)
    }
}
