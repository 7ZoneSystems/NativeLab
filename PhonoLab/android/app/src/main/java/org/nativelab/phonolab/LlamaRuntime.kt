package org.nativelab.phonolab

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

/**
 * Runtime engine backed by llama-server (persistent HTTP).
 *
 * Binary is bundled in APK → extracted to nativeLibraryDir at install.
 * Server started via JNI fork()+execve() — no W^X issues.
 * Chat via HTTP POST to localhost.
 */
class LlamaRuntime(
    private val context: Context,
    private val store: PhonoLabStore,
    private val storageManager: StorageManager? = null,
) {
    companion object {
        private const val TAG = "LlamaRuntime"
    }

    val cppManager = LlamaCppManager(context, store, storageManager)

    private var serverPid: Int = -1
    private var serverPort: Int = 8080
    private var loadedModelPath: String? = null
    private var _isModelLoaded = false
    private var _isServerRunning = false

    // ── Public API ──────────────────────────────────────────────────

    /** Find the llama-server binary. */
    fun findServer(): File? = cppManager.findServer()

    /** Whether the server process is alive. */
    fun isServerRunning(): Boolean {
        if (serverPid <= 0) return false
        val running = cppManager.isRunning(serverPid)
        if (!running) {
            _isServerRunning = false
            serverPid = -1
        }
        return running
    }

    /** Whether a model is loaded in the server. */
    fun isModelLoaded(): Boolean = _isModelLoaded && isServerRunning()

    /** The currently loaded model path. */
    fun loadedModelPath(): String? = loadedModelPath

    /** Get runtime status for UI display. */
    fun runtimeStatus(): LlamaCppManager.RuntimeStatus = cppManager.status()

    /**
     * Start llama-server with a model.
     * If server is already running with a different model, restarts it.
     */
    fun load(model: File) {
        require(model.exists()) { "Model file does not exist: ${model.absolutePath}" }
        require(model.extension.equals("gguf", ignoreCase = true)) {
            "Not a GGUF file: ${model.name}"
        }
        require(model.length() > 1000) { "Model file is too small" }

        val server = findServer() ?: error(
            "llama-server not found in nativeLibraryDir.\n" +
                "Run setup_binaries.sh and rebuild the APK.\n" +
                "Path: ${context.applicationInfo.nativeLibraryDir}"
        )

        // If server is running with the same model, just return
        if (isServerRunning() && loadedModelPath == model.absolutePath && _isModelLoaded) {
            return
        }

        stopServer()
        startServer(server, model)
        loadedModelPath = model.absolutePath
    }

    /** Unload model and stop server. */
    fun unload() {
        stopServer()
        loadedModelPath = null
        _isModelLoaded = false
    }

    /**
     * Generate text via llama-server HTTP API.
     */
    fun generate(prompt: String, onToken: (String) -> Unit): String {
        // Auto-start if needed
        if (!isServerRunning() && loadedModelPath != null) {
            val model = File(loadedModelPath!!)
            val server = findServer()
            if (server != null && model.exists()) {
                startServer(server, model)
            }
        }

        require(isServerRunning()) { "llama-server is not running. Load a model first." }
        require(_isModelLoaded) { "No model loaded. Select and load a model first." }
        require(prompt.length <= 18_000) { "Prompt too long (${prompt.length} chars). Max: 18,000." }

        val requestBody = JSONObject().apply {
            put("prompt", prompt)
            put("n_predict", 384)
            put("temperature", 0.7)
            put("top_p", 0.9)
            put("repeat_penalty", 1.1)
            put("stream", false)
        }

        val url = URL("http://127.0.0.1:$serverPort/completion")
        val conn = url.openConnection() as HttpURLConnection
        conn.requestMethod = "POST"
        conn.setRequestProperty("Content-Type", "application/json")
        conn.connectTimeout = 5_000
        conn.readTimeout = 300_000
        conn.doOutput = true

        try {
            OutputStreamWriter(conn.outputStream).use { writer ->
                writer.write(requestBody.toString())
                writer.flush()
            }

            val status = conn.responseCode
            if (status != 200) {
                val errBody = try { conn.errorStream?.bufferedReader()?.readText() ?: "" } catch (_: Exception) { "" }
                error("llama-server returned HTTP $status: $errBody")
            }

            val responseBody = conn.inputStream.bufferedReader().readText()
            val json = JSONObject(responseBody)
            val content = json.optString("content", "")
            onToken(content)
            return content
        } finally {
            conn.disconnect()
        }
    }

    // ── Auto-install (legacy fallback) ──────────────────────────────

    fun autoInstallBinary(
        progress: (done: Long, total: Long, label: String) -> Unit,
    ): File? = cppManager.downloadAndInstall(progress)

    // ── Server management ───────────────────────────────────────────

    private fun startServer(serverBinary: File, model: File) {
        stopServer()

        val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
        serverPort = findFreePort()

        Log.d(TAG, "Starting server: ${serverBinary.absolutePath}")
        Log.d(TAG, "Model: ${model.absolutePath} (${model.length()} bytes)")
        Log.d(TAG, "Port: $serverPort, Threads: $threads")

        val pid = cppManager.startServer(
            serverBin = serverBinary,
            modelPath = model,
            port = serverPort,
            threads = threads,
            ctxSize = 2048,
            nPredict = 384,
        )

        serverPid = pid

        // Wait for server to be ready
        val startTime = System.currentTimeMillis()
        val timeout = 60_000L
        var ready = false

        while (System.currentTimeMillis() - startTime < timeout && !ready) {
            val exitCode = cppManager.checkProcess(pid)
            if (exitCode > 0) {
                serverPid = -1
                error("llama-server exited immediately with code $exitCode")
            }

            try {
                val healthConn = URL("http://127.0.0.1:$serverPort/health").openConnection() as HttpURLConnection
                healthConn.connectTimeout = 2_000
                healthConn.readTimeout = 2_000
                try {
                    if (healthConn.responseCode == 200) ready = true
                } finally {
                    healthConn.disconnect()
                }
            } catch (_: Exception) {
                Thread.sleep(500)
            }
        }

        if (!ready) {
            stopServer()
            error("llama-server failed to start within 60s. Model may be too large for device.")
        }

        _isServerRunning = true
        _isModelLoaded = true
        loadedModelPath = model.absolutePath
        Log.d(TAG, "Server ready on port $serverPort")
    }

    private fun stopServer() {
        if (serverPid > 0) {
            cppManager.killServer(serverPid)
            serverPid = -1
        }
        _isServerRunning = false
    }

    /**
     * Kill ALL llama processes on the device. Called on app exit.
     */
    fun killAllLlamaProcesses() {
        stopServer()
        val processNames = listOf("llama-server", "llama-cli", "llama_server", "llama_cli")
        for (name in processNames) {
            try {
                val proc = ProcessBuilder("pkill", "-f", name)
                    .redirectErrorStream(true).start()
                proc.waitFor()
            } catch (_: Exception) {
                try {
                    val proc = ProcessBuilder("killall", name)
                        .redirectErrorStream(true).start()
                    proc.waitFor()
                } catch (_: Exception) {}
            }
        }
        loadedModelPath = null
        _isModelLoaded = false
    }

    private fun findFreePort(): Int {
        try {
            java.net.ServerSocket(0).use { return it.localPort }
        } catch (_: Exception) {
            return 8080
        }
    }

    protected fun finalize() {
        killAllLlamaProcesses()
    }
}
