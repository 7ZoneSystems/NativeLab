package org.nativelab.phonolab

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.Socket
import java.net.URL

/**
 * Runtime engine backed by llama-server (persistent HTTP).
 *
 * - Server starts once, model loads once, stays in memory
 * - Chat requests go via HTTP POST to localhost
 * - No process spawn per request — fast, session-friendly
 * - Auto-starts server on first generate() if binary exists
 */
class LlamaRuntime(
    private val context: Context,
    private val store: PhonoLabStore,
    private val storageManager: StorageManager? = null,
) {
    val cppManager = LlamaCppManager(context, store, storageManager)

    private var serverProcess: Process? = null
    private var serverPort: Int = 8080
    private var loadedModelPath: String? = null
    private var _isModelLoaded = false
    private var _isServerRunning = false

    // ── Public API ──────────────────────────────────────────────────

    /** Find the llama-server binary. */
    fun findServer(): File? = cppManager.findServer()

    /** Whether the server process is alive. */
    fun isServerRunning(): Boolean {
        if (serverProcess == null) return false
        try {
            val exit = serverProcess!!.exitValue()
            // Process exited — not running
            _isServerRunning = false
            return false
        } catch (_: IllegalThreadStateException) {
            // Still running
            return true
        }
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

        val server = findServer() ?: error("llama-server not found. Run Setup first.")

        // If server is running with the same model, just mark loaded
        if (isServerRunning() && loadedModelPath == model.absolutePath && _isModelLoaded) {
            return
        }

        // Stop existing server if running with different model
        stopServer()

        // Start server with this model
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
     * Auto-starts server if binary exists but server isn't running.
     */
    fun generate(prompt: String, onToken: (String) -> Unit): String {
        // Auto-start if we have a binary and model but server isn't running
        if (!isServerRunning() && loadedModelPath != null) {
            val model = File(loadedModelPath!!)
            val server = findServer()
            if (server != null && model.exists()) {
                startServer(server, model)
            }
        }

        require(isServerRunning()) {
            "llama-server is not running. Load a model first."
        }
        require(_isModelLoaded) {
            "No model loaded. Select and load a model first."
        }
        require(prompt.length <= 18_000) {
            "Prompt too long (${prompt.length} chars). Max: 18,000."
        }

        // Build request JSON (llama.cpp /completion endpoint)
        val requestBody = JSONObject().apply {
            put("prompt", prompt)
            put("n_predict", 384)
            put("temperature", 0.7)
            put("top_p", 0.9)
            put("repeat_penalty", 1.1)
            put("stream", false)  // non-streaming for simplicity
        }

        val url = URL("http://127.0.0.1:$serverPort/completion")
        val conn = url.openConnection() as HttpURLConnection
        conn.requestMethod = "POST"
        conn.setRequestProperty("Content-Type", "application/json")
        conn.connectTimeout = 5_000
        conn.readTimeout = 300_000  // 5 min for long generations
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

            // Stream the full response as one token chunk
            onToken(content)

            return content
        } finally {
            conn.disconnect()
        }
    }

    // ── Auto-install ────────────────────────────────────────────────

    /** Auto-install binary. Returns installed path or null. */
    fun autoInstallBinary(
        progress: (done: Long, total: Long, label: String) -> Unit,
    ): File? = cppManager.downloadAndInstall(progress)

    // ── Server management ───────────────────────────────────────────

    private fun startServer(serverBinary: File, model: File) {
        stopServer()

        val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
        serverPort = findFreePort()

        // LD_LIBRARY_PATH: .so files are in store.runtimeDir + nativeLibraryDir
        val nativeDir = context.applicationInfo.nativeLibraryDir
        val runtimeDir = store.runtimeDir.absolutePath
        val env = HashMap(System.getenv())
        val ldPaths = mutableListOf<String>()
        // runtimeDir has the extracted .so files
        if (File(runtimeDir).exists()) ldPaths.add(runtimeDir)
        // nativeDir has the deployed copies
        if (nativeDir != runtimeDir) ldPaths.add(nativeDir)
        val existing = env["LD_LIBRARY_PATH"] ?: ""
        if (existing.isNotEmpty()) ldPaths.add(existing)
        env["LD_LIBRARY_PATH"] = ldPaths.joinToString(":")

        val command = listOf(
            serverBinary.absolutePath,
            "-m", model.absolutePath,
            "--port", serverPort.toString(),
            "-t", threads.toString(),
            "--ctx-size", "2048",
            "-n", "384",
        )

        val process = ProcessBuilder(command)
            .redirectErrorStream(true)
            .apply { environment().putAll(env) }
            .start()

        serverProcess = process

        // Wait for server to be ready (poll health endpoint)
        val startTime = System.currentTimeMillis()
        val timeout = 60_000L  // 60s for model loading
        var ready = false

        while (System.currentTimeMillis() - startTime < timeout && !ready) {
            try {
                val healthConn = URL("http://127.0.0.1:$serverPort/health").openConnection() as HttpURLConnection
                healthConn.connectTimeout = 2_000
                healthConn.readTimeout = 2_000
                try {
                    val healthStatus = healthConn.responseCode
                    if (healthStatus == 200) {
                        ready = true
                    }
                } finally {
                    healthConn.disconnect()
                }
            } catch (_: Exception) {
                // Server not ready yet, wait
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
    }

    private fun stopServer() {
        serverProcess?.let { proc ->
            try {
                proc.destroy()
                Thread.sleep(200)
                try {
                    proc.exitValue()
                } catch (_: IllegalThreadStateException) {
                    proc.destroyForcibly()
                }
            } catch (_: Exception) { }
        }
        serverProcess = null
        _isServerRunning = false
    }

    /**
     * Kill ALL llama-server and llama-cli processes on the device.
     * Called on app exit to ensure no orphan processes remain.
     */
    fun killAllLlamaProcesses() {
        // First stop our tracked process
        stopServer()

        // Then kill any orphan llama processes system-wide
        val processNames = listOf("llama-server", "llama-cli", "llama_server", "llama_cli")
        for (name in processNames) {
            try {
                // Use pkill to find and kill by process name
                val proc = ProcessBuilder("pkill", "-f", name)
                    .redirectErrorStream(true)
                    .start()
                proc.waitFor()
            } catch (_: Exception) {
                // pkill might not be available, try killall
                try {
                    val proc = ProcessBuilder("killall", name)
                        .redirectErrorStream(true)
                        .start()
                    proc.waitFor()
                } catch (_: Exception) { }
            }
        }

        // Also try killing by finding PIDs via /proc
        try {
            val proc = ProcessBuilder("sh", "-c",
                "for pid in \$(ps -A -o PID,ARGS 2>/dev/null | grep -E 'llama-server|llama-cli' | grep -v grep | awk '{print \$1}'); do kill -9 \$pid 2>/dev/null; done"
            )
                .redirectErrorStream(true)
                .start()
            proc.waitFor()
        } catch (_: Exception) { }

        // Reset state
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
