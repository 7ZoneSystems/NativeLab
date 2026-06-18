package org.nativelab.phonolab

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.nativelab.phonolab.data.ChatSession
import org.nativelab.phonolab.data.ModelConfig
import org.nativelab.phonolab.data.ModelManager
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
    private val modelManager: ModelManager? = null,
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
    @Volatile private var _abort = false

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

    /** The current model config (for API parameter editing). */
    var loadedConfig: ModelConfig? = null
        private set

    /** Update model config parameters without reloading. */
    fun updateConfig(temperature: Float? = null, topP: Float? = null, topK: Int? = null, repeatPenalty: Float? = null, maxTokens: Int? = null, ctx: Int? = null) {
        loadedConfig?.let { cfg ->
            loadedConfig = cfg.copy(
                temperature = temperature ?: cfg.temperature,
                topP = topP ?: cfg.topP,
                topK = topK ?: cfg.topK,
                repeatPenalty = repeatPenalty ?: cfg.repeatPenalty,
                maxTokens = maxTokens ?: cfg.maxTokens,
                ctx = ctx ?: cfg.ctx,
            )
        }
    }

    /** Get runtime status for UI display. */
    fun runtimeStatus(): LlamaCppManager.RuntimeStatus = cppManager.status()

    /**
     * Start llama-server with a model.
     * Looks up ModelConfig from ModelManager for per-model settings.
     * If server is already running with a different model, restarts it.
     *
     * @return null on success, error message on failure.
     */
    fun load(model: File): String? {
        try {
            if (!model.exists()) return "Model file not found: ${model.name}"
            if (!model.extension.equals("gguf", ignoreCase = true)) return "Not a GGUF file: ${model.name}"
            if (model.length() < 1000) return "Model file is too small: ${model.name}"

            val server = findServer()
                ?: return "llama-server not bundled. Run setup_binaries.sh and rebuild APK."

            // If server is running with the same model, just return
            if (isServerRunning() && loadedModelPath == model.absolutePath && _isModelLoaded) {
                return null
            }

            val cfg = modelManager?.get(model.absolutePath)
            loadedConfig = cfg

            stopServer()
            val startError = startServer(server, model, cfg)
            if (startError != null) return startError
            loadedModelPath = model.absolutePath
            return null
        } catch (e: Exception) {
            Log.e(TAG, "load() failed", e)
            cleanup()
            return "Load failed: ${e.message}"
        }
    }

    /** Signal current generation to abort (does NOT kill server). */
    fun abort() {
        _abort = true
    }

    /** Unload model and stop server. */
    fun unload() {
        _abort = true
        try {
            stopServer()
        } catch (_: Exception) {}
        loadedModelPath = null
        loadedConfig = null
        _isModelLoaded = false
    }

    /** Emergency cleanup — reset all state. */
    private fun cleanup() {
        _abort = true
        try { stopServer() } catch (_: Exception) {}
        serverPid = -1
        loadedModelPath = null
        loadedConfig = null
        _isModelLoaded = false
        _isServerRunning = false
    }

    /**
     * Generate text via llama-server /v1/chat/completions (SSE streaming).
     * The server applies the model's chat template — no raw prompt needed.
     *
     * @return generated text, or error message prefixed with "[ERROR]".
     */
    fun generate(prompt: String, onToken: (String) -> Unit): String {
        _abort = false
        try {
            // Auto-start if needed
            if (!isServerRunning() && loadedModelPath != null) {
                val modelPath = loadedModelPath ?: return "[ERROR] No model loaded."
                val model = File(modelPath)
                val server = findServer()
                if (server != null && model.exists()) {
                    val startErr = startServer(server, model, loadedConfig)
                    if (startErr != null) return "[ERROR] $startErr"
                }
            }

            if (!isServerRunning()) return "[ERROR] Server not running. Load a model first."
            if (!_isModelLoaded) return "[ERROR] No model loaded."
            if (prompt.length > 18_000) return "[ERROR] Prompt too long (${prompt.length} chars). Max: 18,000."

            val session = ChatSession.new()
            session.addMessage("user", prompt)
            val cfg = loadedConfig
            val requestBody = session.buildRequestBody(
                systemPrompt = "You are a helpful AI assistant.",
                temperature = cfg?.temperature ?: 0.7f,
                maxTokens = cfg?.maxTokens ?: 384,
            )

            val url = URL("http://127.0.0.1:$serverPort/v1/chat/completions")
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "POST"
            conn.setRequestProperty("Content-Type", "application/json")
            conn.setRequestProperty("Accept", "text/event-stream")
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
                    return "[ERROR] Server returned HTTP $status: ${errBody.take(200)}"
                }

                val result = StringBuilder()
                conn.inputStream.bufferedReader().use { reader ->
                    var line: String?
                    while (reader.readLine().also { line = it } != null) {
                        // Check abort flag — user stopped generation
                        if (_abort) break

                        val l = line ?: continue
                        if (!l.startsWith("data: ")) continue
                        val data = l.removePrefix("data: ").trim()
                        if (data == "[DONE]") break
                        try {
                            val json = JSONObject(data)
                            val choices = json.optJSONArray("choices") ?: continue
                            if (choices.length() == 0) continue
                            val delta = choices.getJSONObject(0).optJSONObject("delta") ?: continue
                            if (delta.isNull("content")) continue
                            val token = delta.optString("content", "")
                            if (token.isNotEmpty()) {
                                result.append(token)
                                onToken(token)
                            }
                        } catch (_: Exception) { }
                    }
                }
                return if (_abort && result.isEmpty()) "[ABORTED]" else result.toString()
            } finally {
                conn.disconnect()
            }
        } catch (e: java.io.EOFException) {
            // Server was killed while reading — not a real error if user aborted
            Log.w(TAG, "Stream closed (EOF)", e)
            return if (_abort) "[ABORTED]" else "[ERROR] Server closed connection unexpectedly"
        } catch (e: Exception) {
            Log.e(TAG, "generate() failed", e)
            return if (_abort) "[ABORTED]" else "[ERROR] ${e.message}"
        }
    }

    // ── Auto-install (legacy fallback) ──────────────────────────────

    fun autoInstallBinary(
        progress: (done: Long, total: Long, label: String) -> Unit,
    ): File? = cppManager.downloadAndInstall(progress)

    // ── Server management ───────────────────────────────────────────

    /**
     * Start llama-server.
     * @return null on success, error message on failure.
     */
    private fun startServer(serverBinary: File, model: File, cfg: ModelConfig? = null): String? {
        stopServer()

        val defaultThreads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
        val threads = cfg?.threads ?: defaultThreads
        val ctxSize = cfg?.ctx ?: 2048
        val nPredict = cfg?.maxTokens ?: 384
        serverPort = findFreePort()

        val visionInfo = detectVisionModel(model.name)
        val mmproj = if (visionInfo.isVision) detectMmprojForModel(model).ifEmpty { null } else null

        Log.d(TAG, "Starting server: ${serverBinary.absolutePath}")
        Log.d(TAG, "Model: ${model.absolutePath} (${model.length()} bytes)")
        Log.d(TAG, "Port: $serverPort, Threads: $threads, Ctx: $ctxSize, nPredict: $nPredict")
        if (mmproj != null) Log.d(TAG, "Vision model detected, mmproj: $mmproj")
        cfg?.let {
            Log.d(TAG, "Config: temp=${it.temperature}, topP=${it.topP}, topK=${it.topK}, rep=${it.repeatPenalty}")
        }

        val pid = cppManager.startServer(
            serverBin = serverBinary,
            modelPath = model,
            port = serverPort,
            threads = threads,
            ctxSize = ctxSize,
            nPredict = nPredict,
            mmproj = mmproj,
        )

        if (pid <= 0) {
            cleanup()
            return "Failed to start llama-server process. The binary may be missing or incompatible."
        }

        serverPid = pid

        // Wait for server to be ready
        val startTime = System.currentTimeMillis()
        val timeout = 60_000L
        var ready = false

        while (System.currentTimeMillis() - startTime < timeout && !ready) {
            val exitCode = cppManager.checkProcess(pid)
            if (exitCode > 0) {
                cleanup()
                return "llama-server exited immediately with code $exitCode. The model may be corrupted or too large."
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
            cleanup()
            return "llama-server failed to start within 60s. Model may be too large for device memory."
        }

        _isServerRunning = true
        _isModelLoaded = true
        loadedModelPath = model.absolutePath
        Log.d(TAG, "Server ready on port $serverPort")
        return null
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
