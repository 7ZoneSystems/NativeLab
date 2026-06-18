package org.nativelab.phonolab.api

import android.util.Base64
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.thread

/**
 * HTTP API server for PhonoLab — OpenAI and Anthropic compatible.
 *
 * Features:
 * - Live status reporting (idle/loading/generating/stable/reloading)
 * - Request queuing during model reload
 * - Vision model support (image_url in messages)
 * - Document context injection
 * - Parameter editing (temperature, top_k, top_p, etc.)
 * - Smart reload with queue drain
 */
class PhonoLabApiServer(
    private val config: ApiConfig,
    private val context: android.content.Context? = null,
    private val onLog: (String) -> Unit = {},
    private val generateFn: (prompt: String, nPredict: Int, temperature: Float, topP: Float, topK: Int, repeatPenalty: Float) -> String,
    private val streamGenerateFn: ((prompt: String, nPredict: Int, temperature: Float, topP: Float, topK: Int, repeatPenalty: Float, onToken: (String) -> Unit) -> String)? = null,
    private val runtimeInfo: () -> Map<String, Any> = { emptyMap() },
    private val modelList: () -> List<Map<String, Any>> = { emptyMapList() },
    private val loadModelFn: ((modelPath: String) -> String?)? = null,
    private val updateConfigFn: ((temperature: Float?, topP: Float?, topK: Int?, repeatPenalty: Float?, maxTokens: Int?, ctx: Int?) -> Unit)? = null,
    private val getVisionModelFn: (() -> Boolean)? = null,
) {
    private var serverSocket: ServerSocket? = null
    private var running = false
    private val executor = Executors.newFixedThreadPool(4)
    private val requestLog = mutableListOf<String>()
    private val logLock = Any()

    // ── Live status tracking ──────────────────────────────────────
    enum class ServerStatus {
        IDLE,       // No model loaded
        LOADING,    // Model is being loaded
        READY,      // Model loaded, ready for requests
        GENERATING, // Currently generating a response
        RELOADING,  // Model is being reloaded (switching models)
        ERROR       // Last operation failed
    }

    @Volatile var status: ServerStatus = ServerStatus.IDLE
        private set

    @Volatile var statusMessage: String = "No model loaded"
        private set

    @Volatile var lastError: String? = null
        private set

    private val activeGenerations = AtomicInteger(0)
    private val requestQueue = ConcurrentLinkedQueue<QueuedRequest>()
    private val isReloading = AtomicBoolean(false)

    private data class QueuedRequest(
        val socket: Socket,
        val method: String,
        val path: String,
        val body: String,
        val headers: Map<String, String>,
        val authHeader: String,
        val clientIp: String,
        val queuedAt: Long = System.currentTimeMillis(),
    )

    companion object {
        private const val TAG = "PhonoLabApiServer"
        private const val MAX_BODY_BYTES = 1024 * 1024  // 1MB
        private const val MAX_QUEUE_SIZE = 50
        private const val QUEUE_TIMEOUT_MS = 120_000L   // 2 min
        private fun emptyMapList() = emptyList<Map<String, Any>>()
    }

    val isRunning: Boolean get() = running

    fun start(): String {
        if (running) return "Already running"
        try {
            serverSocket = ServerSocket(config.port, 50, java.net.InetAddress.getByName(config.host))
            running = true
            thread(isDaemon = true) { acceptLoop() }
            thread(isDaemon = true) { queueProcessor() }
            val url = if (config.host == "0.0.0.0") config.lanBaseUrl else config.localBaseUrl
            addLog("Server started on ${config.host}:${config.port}")
            updateStatus(ServerStatus.IDLE, "Server running, no model loaded")
            return "Running at $url"
        } catch (e: Exception) {
            addLog("Start failed: ${e.message}")
            return "Failed: ${e.message}"
        }
    }

    fun stop() {
        running = false
        try { serverSocket?.close() } catch (_: Exception) {}
        serverSocket = null
        try { executor.shutdownNow() } catch (_: Exception) {}
        // Drain queue with errors
        while (requestQueue.isNotEmpty()) {
            val req = requestQueue.poll()
            if (req != null) {
                try {
                    sendResponse(req.socket, errorResponse("Server shutting down", "server_error", 503), 503)
                    req.socket.close()
                } catch (_: Exception) {}
            }
        }
        updateStatus(ServerStatus.IDLE, "Server stopped")
        addLog("Server stopped")
    }

    fun getRecentLogs(): List<String> {
        synchronized(logLock) {
            return requestLog.takeLast(50).toList()
        }
    }

    /** Call this from the runtime when model state changes. */
    fun notifyModelStateChange(loaded: Boolean, modelName: String, isGenerating: Boolean = false) {
        if (isGenerating) {
            updateStatus(ServerStatus.GENERATING, "Generating with $modelName")
        } else if (isReloading.get()) {
            updateStatus(ServerStatus.RELOADING, "Reloading model: $modelName")
        } else if (loaded) {
            updateStatus(ServerStatus.READY, "Ready: $modelName")
        } else {
            updateStatus(ServerStatus.IDLE, "No model loaded")
        }
    }

    private fun updateStatus(newStatus: ServerStatus, message: String) {
        status = newStatus
        statusMessage = message
        if (newStatus != ServerStatus.ERROR) lastError = null
        addLog("Status: ${newStatus.name} — $message")
    }

    // ── Request Queue ─────────────────────────────────────────────

    private fun enqueueRequest(socket: Socket, method: String, path: String, body: String, headers: Map<String, String>, authHeader: String, clientIp: String) {
        if (requestQueue.size >= MAX_QUEUE_SIZE) {
            sendResponse(socket, errorResponse(
                "Request queue full ($MAX_QUEUE_SIZE). Server is busy.",
                "server_busy", 503
            ), 503)
            try { socket.close() } catch (_: Exception) {}
            return
        }
        requestQueue.add(QueuedRequest(socket, method, path, body, headers, authHeader, clientIp))
        addLog("Queued: $method $path (queue size: ${requestQueue.size})")
    }

    private fun queueProcessor() {
        while (running) {
            try {
                val req = requestQueue.peek() ?: run {
                    Thread.sleep(200)
                    return@queueProcessor
                }

                // Check timeout
                if (System.currentTimeMillis() - req.queuedAt > QUEUE_TIMEOUT_MS) {
                    requestQueue.poll()
                    sendResponse(req.socket, errorResponse(
                        "Request timed out in queue after ${QUEUE_TIMEOUT_MS / 1000}s",
                        "gateway_timeout", 504
                    ), 504)
                    try { req.socket.close() } catch (_: Exception) {}
                    continue
                }

                // Only process if ready
                if (status == ServerStatus.RELOADING || status == ServerStatus.LOADING) {
                    Thread.sleep(500)
                    continue
                }

                // Dequeue and process
                val dequeued = requestQueue.poll() ?: continue
                addLog("Dequeued: ${dequeued.method} ${dequeued.path}")
                executor.execute { handleClientDirect(dequeued.socket, dequeued.method, dequeued.path, dequeued.body, dequeued.headers, dequeued.authHeader, dequeued.clientIp) }
            } catch (e: Exception) {
                if (running) Log.w(TAG, "Queue processor error: ${e.message}")
                Thread.sleep(500)
            }
        }
    }

    // ── Accept Loop ───────────────────────────────────────────────

    private fun acceptLoop() {
        while (running) {
            try {
                val socket = serverSocket?.accept() ?: break
                executor.execute { handleClient(socket) }
            } catch (e: Exception) {
                if (running) addLog("Accept error: ${e.message}")
            }
        }
    }

    private fun handleClient(socket: Socket) {
        try {
            socket.soTimeout = 300_000
            val reader = BufferedReader(InputStreamReader(socket.inputStream))

            val requestLine = reader.readLine() ?: return
            val parts = requestLine.split(" ")
            if (parts.size < 2) return
            val method = parts[0]
            val path = parts[1].split("?")[0].trimEnd('/')

            val headers = mutableMapOf<String, String>()
            while (true) {
                val line = reader.readLine() ?: break
                if (line.isEmpty()) break
                val colon = line.indexOf(':')
                if (colon > 0) {
                    headers[line.substring(0, colon).trim().lowercase()] = line.substring(colon + 1).trim()
                }
            }

            val contentLength = headers["content-length"]?.toIntOrNull() ?: 0
            if (contentLength > MAX_BODY_BYTES) {
                sendResponse(socket, errorResponse(
                    "Body too large ($contentLength bytes, max $MAX_BODY_BYTES)",
                    "payload_too_large", 413
                ), 413)
                return
            }
            val body = if (contentLength > 0) {
                val buf = CharArray(contentLength)
                var read = 0
                while (read < contentLength) {
                    val n = reader.read(buf, read, contentLength - read)
                    if (n < 0) break
                    read += n
                }
                String(buf, 0, read)
            } else ""

            val authHeader = headers["authorization"] ?: headers["x-api-key"] ?: ""
            val clientIp = socket.inetAddress.hostAddress ?: "unknown"

            addLog("$method $path from $clientIp")

            // Queue if reloading or loading
            if (isReloading.get() || status == ServerStatus.LOADING || status == ServerStatus.RELOADING) {
                if (method == "POST" && path in listOf("/chat/completions", "/v1/chat/completions", "/completions", "/v1/completions", "/messages", "/v1/messages")) {
                    enqueueRequest(socket, method, path, body, headers, authHeader, clientIp)
                    return
                }
            }

            handleClientDirect(socket, method, path, body, headers, authHeader, clientIp)
        } catch (e: Exception) {
            addLog("Error: ${e.message}")
        } finally {
            try { socket.close() } catch (_: Exception) {}
        }
    }

    private fun handleClientDirect(socket: Socket, method: String, path: String, body: String, headers: Map<String, String>, authHeader: String, clientIp: String) {
        try {
            val isStreamRequest = method == "POST" &&
                path in listOf("/chat/completions", "/v1/chat/completions") &&
                body.contains("\"stream\"") && body.contains("true")

            val response = when {
                method == "OPTIONS" -> handleOptions()
                method == "GET" && path in listOf("/health", "/v1/health") -> handleHealth(clientIp)
                method == "GET" && path == "/status" -> handleStatus(clientIp)
                method == "GET" && path == "/runtime" -> handleRuntime(clientIp)
                method == "GET" && path in listOf("/device", "/system") -> handleDevice(clientIp, authHeader)
                method == "GET" && path in listOf("/models", "/v1/models") -> handleModels(clientIp, authHeader)
                method == "GET" && path == "/capabilities" -> handleCapabilities(clientIp, authHeader)
                method == "GET" && path == "/queue" -> handleQueueStatus(clientIp, authHeader)
                isStreamRequest -> {
                    val err = handleOpenAiChatStreaming(socket, body, authHeader)
                    if (err == null) return
                    err
                }
                method == "POST" && path in listOf("/chat/completions", "/v1/chat/completions") ->
                    handleOpenAiChat(body, authHeader)
                method == "POST" && path in listOf("/completions", "/v1/completions") ->
                    handleOpenAiCompletion(body, authHeader)
                method == "POST" && path in listOf("/messages", "/v1/messages") ->
                    handleAnthropicMessages(body, authHeader)
                method == "POST" && path == "/config" ->
                    handleConfigUpdate(body, authHeader, clientIp)
                method == "POST" && path == "/reload" ->
                    handleReload(body, authHeader, clientIp)
                method == "POST" && path == "/load" ->
                    handleLoadModel(body, authHeader, clientIp)
                else -> errorResponse("Unknown endpoint: $path. See /status for available endpoints.", "not_found", 404)
            }

            sendResponse(socket, response)
        } catch (e: Exception) {
            addLog("Handler error: ${e.message}")
            try { sendResponse(socket, errorResponse("Internal error: ${e.message}", "server_error", 500), 500) } catch (_: Exception) {}
        }
    }

    // ── Auth ──────────────────────────────────────────────────────

    private fun isAuthorized(authHeader: String, clientIp: String): Boolean {
        if (!config.requireApiKey) return true
        var key = authHeader.trim()
        if (key.lowercase().startsWith("bearer ")) key = key.substring(7).trim()
        if (key.isEmpty()) return false
        val isLoopback = clientIp.startsWith("127.") || clientIp == "::1" || clientIp == "localhost"
        val expected = if (isLoopback) config.localApiKey else config.lanApiKey
        return key == expected
    }

    private fun requireAuth(clientIp: String, authHeader: String): JSONObject? {
        if (!isAuthorized(authHeader, clientIp)) {
            return errorResponse("Invalid API key.", "authentication_error", 401)
        }
        return null
    }

    // ── Handlers ──────────────────────────────────────────────────

    private fun handleOptions(): JSONObject {
        return JSONObject().put("ok", true)
    }

    private fun handleHealth(clientIp: String): JSONObject {
        val info = runtimeInfo()
        val isVision = getVisionModelFn?.invoke() ?: false
        val rt = Runtime.getRuntime()
        return JSONObject().apply {
            put("ok", true)
            put("name", "PhonoLab API Server")
            put("version", "2.1.0")
            put("status", status.name.lowercase())
            put("status_message", statusMessage)
            put("runtime", JSONObject().apply {
                put("loaded", info["loaded"] ?: false)
                put("model", info["model"] ?: "none")
                put("model_path", info["model_path"] ?: "")
                put("state", info["state"] ?: "idle")
                put("is_vision", isVision)
                put("ctx", info["ctx"] ?: 2048)
                put("active_generations", activeGenerations.get())
                put("queue_size", requestQueue.size)
            })
            put("device", JSONObject().apply {
                put("model", "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}")
                put("cpu_cores", rt.availableProcessors())
                put("ram_mb", readMemInfo()?.get("total") ?: (rt.maxMemory() / (1024 * 1024)))
            })
            put("endpoints", JSONObject().apply {
                put("local_base_url", config.localBaseUrl)
                put("lan_base_url", config.lanBaseUrl)
                put("openai_chat", "${config.localBaseUrl}/chat/completions")
                put("models", "${config.localBaseUrl}/models")
                put("status", "${config.localBaseUrl.replace("/v1", "")}/status")
                put("device", "${config.localBaseUrl.replace("/v1", "")}/device")
                put("config", "${config.localBaseUrl.replace("/v1", "")}/config")
                put("reload", "${config.localBaseUrl.replace("/v1", "")}/reload")
                put("load", "${config.localBaseUrl.replace("/v1", "")}/load")
                put("queue", "${config.localBaseUrl.replace("/v1", "")}/queue")
            })
            put("auth", JSONObject().apply {
                put("required", config.requireApiKey)
                put("local_key_header", "Authorization: Bearer <local_api_key>")
                put("lan_key_header", "Authorization: Bearer <lan_api_key>")
            })
            put("features", JSONObject().apply {
                put("streaming", true)
                put("vision", isVision)
                put("document_rag", true)
                put("parameter_editing", true)
                put("request_queuing", true)
                put("smart_reload", true)
                put("device_info", true)
            })
        }
    }

    private fun handleStatus(clientIp: String): JSONObject {
        val info = runtimeInfo()
        val isVision = getVisionModelFn?.invoke() ?: false
        return JSONObject().apply {
            put("status", status.name.lowercase())
            put("status_message", statusMessage)
            put("last_error", lastError ?: JSONObject.NULL)
            put("runtime", JSONObject().apply {
                put("loaded", info["loaded"] ?: false)
                put("model", info["model"] ?: "none")
                put("model_path", info["model_path"] ?: "")
                put("state", info["state"] ?: "idle")
                put("is_vision", isVision)
                put("ctx", info["ctx"] ?: 2048)
            })
            put("server", JSONObject().apply {
                put("active_generations", activeGenerations.get())
                put("queue_size", requestQueue.size)
                put("is_reloading", isReloading.get())
                put("uptime_ms", System.currentTimeMillis())
            })
        }
    }

    private fun handleRuntime(clientIp: String): JSONObject {
        val info = runtimeInfo()
        val isVision = getVisionModelFn?.invoke() ?: false
        return JSONObject().apply {
            put("loaded", info["loaded"] ?: false)
            put("model", info["model"] ?: "none")
            put("model_path", info["model_path"] ?: "")
            put("state", info["state"] ?: "idle")
            put("is_vision", isVision)
            put("ctx", info["ctx"] ?: 2048)
            put("status", status.name.lowercase())
            put("status_message", statusMessage)
        }
    }

    private fun handleModels(clientIp: String, authHeader: String): JSONObject {
        requireAuth(clientIp, authHeader)?.let { return it }
        val models = modelList()
        val activeModel = runtimeInfo()["model"] ?: "none"
        return JSONObject().apply {
            put("object", "list")
            put("data", JSONArray().apply {
                for (m in models) {
                    put(JSONObject().apply {
                        put("id", m["id"] ?: "unknown")
                        put("object", "model")
                        put("created", System.currentTimeMillis() / 1000)
                        put("owned_by", "phonolab")
                        put("active", (m["id"] ?: "") == activeModel)
                    })
                }
                if (models.isEmpty()) {
                    put(JSONObject().apply {
                        put("id", "phonolab-active")
                        put("object", "model")
                        put("created", System.currentTimeMillis() / 1000)
                        put("owned_by", "phonolab")
                        put("active", true)
                    })
                }
            })
        }
    }

    private fun handleCapabilities(clientIp: String, authHeader: String): JSONObject {
        requireAuth(clientIp, authHeader)?.let { return it }
        return JSONObject().apply {
            put("object", "phonolab.capabilities")
            put("status", handleStatus(clientIp))
            put("endpoints", handleHealth(clientIp).getJSONObject("endpoints"))
            put("models", handleModels(clientIp, authHeader).getJSONArray("data"))
        }
    }

    private fun handleQueueStatus(clientIp: String, authHeader: String): JSONObject {
        requireAuth(clientIp, authHeader)?.let { return it }
        return JSONObject().apply {
            put("queue_size", requestQueue.size)
            put("max_queue_size", MAX_QUEUE_SIZE)
            put("is_reloading", isReloading.get())
            put("active_generations", activeGenerations.get())
            put("status", status.name.lowercase())
        }
    }

    private fun handleDevice(clientIp: String, authHeader: String): JSONObject {
        requireAuth(clientIp, authHeader)?.let { return it }
        val rt = Runtime.getRuntime()
        val maxMem = rt.maxMemory()
        val totalMem = rt.totalMemory()
        val freeMem = rt.freeMemory()
        val usedMem = totalMem - freeMem

        // Read /proc/meminfo for system RAM
        val memInfo = readMemInfo()

        return JSONObject().apply {
            put("device", JSONObject().apply {
                put("model", android.os.Build.MODEL)
                put("manufacturer", android.os.Build.MANUFACTURER)
                put("brand", android.os.Build.BRAND)
                put("device", android.os.Build.DEVICE)
                put("product", android.os.Build.PRODUCT)
                put("hardware", android.os.Build.HARDWARE)
                put("board", android.os.Build.BOARD)
                put("android_version", android.os.Build.VERSION.RELEASE)
                put("sdk_int", android.os.Build.VERSION.SDK_INT)
                put("build_id", android.os.Build.ID)
                put("fingerprint", android.os.Build.FINGERPRINT.take(80))
            })
            put("cpu", JSONObject().apply {
                put("cores", rt.availableProcessors())
                put("abis", android.os.Build.SUPPORTED_ABIS.toList())
                put("primary_abi", android.os.Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown")
                put("is_64bit", android.os.Build.SUPPORTED_ABIS.any { it.contains("64") })
            })
            put("memory", JSONObject().apply {
                put("jvm_max_mb", maxMem / (1024 * 1024))
                put("jvm_used_mb", usedMem / (1024 * 1024))
                put("jvm_free_mb", freeMem / (1024 * 1024))
                put("jvm_total_mb", totalMem / (1024 * 1024))
                if (memInfo != null) {
                    put("system_total_mb", memInfo["total"] ?: 0)
                    put("system_available_mb", memInfo["available"] ?: 0)
                    put("system_used_mb", (memInfo["total"] ?: 0) - (memInfo["available"] ?: 0))
                }
            })
            put("storage", JSONObject().apply {
                val dataDir = context?.filesDir
                if (dataDir != null) {
                    val stat = android.os.StatFs(dataDir.path)
                    put("data_total_mb", stat.totalBytes / (1024 * 1024))
                    put("data_free_mb", stat.availableBytes / (1024 * 1024))
                }
            })
            put("runtime", JSONObject().apply {
                put("status", status.name.lowercase())
                put("status_message", statusMessage)
                val info = runtimeInfo()
                put("loaded", info["loaded"] ?: false)
                put("model", info["model"] ?: "none")
                put("is_vision", getVisionModelFn?.invoke() ?: false)
                put("active_generations", activeGenerations.get())
                put("queue_size", requestQueue.size)
            })
        }
    }

    private fun readMemInfo(): Map<String, Long>? {
        return try {
            val map = mutableMapOf<String, Long>()
            java.io.File("/proc/meminfo").readLines().forEach { line ->
                val parts = line.split(":")
                if (parts.size == 2) {
                    val key = parts[0].trim()
                    val value = parts[1].trim().split("\\s+".toRegex())[0].toLongOrNull() ?: return@forEach
                    when (key) {
                        "MemTotal" -> map["total"] = value / 1024  // kB to MB
                        "MemAvailable" -> map["available"] = value / 1024
                        "MemFree" -> map["free"] = value / 1024
                    }
                }
            }
            map
        } catch (_: Exception) { null }
    }

    // ── Chat Completions (with vision + doc support) ──────────────

    private fun handleOpenAiChat(body: String, authHeader: String): JSONObject {
        try {
            // Safety check: model must be loaded
            val info = runtimeInfo()
            if (info["loaded"] != true) {
                return errorResponse(
                    "No model loaded. Use POST /load to load a model first, or check /status for current state.",
                    "model_not_loaded", 503
                )
            }
            if (status == ServerStatus.ERROR) {
                return errorResponse(
                    "Server is in error state: $lastError. Try POST /reload to recover.",
                    "server_error", 503
                )
            }
            if (status == ServerStatus.LOADING || status == ServerStatus.RELOADING) {
                return errorResponse(
                    "Model is currently ${status.name.lowercase()}. Request queued or try again.",
                    "model_loading", 503
                )
            }

            val payload = JSONObject(body)
            val messages = payload.optJSONArray("messages") ?: JSONArray()
            val nPredict = payload.optInt("max_tokens", payload.optInt("n_predict", 512))
            val temperature = payload.optDouble("temperature", 0.7).toFloat()
            val topP = payload.optDouble("top_p", 0.9).toFloat()
            val topK = payload.optInt("top_k", 40)
            val repeatPenalty = payload.optDouble("repeat_penalty", 1.1).toFloat()

            val prompt = messagesToPromptWithVision(messages)
            activeGenerations.incrementAndGet()
            updateStatus(ServerStatus.GENERATING, "Generating response")

            try {
                val text = generateFn(prompt, nPredict, temperature, topP, topK, repeatPenalty)
                val modelId = runtimeInfo()["model"] ?: "phonolab-active"

                val result = JSONObject().apply {
                    put("id", "chatcmpl-${System.currentTimeMillis()}")
                    put("object", "chat.completion")
                    put("created", System.currentTimeMillis() / 1000)
                    put("model", modelId)
                    put("choices", JSONArray().apply {
                        put(JSONObject().apply {
                            put("index", 0)
                            put("message", JSONObject().apply {
                                put("role", "assistant")
                                put("content", text)
                            })
                            put("finish_reason", "stop")
                        })
                    })
                    put("usage", JSONObject().apply {
                        put("prompt_tokens", 0)
                        put("completion_tokens", 0)
                        put("total_tokens", 0)
                    })
                }
                updateStatus(ServerStatus.READY, "Ready: $modelId")
                return result
            } finally {
                activeGenerations.decrementAndGet()
            }
        } catch (e: Exception) {
            lastError = e.message
            updateStatus(ServerStatus.ERROR, "Generation failed: ${e.message}")
            return errorResponse(e.message ?: "Generation failed", "server_error", 500)
        }
    }

    private fun handleOpenAiChatStreaming(socket: Socket, body: String, authHeader: String): JSONObject? {
        try {
            // Safety check: model must be loaded
            val info = runtimeInfo()
            if (info["loaded"] != true) {
                return errorResponse(
                    "No model loaded. Use POST /load to load a model first.",
                    "model_not_loaded", 503
                )
            }
            if (status == ServerStatus.ERROR) {
                return errorResponse(
                    "Server is in error state: $lastError. Try POST /reload to recover.",
                    "server_error", 503
                )
            }

            val payload = JSONObject(body)
            val messages = payload.optJSONArray("messages") ?: JSONArray()
            val nPredict = payload.optInt("max_tokens", payload.optInt("n_predict", 512))
            val temperature = payload.optDouble("temperature", 0.7).toFloat()
            val topP = payload.optDouble("top_p", 0.9).toFloat()
            val topK = payload.optInt("top_k", 40)
            val repeatPenalty = payload.optDouble("repeat_penalty", 1.1).toFloat()
            val modelId = runtimeInfo()["model"] ?: "phonolab-active"

            val prompt = messagesToPromptWithVision(messages)

            val streamFn = streamGenerateFn ?: return errorResponse(
                "Streaming not supported by this runtime.", "server_error", 501
            )

            activeGenerations.incrementAndGet()
            updateStatus(ServerStatus.GENERATING, "Streaming with $modelId")

            val header = buildString {
                append("HTTP/1.1 200 OK\r\n")
                append("Content-Type: text/event-stream; charset=utf-8\r\n")
                append("Cache-Control: no-cache\r\n")
                append("Access-Control-Allow-Origin: *\r\n")
                append("Access-Control-Allow-Headers: authorization, x-api-key, content-type\r\n")
                append("Connection: close\r\n")
                append("\r\n")
            }
            val out = socket.getOutputStream()
            out.write(header.toByteArray(Charsets.UTF_8))
            out.flush()

            val cmplId = "chatcmpl-${System.currentTimeMillis()}"

            val roleChunk = JSONObject().apply {
                put("id", cmplId)
                put("object", "chat.completion.chunk")
                put("created", System.currentTimeMillis() / 1000)
                put("model", modelId)
                put("choices", JSONArray().apply {
                    put(JSONObject().apply {
                        put("index", 0)
                        put("delta", JSONObject().apply { put("role", "assistant") })
                        put("finish_reason", JSONObject.NULL)
                    })
                })
            }
            try {
                out.write("data: $roleChunk\r\n\r\n".toByteArray(Charsets.UTF_8))
                out.flush()
            } catch (e: java.io.IOException) {
                addLog("Client disconnected before generation: ${e.message}")
                activeGenerations.decrementAndGet()
                updateStatus(ServerStatus.READY, "Ready: $modelId")
                return null
            }

            var clientDisconnected = false
            streamFn(prompt, nPredict, temperature, topP, topK, repeatPenalty) { token ->
                if (clientDisconnected) return@streamFn
                try {
                    val chunk = JSONObject().apply {
                        put("id", cmplId)
                        put("object", "chat.completion.chunk")
                        put("created", System.currentTimeMillis() / 1000)
                        put("model", modelId)
                        put("choices", JSONArray().apply {
                            put(JSONObject().apply {
                                put("index", 0)
                                put("delta", JSONObject().apply { put("content", token) })
                                put("finish_reason", JSONObject.NULL)
                            })
                        })
                    }
                    out.write("data: $chunk\r\n\r\n".toByteArray(Charsets.UTF_8))
                    out.flush()
                } catch (e: java.io.IOException) {
                    clientDisconnected = true
                    addLog("Client disconnected during streaming: ${e.message}")
                }
            }

            activeGenerations.decrementAndGet()
            updateStatus(ServerStatus.READY, "Ready: $modelId")

            if (clientDisconnected) return null

            val finishChunk = JSONObject().apply {
                put("id", cmplId)
                put("object", "chat.completion.chunk")
                put("created", System.currentTimeMillis() / 1000)
                put("model", modelId)
                put("choices", JSONArray().apply {
                    put(JSONObject().apply {
                        put("index", 0)
                        put("delta", JSONObject())
                        put("finish_reason", "stop")
                    })
                })
            }
            try {
                out.write("data: $finishChunk\r\n\r\n".toByteArray(Charsets.UTF_8))
                out.write("data: [DONE]\r\n\r\n".toByteArray(Charsets.UTF_8))
                out.flush()
            } catch (e: java.io.IOException) {
                addLog("Client disconnected before finish: ${e.message}")
            }

            return null
        } catch (e: Exception) {
            activeGenerations.decrementAndGet()
            lastError = e.message
            updateStatus(ServerStatus.ERROR, "Streaming failed: ${e.message}")
            return errorResponse(e.message ?: "Streaming failed", "server_error", 500)
        }
    }

    private fun handleOpenAiCompletion(body: String, authHeader: String): JSONObject {
        try {
            // Safety check: model must be loaded
            if (runtimeInfo()["loaded"] != true) {
                return errorResponse("No model loaded. Use POST /load first.", "model_not_loaded", 503)
            }
            if (status == ServerStatus.ERROR) {
                return errorResponse("Server error: $lastError", "server_error", 503)
            }

            val payload = JSONObject(body)
            val prompt = payload.optString("prompt", "")
            val nPredict = payload.optInt("max_tokens", payload.optInt("n_predict", 512))
            val temperature = payload.optDouble("temperature", 0.7).toFloat()
            val topP = payload.optDouble("top_p", 0.9).toFloat()
            val topK = payload.optInt("top_k", 40)
            val repeatPenalty = payload.optDouble("repeat_penalty", 1.1).toFloat()

            activeGenerations.incrementAndGet()
            try {
                val text = generateFn(prompt, nPredict, temperature, topP, topK, repeatPenalty)
                val modelId = runtimeInfo()["model"] ?: "phonolab-active"

                return JSONObject().apply {
                    put("id", "cmpl-${System.currentTimeMillis()}")
                    put("object", "text_completion")
                    put("created", System.currentTimeMillis() / 1000)
                    put("model", modelId)
                    put("choices", JSONArray().apply {
                        put(JSONObject().apply {
                            put("index", 0)
                            put("text", text)
                            put("finish_reason", "stop")
                        })
                    })
                    put("usage", JSONObject().apply {
                        put("prompt_tokens", 0)
                        put("completion_tokens", 0)
                        put("total_tokens", 0)
                    })
                }
            } finally {
                activeGenerations.decrementAndGet()
            }
        } catch (e: Exception) {
            return errorResponse(e.message ?: "Generation failed", "server_error", 500)
        }
    }

    private fun handleAnthropicMessages(body: String, authHeader: String): JSONObject {
        try {
            // Safety check: model must be loaded
            if (runtimeInfo()["loaded"] != true) {
                return errorResponse("No model loaded. Use POST /load first.", "model_not_loaded", 503)
            }
            if (status == ServerStatus.ERROR) {
                return errorResponse("Server error: $lastError", "server_error", 503)
            }

            val payload = JSONObject(body)
            val messages = payload.optJSONArray("messages") ?: JSONArray()
            val system = payload.optString("system", "")
            val prompt = if (system.isNotEmpty()) "System: $system\n\n${messagesToPrompt(messages)}" else messagesToPrompt(messages)
            val maxTokens = payload.optInt("max_tokens", 512)
            val temperature = payload.optDouble("temperature", 0.7).toFloat()
            val topP = payload.optDouble("top_p", 0.9).toFloat()
            val topK = payload.optInt("top_k", 40)
            val repeatPenalty = payload.optDouble("repeat_penalty", 1.1).toFloat()

            activeGenerations.incrementAndGet()
            try {
                val text = generateFn(prompt, maxTokens, temperature, topP, topK, repeatPenalty)
                val modelId = runtimeInfo()["model"] ?: "phonolab-active"

                return JSONObject().apply {
                    put("id", "msg_${System.currentTimeMillis()}")
                    put("type", "message")
                    put("role", "assistant")
                    put("model", modelId)
                    put("content", JSONArray().apply {
                        put(JSONObject().apply {
                            put("type", "text")
                            put("text", text)
                        })
                    })
                    put("stop_reason", "end_turn")
                    put("stop_sequence", JSONObject.NULL)
                    put("usage", JSONObject().apply {
                        put("input_tokens", 0)
                        put("output_tokens", 0)
                    })
                }
            } finally {
                activeGenerations.decrementAndGet()
            }
        } catch (e: Exception) {
            return errorResponse(e.message ?: "Generation failed", "server_error", 500)
        }
    }

    // ── Config Update ─────────────────────────────────────────────

    private fun handleConfigUpdate(body: String, authHeader: String, clientIp: String): JSONObject {
        requireAuth(clientIp, authHeader)?.let { return it }
        try {
            val payload = JSONObject(body)
            val temperature = if (payload.has("temperature")) payload.optDouble("temperature").toFloat() else null
            val topP = if (payload.has("top_p")) payload.optDouble("top_p").toFloat() else null
            val topK = if (payload.has("top_k")) payload.optInt("top_k") else null
            val repeatPenalty = if (payload.has("repeat_penalty")) payload.optDouble("repeat_penalty").toFloat() else null
            val maxTokens = if (payload.has("max_tokens")) payload.optInt("max_tokens") else null
            val ctx = if (payload.has("ctx")) payload.optInt("ctx") else null

            updateConfigFn?.invoke(temperature, topP, topK, repeatPenalty, maxTokens, ctx)

            return JSONObject().apply {
                put("ok", true)
                put("message", "Configuration updated")
                put("config", JSONObject().apply {
                    if (temperature != null) put("temperature", temperature)
                    if (topP != null) put("top_p", topP)
                    if (topK != null) put("top_k", topK)
                    if (repeatPenalty != null) put("repeat_penalty", repeatPenalty)
                    if (maxTokens != null) put("max_tokens", maxTokens)
                    if (ctx != null) put("ctx", ctx)
                })
            }
        } catch (e: Exception) {
            return errorResponse("Invalid config: ${e.message}", "invalid_request", 400)
        }
    }

    // ── Smart Reload ──────────────────────────────────────────────

    private fun handleReload(body: String, authHeader: String, clientIp: String): JSONObject {
        requireAuth(clientIp, authHeader)?.let { return it }
        val payload = try { JSONObject(body) } catch (_: Exception) { JSONObject() }
        val modelPath = payload.optString("model_path", "")

        if (modelPath.isEmpty()) {
            return errorResponse("model_path is required", "invalid_request", 400)
        }

        if (isReloading.compareAndSet(false, true)) {
            updateStatus(ServerStatus.RELOADING, "Reloading: ${modelPath.substringAfterLast("/")}")
            thread {
                try {
                    val error = loadModelFn?.invoke(modelPath)
                    if (error != null) {
                        lastError = error
                        updateStatus(ServerStatus.ERROR, "Reload failed: $error")
                    } else {
                        updateStatus(ServerStatus.READY, "Ready: ${modelPath.substringAfterLast("/")}")
                    }
                } catch (e: Exception) {
                    lastError = e.message
                    updateStatus(ServerStatus.ERROR, "Reload crashed: ${e.message}")
                } finally {
                    isReloading.set(false)
                }
            }
            return JSONObject().apply {
                put("ok", true)
                put("message", "Reload started")
                put("model_path", modelPath)
                put("queued_requests", requestQueue.size)
            }
        } else {
            return errorResponse("Reload already in progress", "conflict", 409)
        }
    }

    private fun handleLoadModel(body: String, authHeader: String, clientIp: String): JSONObject {
        return handleReload(body, authHeader, clientIp)
    }

    // ── Helpers ───────────────────────────────────────────────────

    /**
     * Parse messages with vision support.
     * If a message contains image_url content, extracts base64 and prepends to prompt.
     */
    private fun messagesToPromptWithVision(messages: JSONArray): String {
        val parts = mutableListOf<String>()
        var hasImages = false

        for (i in 0 until messages.length()) {
            val msg = messages.optJSONObject(i) ?: continue
            val role = msg.optString("role", "user").uppercase()
            val content = msg.get("content")

            when (content) {
                is String -> {
                    parts.add("$role: $content")
                }
                is JSONArray -> {
                    // Vision format: array of text + image_url objects
                    val textParts = mutableListOf<String>()
                    for (j in 0 until content.length()) {
                        val item = content.optJSONObject(j) ?: continue
                        when (item.optString("type")) {
                            "text" -> textParts.add(item.optString("text", ""))
                            "image_url" -> {
                                hasImages = true
                                val url = item.optJSONObject("image_url")?.optString("url", "") ?: ""
                                if (url.startsWith("data:")) {
                                    textParts.add("[Image attached: base64 encoded ${url.substringBefore(",").substringAfter("/")}]")
                                } else {
                                    textParts.add("[Image: $url]")
                                }
                            }
                        }
                    }
                    parts.add("$role: ${textParts.joinToString(" ")}")
                }
                else -> {
                    parts.add("$role: $content")
                }
            }
        }

        if (hasImages) {
            parts.add(0, "[Note: This conversation includes images. The model should describe and analyze them.]")
        }

        return parts.joinToString("\n\n") + "\n\nASSISTANT: "
    }

    private fun messagesToPrompt(messages: JSONArray): String {
        val parts = mutableListOf<String>()
        for (i in 0 until messages.length()) {
            val msg = messages.optJSONObject(i) ?: continue
            val role = msg.optString("role", "user").uppercase()
            val content = msg.optString("content", "")
            parts.add("$role: $content")
        }
        return parts.joinToString("\n\n") + "\n\nASSISTANT: "
    }

    private fun errorResponse(message: String, code: String, status: Int): JSONObject {
        return JSONObject().apply {
            put("error", JSONObject().apply {
                put("message", message)
                put("type", code)
                put("code", code)
                put("status", status)
                put("server_status", this@PhonoLabApiServer.status.name.lowercase())
                put("queue_size", requestQueue.size)
            })
        }
    }

    private fun sendResponse(socket: Socket, body: JSONObject, status: Int = 200, extraHeaders: Map<String, String> = emptyMap()) {
        val bodyBytes = body.toString().toByteArray(Charsets.UTF_8)
        val statusText = when (status) {
            200 -> "OK"
            400 -> "Bad Request"
            401 -> "Unauthorized"
            404 -> "Not Found"
            409 -> "Conflict"
            413 -> "Request Entity Too Large"
            500 -> "Internal Server Error"
            503 -> "Service Unavailable"
            504 -> "Gateway Timeout"
            else -> "OK"
        }
        val headers = buildString {
            append("HTTP/1.1 $status $statusText\r\n")
            append("Content-Type: application/json; charset=utf-8\r\n")
            append("Content-Length: ${bodyBytes.size}\r\n")
            append("Access-Control-Allow-Origin: *\r\n")
            append("Access-Control-Allow-Headers: authorization, x-api-key, content-type\r\n")
            append("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n")
            append("X-PhonoLab-Status: ${this@PhonoLabApiServer.status.name.lowercase()}\r\n")
            append("X-PhonoLab-Queue: ${requestQueue.size}\r\n")
            append("Connection: close\r\n")
            for ((k, v) in extraHeaders) append("$k: $v\r\n")
            append("\r\n")
        }
        try {
            socket.getOutputStream().write(headers.toByteArray(Charsets.UTF_8))
            socket.getOutputStream().write(bodyBytes)
            socket.getOutputStream().flush()
        } catch (_: Exception) {}
    }

    private fun addLog(msg: String) {
        val ts = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.US).format(java.util.Date())
        val entry = "$ts $msg"
        synchronized(logLock) {
            requestLog.add(entry)
            if (requestLog.size > 200) requestLog.removeAt(0)
        }
        onLog(entry)
    }
}
