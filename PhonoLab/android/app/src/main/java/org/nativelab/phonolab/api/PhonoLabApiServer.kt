package org.nativelab.phonolab.api

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.ServerSocket
import java.net.Socket
import java.security.MessageDigest
import java.util.concurrent.Executors
import kotlin.concurrent.thread

/**
 * HTTP API server for PhonoLab — OpenAI and Anthropic compatible.
 * Runs on Android, accepts requests from NativeLab nodes on LAN/WAN.
 */
class PhonoLabApiServer(
    private val config: ApiConfig,
    private val onLog: (String) -> Unit = {},
    private val generateFn: (prompt: String, nPredict: Int, temperature: Float, topP: Float) -> String,
    private val streamGenerateFn: ((prompt: String, nPredict: Int, temperature: Float, topP: Float, onToken: (String) -> Unit) -> String)? = null,
    private val runtimeInfo: () -> Map<String, Any> = { emptyMap() },
    private val modelList: () -> List<Map<String, Any>> = { emptyMapList() },
) {
    private var serverSocket: ServerSocket? = null
    private var running = false
    private val executor = Executors.newFixedThreadPool(4)
    private val requestLog = mutableListOf<String>()
    private val logLock = Any()

    companion object {
        private const val MAX_BODY_BYTES = 1024 * 1024  // 1MB
        private fun emptyMapList() = emptyList<Map<String, Any>>()
    }

    val isRunning: Boolean get() = running

    fun start(): String {
        if (running) return "Already running"
        try {
            serverSocket = ServerSocket(config.port, 50, java.net.InetAddress.getByName(config.host))
            running = true
            thread(isDaemon = true) { acceptLoop() }
            val url = if (config.host == "0.0.0.0") config.lanBaseUrl else config.localBaseUrl
            addLog("Server started on ${config.host}:${config.port}")
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
        addLog("Server stopped")
    }

    fun getRecentLogs(): List<String> {
        synchronized(logLock) {
            return requestLog.takeLast(50).toList()
        }
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
            socket.soTimeout = 300_000  // 5 min for long generations
            val reader = BufferedReader(InputStreamReader(socket.inputStream))

            // Read request line
            val requestLine = reader.readLine() ?: return
            val parts = requestLine.split(" ")
            if (parts.size < 2) return
            val method = parts[0]
            val path = parts[1].split("?")[0].trimEnd('/')

            // Read headers
            val headers = mutableMapOf<String, String>()
            while (true) {
                val line = reader.readLine() ?: break
                if (line.isEmpty()) break
                val colon = line.indexOf(':')
                if (colon > 0) {
                    headers[line.substring(0, colon).trim().lowercase()] = line.substring(colon + 1).trim()
                }
            }

            // Read body
            val contentLength = headers["content-length"]?.toIntOrNull() ?: 0
            if (contentLength > MAX_BODY_BYTES) {
                sendResponse(socket, errorResponse(
                    "Body too large (${contentLength} bytes, max ${MAX_BODY_BYTES})",
                    "payload_too_large", 413
                ))
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

            // Parse auth
            val authHeader = headers["authorization"] ?: headers["x-api-key"] ?: ""
            val clientIp = socket.inetAddress.hostAddress ?: "unknown"

            addLog("$method $path from $clientIp")

            // Route
            val isStreamRequest = method == "POST" &&
                path in listOf("/chat/completions", "/v1/chat/completions") &&
                body.contains("\"stream\"") && body.contains("true")

            val response = when {
                method == "OPTIONS" -> handleOptions()
                method == "GET" && path in listOf("/health", "/v1/health") -> handleHealth(clientIp)
                method == "GET" && path == "/runtime" -> handleRuntime(clientIp)
                method == "GET" && path in listOf("/models", "/v1/models") -> handleModels(clientIp, authHeader)
                method == "GET" && path == "/capabilities" -> handleCapabilities(clientIp, authHeader)
                isStreamRequest -> {
                    // Streaming response — sent directly to socket, returns null if success
                    val err = handleOpenAiChatStreaming(socket, body, authHeader)
                    if (err == null) return // Already sent
                    err
                }
                method == "POST" && path in listOf("/chat/completions", "/v1/chat/completions") ->
                    handleOpenAiChat(body, authHeader)
                method == "POST" && path in listOf("/completions", "/v1/completions") ->
                    handleOpenAiCompletion(body, authHeader)
                method == "POST" && path in listOf("/messages", "/v1/messages") ->
                    handleAnthropicMessages(body, authHeader)
                else -> errorResponse("Unknown endpoint: $path", "not_found", 404)
            }

            sendResponse(socket, response)
        } catch (e: Exception) {
            addLog("Error: ${e.message}")
        } finally {
            try { socket.close() } catch (_: Exception) {}
        }
    }

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

    // ── Handlers ────────────────────────────────────────────────────

    private fun handleOptions(): JSONObject {
        return JSONObject().put("ok", true)
    }

    private fun handleHealth(clientIp: String): JSONObject {
        val info = runtimeInfo()
        return JSONObject().apply {
            put("ok", true)
            put("name", "PhonoLab API Server")
            put("runtime", JSONObject().apply {
                put("loaded", info["loaded"] ?: false)
                put("model", info["model"] ?: "none")
                put("state", info["state"] ?: "idle")
            })
            put("endpoints", JSONObject().apply {
                put("local_base_url", config.localBaseUrl)
                put("lan_base_url", config.lanBaseUrl)
                put("openai_chat", "${config.localBaseUrl}/chat/completions")
                put("models", "${config.localBaseUrl}/models")
            })
            put("auth", JSONObject().apply {
                put("required", config.requireApiKey)
                put("local_key_header", "Authorization: Bearer <local_api_key>")
                put("lan_key_header", "Authorization: Bearer <lan_api_key>")
            })
        }
    }

    private fun handleRuntime(clientIp: String): JSONObject {
        val info = runtimeInfo()
        return JSONObject().apply {
            put("loaded", info["loaded"] ?: false)
            put("model", info["model"] ?: "none")
            put("model_path", info["model_path"] ?: "")
            put("state", info["state"] ?: "idle")
            put("ctx", info["ctx"] ?: 2048)
        }
    }

    private fun handleModels(clientIp: String, authHeader: String): JSONObject {
        requireAuth(clientIp, authHeader)?.let { return it }
        val models = modelList()
        return JSONObject().apply {
            put("object", "list")
            put("data", JSONArray().apply {
                for (m in models) {
                    put(JSONObject().apply {
                        put("id", m["id"] ?: "unknown")
                        put("object", "model")
                        put("created", System.currentTimeMillis() / 1000)
                        put("owned_by", "phonolab")
                    })
                }
                if (models.isEmpty()) {
                    put(JSONObject().apply {
                        put("id", "phonolab-active")
                        put("object", "model")
                        put("created", System.currentTimeMillis() / 1000)
                        put("owned_by", "phonolab")
                    })
                }
            })
        }
    }

    private fun handleCapabilities(clientIp: String, authHeader: String): JSONObject {
        requireAuth(clientIp, authHeader)?.let { return it }
        return JSONObject().apply {
            put("object", "phonolab.capabilities")
            put("runtime", handleRuntime(clientIp))
            put("endpoints", handleHealth(clientIp).getJSONObject("endpoints"))
            put("models", handleModels(clientIp, authHeader).getJSONArray("data"))
        }
    }

    private fun handleOpenAiChat(body: String, authHeader: String): JSONObject {
        try {
            val payload = JSONObject(body)
            val messages = payload.optJSONArray("messages") ?: JSONArray()
            val prompt = messagesToPrompt(messages)
            val nPredict = payload.optInt("max_tokens", payload.optInt("n_predict", 512))
            val temperature = payload.optDouble("temperature", 0.7).toFloat()
            val topP = payload.optDouble("top_p", 0.9).toFloat()
            val stream = payload.optBoolean("stream", false)

            val text = generateFn(prompt, nPredict, temperature, topP)
            val modelId = runtimeInfo()["model"] ?: "phonolab-active"

            return JSONObject().apply {
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
        } catch (e: Exception) {
            return errorResponse(e.message ?: "Generation failed", "server_error", 500)
        }
    }

    /**
     * Handle streaming chat completion — sends SSE chunks to the socket directly.
     * Returns null (response already sent), or an error JSONObject.
     */
    private fun handleOpenAiChatStreaming(
        socket: Socket,
        body: String,
        authHeader: String,
    ): JSONObject? {
        try {
            val payload = JSONObject(body)
            val messages = payload.optJSONArray("messages") ?: JSONArray()
            val prompt = messagesToPrompt(messages)
            val nPredict = payload.optInt("max_tokens", payload.optInt("n_predict", 512))
            val temperature = payload.optDouble("temperature", 0.7).toFloat()
            val topP = payload.optDouble("top_p", 0.9).toFloat()
            val modelId = runtimeInfo()["model"] ?: "phonolab-active"

            val streamFn = streamGenerateFn ?: return errorResponse(
                "Streaming not supported by this runtime.", "server_error", 501
            )

            // Send SSE headers
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

            // Role chunk
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
                return null
            }

            // Generate with token callback — guard writes against client disconnect
            var clientDisconnected = false
            streamFn(prompt, nPredict, temperature, topP) { token ->
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

            if (clientDisconnected) return null

            // Finish chunk
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

            return null // Already sent
        } catch (e: Exception) {
            return errorResponse(e.message ?: "Streaming failed", "server_error", 500)
        }
    }

    private fun handleOpenAiCompletion(body: String, authHeader: String): JSONObject {
        try {
            val payload = JSONObject(body)
            val prompt = payload.optString("prompt", "")
            val nPredict = payload.optInt("max_tokens", payload.optInt("n_predict", 512))
            val temperature = payload.optDouble("temperature", 0.7).toFloat()
            val topP = payload.optDouble("top_p", 0.9).toFloat()

            val text = generateFn(prompt, nPredict, temperature, topP)
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
        } catch (e: Exception) {
            return errorResponse(e.message ?: "Generation failed", "server_error", 500)
        }
    }

    private fun handleAnthropicMessages(body: String, authHeader: String): JSONObject {
        try {
            val payload = JSONObject(body)
            val messages = payload.optJSONArray("messages") ?: JSONArray()
            val system = payload.optString("system", "")
            val prompt = if (system.isNotEmpty()) "System: $system\n\n${messagesToPrompt(messages)}" else messagesToPrompt(messages)
            val maxTokens = payload.optInt("max_tokens", 512)
            val temperature = payload.optDouble("temperature", 0.7).toFloat()
            val topP = payload.optDouble("top_p", 0.9).toFloat()

            val text = generateFn(prompt, maxTokens, temperature, topP)
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
        } catch (e: Exception) {
            return errorResponse(e.message ?: "Generation failed", "server_error", 500)
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────

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
            413 -> "Request Entity Too Large"
            500 -> "Internal Server Error"
            else -> "OK"
        }
        val headers = buildString {
            append("HTTP/1.1 $status $statusText\r\n")
            append("Content-Type: application/json; charset=utf-8\r\n")
            append("Content-Length: ${bodyBytes.size}\r\n")
            append("Access-Control-Allow-Origin: *\r\n")
            append("Access-Control-Allow-Headers: authorization, x-api-key, content-type\r\n")
            append("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n")
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
}
