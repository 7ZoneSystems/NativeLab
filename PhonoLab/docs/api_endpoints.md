# PhonoLab API Endpoints

Complete reference for the PhonoLab LAN/WAN API server. All endpoints are available on both local (`127.0.0.1`) and LAN/WAN interfaces.

---

## Base URLs

| Interface | URL |
|-----------|-----|
| Local | `http://127.0.0.1:{port}/v1` |
| LAN | `http://{device-ip}:{port}/v1` |
| WAN | `http://{public-ip}:{port}/v1` |

Default port: `8787`

---

## Authentication

All endpoints (except `/health` and `/status`) require an API key when `require_api_key` is enabled.

```bash
# Local requests
curl -H "Authorization: Bearer {local_api_key}" http://127.0.0.1:8787/v1/models

# LAN/WAN requests
curl -H "Authorization: Bearer {lan_api_key}" http://192.168.1.100:8787/v1/models
```

API keys are auto-generated UUIDs with `nl-` prefix. View keys in the app's API tab.

---

## Status & Health

### GET /health
Server health check with full status. No auth required.

```bash
curl http://127.0.0.1:8787/health
```

**Response:**
```json
{
  "ok": true,
  "name": "PhonoLab API Server",
  "version": "2.0.0",
  "status": "ready",
  "status_message": "Ready: SmolLM2-360M-Instruct.Q4_K_M.gguf",
  "runtime": {
    "loaded": true,
    "model": "SmolLM2-360M-Instruct.Q4_K_M.gguf",
    "model_path": "/path/to/model.gguf",
    "state": "loaded",
    "is_vision": false,
    "ctx": 2048,
    "active_generations": 0,
    "queue_size": 0
  },
  "endpoints": { ... },
  "auth": { ... },
  "features": {
    "streaming": true,
    "vision": false,
    "document_rag": true,
    "parameter_editing": true,
    "request_queuing": true,
    "smart_reload": true,
    "device_info": true
  }
}
```

### GET /status
Detailed live status. No auth required.

```bash
curl http://127.0.0.1:8787/status
```

**Response:**
```json
{
  "status": "ready",
  "status_message": "Ready: model.gguf",
  "last_error": null,
  "runtime": {
    "loaded": true,
    "model": "model.gguf",
    "is_vision": false,
    "ctx": 2048
  },
  "server": {
    "active_generations": 0,
    "queue_size": 0,
    "is_reloading": false,
    "uptime_ms": 1234567890
  }
}
```

**Status values:**
| Status | Meaning |
|--------|---------|
| `idle` | No model loaded, server running |
| `loading` | Model is being loaded |
| `ready` | Model loaded, ready for requests |
| `generating` | Currently generating a response |
| `reloading` | Model is being reloaded (switching) |
| `error` | Last operation failed |

### GET /runtime
Runtime info only. No auth required.

```bash
curl http://127.0.0.1:8787/runtime
```

### GET /queue
Queue status. Requires auth.

```bash
curl -H "Authorization: Bearer {key}" http://127.0.0.1:8787/queue
```

**Response:**
```json
{
  "queue_size": 2,
  "max_queue_size": 50,
  "is_reloading": false,
  "active_generations": 1,
  "status": "generating"
}
```

### GET /device
Device hardware and system info. Requires auth.

```bash
curl -H "Authorization: Bearer {key}" http://127.0.0.1:8787/device
```

**Response:**
```json
{
  "device": {
    "model": "Pixel 8",
    "manufacturer": "Google",
    "brand": "google",
    "device": "shiba",
    "product": "shiba",
    "hardware": "tensor",
    "board": "tensor",
    "android_version": "14",
    "sdk_int": 34,
    "build_id": "AP1A.240505.005",
    "fingerprint": "google/shiba/shiba:14/..."
  },
  "cpu": {
    "cores": 8,
    "abis": ["arm64-v8a", "armeabi-v7a", "armeabi"],
    "primary_abi": "arm64-v8a",
    "is_64bit": true
  },
  "memory": {
    "jvm_max_mb": 256,
    "jvm_used_mb": 128,
    "jvm_free_mb": 64,
    "jvm_total_mb": 192,
    "system_total_mb": 8192,
    "system_available_mb": 4096,
    "system_used_mb": 4096
  },
  "storage": {
    "data_total_mb": 128000,
    "data_free_mb": 64000
  },
  "runtime": {
    "status": "ready",
    "status_message": "Ready: model.gguf",
    "loaded": true,
    "model": "model.gguf",
    "is_vision": false,
    "active_generations": 0,
    "queue_size": 0
  }
}
```

### GET /system
Alias for `/device`.

---

## Models

### GET /v1/models
List all available models. Requires auth.

```bash
curl -H "Authorization: Bearer {key}" http://127.0.0.1:8787/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "SmolLM2-360M-Instruct.Q4_K_M.gguf",
      "object": "model",
      "created": 1234567890,
      "owned_by": "phonolab",
      "active": true
    }
  ]
}
```

### GET /capabilities
Full capabilities. Requires auth.

```bash
curl -H "Authorization: Bearer {key}" http://127.0.0.1:8787/capabilities
```

---

## Chat Completions (OpenAI Compatible)

### POST /v1/chat/completions
Generate a chat completion. Supports streaming and vision.

```bash
# Non-streaming
curl -X POST http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer {key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model.gguf",
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "stream": false
  }'
```

```bash
# Streaming (SSE)
curl -X POST http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer {key}" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a poem"}],
    "stream": true
  }'
```

**Request parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | auto | Model ID (optional, uses loaded model) |
| `messages` | array | required | Chat messages |
| `max_tokens` | int | 512 | Max tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | 0.9 | Nucleus sampling (0.0-1.0) |
| `top_k` | int | 40 | Top-K sampling |
| `repeat_penalty` | float | 1.1 | Repeat penalty (1.0-2.0) |
| `stream` | bool | false | Enable SSE streaming |

### Vision Support

Send images as base64 data URLs in the `image_url` content:

```bash
curl -X POST http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer {key}" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
        ]
      }
    ]
  }'
```

### Document Context

Include document text directly in the message:

```bash
curl -X POST http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer {key}" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "Answer based on the document."},
      {"role": "user", "content": "[Document content]\nYour document text here...\n[End document]\n\nSummarize this document."}
    ]
  }'
```

---

## Text Completions (OpenAI Compatible)

### POST /v1/completions
Raw text completion.

```bash
curl -X POST http://127.0.0.1:8787/v1/completions \
  -H "Authorization: Bearer {key}" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.8
  }'
```

---

## Anthropic Compatible

### POST /v1/messages
Anthropic Messages API format.

```bash
curl -X POST http://127.0.0.1:8787/v1/messages \
  -H "Authorization: Bearer {key}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model.gguf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "system": "You are helpful.",
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1
  }'
```

---

## Configuration

### POST /config
Update model parameters at runtime. Requires auth.

```bash
curl -X POST http://127.0.0.1:8787/config \
  -H "Authorization: Bearer {key}" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 0.95,
    "repeat_penalty": 1.2,
    "max_tokens": 512,
    "ctx": 4096
  }'
```

**All parameters are optional.** Only specified values are updated.

**Response:**
```json
{
  "ok": true,
  "message": "Configuration updated",
  "config": {
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 0.95,
    "repeat_penalty": 1.2,
    "max_tokens": 512,
    "ctx": 4096
  }
}
```

---

## Model Management

### POST /reload
Smart reload a model. Queues incoming requests during reload. Requires auth.

```bash
curl -X POST http://127.0.0.1:8787/reload \
  -H "Authorization: Bearer {key}" \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/path/to/new/model.gguf"}'
```

**Response:**
```json
{
  "ok": true,
  "message": "Reload started",
  "model_path": "/path/to/new/model.gguf",
  "queued_requests": 3
}
```

During reload:
- Status changes to `reloading`
- New chat/completion requests are queued (up to 50)
- Queue timeout: 2 minutes per request
- Requests are automatically processed when reload completes

### POST /load
Alias for `/reload`.

---

## Request Queuing

When the model is reloading or loading, incoming chat/completion requests are automatically queued:

- **Max queue size:** 50 requests
- **Queue timeout:** 2 minutes per request
- **Queue full:** Returns 503 with `server_busy` error
- **Queue timeout:** Returns 504 with `gateway_timeout` error

Queue status is reported in:
- Response headers: `X-PhonoLab-Queue: {size}`
- `/status` endpoint
- `/queue` endpoint
- Error responses include `queue_size` and `server_status`

---

## Response Headers

All responses include these headers:

| Header | Description |
|--------|-------------|
| `X-PhonoLab-Status` | Current server status (idle/loading/ready/generating/reloading/error) |
| `X-PhonoLab-Queue` | Current queue size |
| `Access-Control-Allow-Origin` | `*` (CORS enabled) |
| `Access-Control-Allow-Headers` | `authorization, x-api-key, content-type` |
| `Access-Control-Allow-Methods` | `GET, POST, OPTIONS` |

---

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "message": "Human-readable error description",
    "type": "error_code",
    "code": "error_code",
    "status": 400,
    "server_status": "ready",
    "queue_size": 0
  }
}
```

**Error codes:**
| Code | HTTP Status | Meaning |
|------|-------------|---------|
| `authentication_error` | 401 | Invalid or missing API key |
| `invalid_request` | 400 | Bad request parameters |
| `not_found` | 404 | Unknown endpoint |
| `payload_too_large` | 413 | Request body exceeds 1MB |
| `conflict` | 409 | Reload already in progress |
| `model_not_loaded` | 503 | No model loaded — use POST /load first |
| `model_loading` | 503 | Model is currently loading/reloading |
| `server_error` | 500 | Internal server error |
| `server_busy` | 503 | Queue full (50 requests) |
| `gateway_timeout` | 504 | Request timed out in queue (2 min) |
| `not_implemented` | 501 | Feature not supported |

---

## Streaming (SSE)

When `"stream": true` is set, responses are sent as Server-Sent Events:

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","model":"model.gguf","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","model":"model.gguf","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","model":"model.gguf","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

## CORS

All endpoints support CORS with `Access-Control-Allow-Origin: *`. Preflight `OPTIONS` requests are handled automatically.

---

## Usage Examples

### Python (requests)
```python
import requests

API = "http://192.168.1.100:8787"
KEY = "nl-your-api-key-here"

# Chat
resp = requests.post(f"{API}/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
}, headers={"Authorization": f"Bearer {KEY}"})
print(resp.json()["choices"][0]["message"]["content"])

# Update config
requests.post(f"{API}/config", json={
    "temperature": 0.9,
    "top_k": 50
}, headers={"Authorization": f"Bearer {KEY}"})

# Smart reload
requests.post(f"{API}/reload", json={
    "model_path": "/path/to/model.gguf"
}, headers={"Authorization": f"Bearer {KEY}"})
```

### curl (streaming)
```bash
curl -N -X POST http://192.168.1.100:8787/v1/chat/completions \
  -H "Authorization: Bearer nl-your-key" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a haiku"}],"stream":true}'
```

### JavaScript (fetch)
```javascript
const res = await fetch("http://192.168.1.100:8787/v1/chat/completions", {
  method: "POST",
  headers: {
    "Authorization": "Bearer nl-your-key",
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    messages: [{ role: "user", content: "Hello!" }],
    stream: true
  })
});

const reader = res.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const text = decoder.decode(value);
  // Parse SSE lines
  for (const line of text.split("\n")) {
    if (line.startsWith("data: ") && line !== "data: [DONE]") {
      const data = JSON.parse(line.slice(6));
      process.stdout.write(data.choices[0]?.delta?.content || "");
    }
  }
}
```

---

## NativeLab Integration

PhonoLab's API is compatible with NativeLab's API model profiles. Add it as an OpenAI-compatible endpoint:

1. In NativeLab, go to **Dev > Devices**
2. Click **Scan Network** to discover PhonoLab devices on your LAN
3. Select a device and click **Register as API Model**
4. If the device requires authentication, enter the LAN API key from PhonoLab's Dev > API Server tab
5. The device appears as an API model in the pipeline builder and chat

### Smart Auth Flow

- **First connection**: NativeLab tries connecting without a key. If 401, it prompts for the key.
- **Stored key**: On subsequent connections, the stored key is used automatically.
- **Key change**: If the key changes on PhonoLab, NativeLab detects the 401 and prompts for the new key.
- **No auth**: If PhonoLab has `require_api_key` disabled, NativeLab connects without a key.

### Device Capabilities

When registered, PhonoLab devices are available in:
- **Pipeline builder**: as model blocks with device-specific parameters
- **AI Pipeline Builder**: auto-assigned based on task type (vision, reasoning, etc.)
- **Chat**: as API model endpoints in the model picker
- **Config editing**: temperature, top_k, top_p, repeat_penalty via GUI sliders
