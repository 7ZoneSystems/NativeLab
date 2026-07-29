# Cross-Platform Integration

NativeLab and PhonoLab are designed to work together seamlessly across desktop and mobile platforms, creating a unified local-first AI ecosystem.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          NativeLab Desktop                              │
│  • PyQt6 GUI with pipeline builder, labs, and advanced features        │
│  • CLI client for terminal users                                       │
│  • Local llama.cpp server or CLI inference                             │
│  • API integrations (OpenAI, Anthropic, Ollama)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          PhonoLab Android                               │
│  • On-device llama.cpp inference via bundled binaries                   │
│  • Vision model support (LLaVA, Qwen-VL, etc.)                         │
│  • RAG document processing for PDFs, docs, and text files              │
│  • LAN API server for desktop-to-mobile communication                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Integration Points

### 1. LAN Device Discovery & API Integration

NativeLab can automatically discover PhonoLab devices on the same local network and use them as remote AI servers:

- **Device Discovery**: NativeLab's "Devices" tab scans LAN for PhonoLab API servers
- **Smart Authentication**: Automatic key exchange and management
- **Seamless Integration**: PhonoLab appears as a regular API model in NativeLab's model selector
- **Cross-Platform Workflows**: Use desktop NativeLab to orchestrate pipelines that include mobile PhonoLab inference

### 2. Shared Model Ecosystem

Both platforms use the same GGUF model format and support identical quantization levels:

- **Model Compatibility**: Models downloaded on desktop can be transferred to Android and vice versa
- **Shared Catalog**: Same model catalog (SmolLM2, Qwen2.5, Llama 3.2, etc.) available on both platforms
- **Consistent Configuration**: Same model parameters (temperature, top_p, context size) work identically

### 3. Unified API Standards

PhonoLab implements the same OpenAI and Anthropic API standards as NativeLab's API integrations:

- **OpenAI-compatible endpoints**: `/v1/chat/completions`, `/v1/models`, `/v1/health`
- **Anthropic-compatible endpoints**: `/v1/messages`, `/v1/health`
- **Streaming support**: Server-Sent Events (SSE) for real-time token streaming
- **Vision support**: Same image URL format for multimodal requests

### 4. Shared Documentation & Development Patterns

- **Consistent Error Handling**: Same multi-layer error handling patterns (fatal vs. non-fatal)
- **Unified Configuration**: Similar configuration structures and defaults
- **Shared Best Practices**: Math rendering, security hardening, and performance optimizations
- **Common Development Workflow**: Same testing, CI/CD, and documentation standards

## Getting Started with Cross-Platform Workflows

### Using PhonoLab as a Remote Server from NativeLab

1. Install PhonoLab on your Android device
2. Enable the API server in PhonoLab's "API Endpoint" tab
3. In NativeLab desktop, go to "Dev" → "Devices" tab
4. Click "Scan Network" to discover your PhonoLab device
5. Click "Register as Model" to add it to your model registry
6. Select it from the model dropdown and use it like any other model

### Transferring Models Between Platforms

- **From Desktop to Mobile**: Download models using NativeLab's model downloader, then transfer the `.gguf` files to your Android device
- **From Mobile to Desktop**: Use PhonoLab's model downloader, then transfer files via USB or cloud storage

### Cross-Platform Development

When developing new features:

- **Desktop-first**: Implement core logic in Python for NativeLab
- **Mobile adaptation**: Port to Kotlin/Java for PhonoLab with JNI bindings
- **Shared libraries**: Extract common logic into shared libraries when possible
- **Consistent testing**: Test features on both platforms before release

## Platform-Specific Capabilities

| Feature | NativeLab Desktop | PhonoLab Android |
|---------|-------------------|------------------|
| **Model Size** | Supports larger models (up to 13B+ on high-end desktops) | Optimized for mobile (up to 3B on most devices) |
| **Vision Models** | Supported via API integrations | Native on-device vision model support |
| **RAG Processing** | PDF/document processing via Python libraries | Native on-device PDF/text/docx processing |
| **Storage** | File system access | SAF (Storage Access Framework) with permission handling |
| **UI Capabilities** | Advanced pipeline builder, labs, and visualization | Optimized chat interface with mobile-specific UX |
| **Performance** | Higher throughput, larger context windows | Optimized for battery life and memory constraints |

## Troubleshooting Cross-Platform Issues

### Common Issues and Solutions

- **Device not discovered**: Ensure both devices are on the same WiFi network and PhonoLab API server is enabled
- **Connection timeouts**: Check firewall settings and ensure PhonoLab has internet permission
- **Model loading failures**: Verify model compatibility (quantization level, architecture)
- **Authentication errors**: Reset API keys in both PhonoLab and NativeLab
- **Streaming issues**: Check network stability and try reducing context size

### Diagnostic Commands

```bash
# Test PhonoLab API server connectivity
curl http://[PHONO_LAB_IP]:8080/health

# List available models on PhonoLab
curl http://[PHONO_LAB_IP]:8080/v1/models

# Test a simple chat request
curl -X POST http://[PHONO_LAB_IP]:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "phonolab-active", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'
```

## Future Integration Plans

- **Synchronized Sessions**: Share chat sessions between desktop and mobile
- **Unified Model Registry**: Cloud-synced model catalog across platforms
- **Cross-Platform Pipelines**: Create pipelines that span both desktop and mobile devices
- **Shared Labs**: Experimental features available on both platforms
- **Unified Update System**: Single update mechanism for both NativeLab and PhonoLab

---

This cross-platform integration makes NativeLab and PhonoLab a complete ecosystem for local AI development and usage, from powerful desktop workflows to convenient mobile inference.