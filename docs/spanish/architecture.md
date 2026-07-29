# Arquitectura

NativeLab es una aplicación por capas. Cada capa solo se comunica con la que está debajo de ella, lo que mantiene las interfaces GUI, CLI y Labs intercambiables sobre la misma superficie de motor.

---

## Capas

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                              Frontends                                  │
│  ┌──────────────────────┐   ┌──────────────────┐   ┌────────────────┐   │
│  │  MainWindow (PyQt6)  │   │  CLI ChatREPL    │   │  Paneles Labs  │   │
│  │  • Chat / Modelos    │   │  (cli/chat.py)   │   │  (labs/*.py)   │   │
│  │  • Pipeline / Servidor│  │                  │   │                │   │
│  │  • MCP / Descarga    │   │                  │   │                │   │
│  │  • Labs / Registros  │   │                  │   │                │   │
│  │  • Dispositivos (LAN) │   │                  │   │                │   │
│  └──────────────────────┘   └──────────────────┘   └────────────────┘   │
│              │                       │                    │             │
│              └───────────┬───────────┴────────────────────┘             │
│                          ▼                                              │
│              ┌───────────────────────────┐                              │
│              │  LabEndpoints (labs/)     │  ← superficie compartida     │
│              │   • status_text / model   │                              │
│              │   • call_llm() sync       │                              │
│              │   • request_*() reverse   │                              │
│              └───────────────────────────┘                              │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Backend centralizado (core/)                        │
│  NativeLabBackend    - fachada unificada para todas las operaciones     │
│   • load_model / unload_model / generate                                │
│   • scan_network / test_device / register_device                        │
│   • fetch_hf_gguf_files / fetch_llama_cpp_releases                      │
│  NativeLabHttpClient - todas las operaciones HTTP con reintentos/timeouts│
│   • get / post / stream                                                 │
│   • post_openai_stream / post_anthropic_stream                          │
│  EngineStatus        - estado normalizado del motor                     │
│  LlmErrorNotice      - reporte estructurado de errores                  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            Capa de motor                                │
│  LlamaEngine        - gestiona procesos subprocess de llama-server /   │
│                       llama-cli                                         │
│   ├── ServerStreamWorker  (HTTP /completion streaming)                  │
│   └── CliStreamWorker     (stdout por prompt de llama-cli)              │
│  ApiEngine          - compatible con OpenAI / Anthropic                │
│   └── ApiStreamWorker     (streaming HTTP)                              │
│  PipelineExecutionWorker  ChunkedSummaryWorker  MultiPdfSummaryWorker   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Motor de referencias                             │
│  SessionReferenceStore  ─►  SmartReference                              │
│                              ScriptSmartReference  ─►  ScriptParser     │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Persistencia                                  │
│  Session (JSON en sessions/)       ApiRegistry (api_models.json)       │
│  ModelRegistry (custom_models.json + model_configs.json)                │
│  ServerConfig (server_config.json)   APP_CONFIG (app_config.json)       │
│  ParallelPrefs (parallel_prefs.json) PausedJobs (paused_jobs/)          │
│  McpConfig (mcp_config.json)         CLI prefs (cli_prefs.json)         │
└─────────────────────────────────────────────────────────────────────────┘
```

Las frontends nunca llaman directamente a la capa de motor; todo pasa por `LabEndpoints`. Así es como la CLI y la GUI se mantienen sincronizadas sin duplicar código.

---

## La superficie de `LabEndpoints`

`nativelab/labs/endpoints.py` es el contrato entre las frontends y los motores.

**Leer estado**

```python
endpoints.status_text        # "🟢 Server  :8612"
endpoints.model_path         # ruta absoluta del GGUF activo
endpoints.model_name         # solo el nombre del archivo
endpoints.mode               # "server" | "cli" | "api" | "unloaded"
endpoints.ctx_value          # int
endpoints.server_port        # int
endpoints.is_loaded          # bool
endpoints.snapshot()         # dict con todo lo anterior
endpoints.model_family()     # plantilla ModelFamily (BOS/EOS, prefijos…)
```

**Llamadas LLM** (sincrónicas, seguras desde un hilo worker; enruta automáticamente API → servidor → CLI)

```python
endpoints.call_llm(
    messages=[{"role": "user", "content": "…"}],
    system_prompt="You are…",
    n_predict=512,
    temperature=0.3,
)
```

**Enrutamiento inverso** - las frontends o labs solicitan cambios de estado a la app anfitriona:

```python
endpoints.request_load_model("/path/to/model.gguf")
endpoints.request_context(8192)
endpoints.request_unload()
endpoints.ensure_server(log_cb=...)
```

**Señales** para actualizaciones en vivo:

```python
endpoints.engine_changed.connect(handler)
endpoints.status_changed.connect(handler)   # str
```

La aplicación anfitriona (`MainWindow` para la GUI, `cli.chat._build_endpoints` para la CLI) conecta una sola vez los proveedores de motor y los hooks de enrutamiento inverso al iniciar. Todo lo que está debajo usa la misma instancia.

---

## Backend centralizado (`core/backend.py`)

El backend centralizado proporciona una interfaz unificada para todas las operaciones de NativeLab. Se sitúa entre las frontends/motores y la capa de red, garantizando manejo consistente de errores, autenticación y timeouts.

### `NativeLabBackend`

Punto de entrada único para todas las operaciones del backend:

```python
from nativelab.core.backend import get_backend

backend = get_backend()

# Operaciones de modelos
result = backend.load_model("/path/to/model.gguf")
result = backend.load_api_model(api_config)
result = backend.unload_model()
models = backend.list_models()
api_models = backend.list_api_models()

# Generación
result = backend.generate(
    messages=[{"role": "user", "content": "..."}],
    model_ref="@api/MyAPI",  # o ruta de modelo local
    n_predict=512,
    temperature=0.7,
    on_token=lambda t: print(t, end=""),
)

# Operaciones de dispositivos
devices = backend.scan_network()
result = backend.test_device(device, api_key="nl-...")
result = backend.register_device_as_model(device)
result = backend.load_model_on_device(device, "/path/to/model.gguf")

# HuggingFace
result = backend.fetch_hf_gguf_files("bartowski/SmolLM2-360M-Instruct-GGUF")
result = backend.fetch_llama_cpp_releases()
```

### `NativeLabHttpClient`

Todas las operaciones HTTP pasan por el cliente centralizado:

```python
from nativelab.core.http_client import get_http_client

http = get_http_client()

# Solicitudes básicas
resp = http.get("http://localhost:8080/health")
resp = http.post("http://api.openai.com/v1/chat/completions", body={...})

# Streaming
for line in http.stream("http://...", body={...}):
    process(line)

# Streaming compatible con OpenAI
text = http.post_openai_stream(
    url="http://localhost:8080/v1/chat/completions",
    messages=[...],
    model="model.gguf",
    on_token=lambda t: print(t, end=""),
)

# Streaming compatible con Anthropic
text = http.post_anthropic_stream(
    url="https://api.anthropic.com/v1/messages",
    messages=[...],
    model="claude-3-sonnet",
    api_key="sk-...",
)
```

### `BackendResult`

Todas las operaciones devuelven `BackendResult`:

```python
result = backend.load_model("/path/to/model.gguf")
if result.ok:
    print(result.data)  # {"model_path": "/path/to/model.gguf"}
else:
    print(result.error)  # "Model file not found"
    if result.error_notice:
        print(result.error_notice.user_message)  # Error legible con pasos de acción
```

### `EngineStatus`

Estado normalizado del motor:

```python
status = backend.get_engine_status()
print(status.status_text)  # "Server  :8612"
print(status.is_loaded)    # True
print(status.model_name)   # "mistral-7b-Q4_K_M.gguf"
print(status.backend)      # "server"
```

---

## Capa de motor

### `LlamaEngine`

El motor de inferencia local. Intenta primero `llama-server` (streaming HTTP, el modelo permanece residente), y si el binario del servidor no está disponible, cae a `llama-cli` (subproceso por prompt).

Métodos clave:

* `load(model_path, ctx, log_cb)` - inicia el servidor o cambia al modo CLI.
* `create_worker(prompt, n_predict, model_path)` - devuelve un `QThread` de streaming para la GUI.
* `ensure_server(log_cb)` - levanta el servidor si actualmente está en modo CLI.
* `shutdown()` - mata procesos hijos y reinicia el estado.

Banderas de estado: `is_loaded`, `mode`, `status_text`, `server_port`.

### `ApiEngine`

Reemplazo directo que llama a una API remota. Tiene la misma forma (`load`, `create_worker`, `is_loaded`, `status_text`) para que los workers de pipeline/summarization/reference no dependan de qué motor se esté usando.

Ambos motores son leídos por `LabEndpoints.active_engine()` - API tiene prioridad cuando está cargado; si no, se usa el local.

---

## Punto de entrada de la GUI y módulos de `MainWindow`

`nativelab/main.py` es intencionalmente pequeño. Maneja el arranque de la GUI, la creación de `QApplication`, la configuración de fuente/icono, el manejo de SIGINT y el lanzamiento de `MainWindow`.

La implementación de la ventana vive en `nativelab/UI/mainwindow/`:

| Módulo                | Responsabilidad                                               |
| --------------------- | ------------------------------------------------------------- |
| `window.py`           | Clase `MainWindow` y composición de mixins.                   |
| `ui_build.py`         | Diseño de alto nivel y construcción de pestañas.              |
| `engine_runtime.py`   | Carga/descarga de modelos locales/API y estado de ejecución.  |
| `auto_setup.py`       | Wiring de la UI de auto setup inicial y activada por ajustes. |
| `context_controls.py` | Medidor de contexto y controles de recarga de contexto.       |
| `chat_pipeline.py`    | Orquestación del modo chat/pipeline.                          |
| `documents.py`        | Referencias, resúmenes, trabajos multi-PDF.                   |
| `labs.py`             | Wiring de la pestaña Labs e inyección de endpoints.           |
| `models.py`           | Refresco y selección del registro de modelos/API.             |
| `sessions.py`         | Ciclo de vida y persistencia de sesiones.                     |
| `status_view.py`      | Barra de estado, tema y ayudas de vista.                      |
| `shared.py`           | Importaciones/constantes comunes para el paquete dividido.    |

El cierre de workers de la GUI se centraliza mediante `nativelab/UI/qt_workers.py` para que los eventos de cierre, reconstrucción del tema, descargas, auto setup, cargas de modelos, workers de cuentas y workers específicos de pestañas se detengan de forma consistente antes de que los widgets se eliminen.

---

## Límite de helpers nativos

NativeLab usa C/Rust solo para rutas críticas determinísticas. Python conserva la propiedad de los widgets Qt, la orquestación de plugins/backend, el ciclo de vida de procesos subprocess, las llamadas a modelos/API y el manejo de errores visibles para el usuario.

| Área nativa           | Archivos                                                      | Propósito                                                                                                                                                                      |
| --------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Helpers del motor     | `nativelab/native/_core.c`, `engine_helpers.py`               | Ensamblado de prompts, normalización de sampler, argumentos CLI del sampler, extracción de imagen/base64, detección de errores de contexto, división de chunks de referencias. |
| Núcleo de pipeline    | `nativelab/native/pipeline_core.c`, `pipeline_core.py`        | Normalización de IDs de bloques, remapeo de conexiones, comprobaciones de bucles/ciclos, selección de rutas, helpers de transform/merge, registros de validación.              |
| Detección de modelos  | `nativelab/native/rust_model.rs`, `rust_model.py`             | Detección de familia y quantización con soporte opcional en Rust.                                                                                                              |
| Helpers de AI Builder | `nativelab/pipelinebuilder/aibuilder/aibuilder_core.c`, `.rs` | Estimación de tokens y detección de spans de objetos JSON para respuestas generadas por el pipeline.                                                                           |

Todos los helpers nativos son opcionales. Si `_native_core` o la biblioteca compartida de Rust no están presentes, la ruta de respaldo en Python sigue activa.

---

## Subsistema de pipeline

El constructor de pipelines está dividido por responsabilidad:

| Archivo/paquete              | Responsabilidad                                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------------------------- |
| `pipebuilder.py`             | Pestaña PyQt, sidebars, pestañas Execution/AI Builder, acciones del usuario.                            |
| `canvas.py`                  | Edición visual del grafo, movimiento de bloques, conexiones de puertos, paneo, crecimiento del canvas.  |
| `pipblck.py` / `blck_typ.py` | Estructuras de datos de bloques y conexiones.                                                           |
| `pipefunctions.py`           | Persistencia JSON de guardar/cargar/ejemplos de pipelines.                                              |
| `graph_ops.py`               | Operaciones centrales del grafo y helpers nativos de ID/bucles.                                         |
| `execution_core.py`          | Helpers deterministas de ejecución usados por el worker.                                                |
| `validation.py`              | Validación compartida de pipelines y mensajes de validación visibles para el usuario.                   |
| `executionWorker.py`         | Runtime `QThread` para ejecución de bloques y llamadas a modelos.                                       |
| `aibuilder/`                 | UI del AI Pipeline Builder, planificación de prompts, extracción JSON, contexto inteligente, historial. |
| `examples/`                  | Presets JSON de pipelines de ejemplo empaquetados.                                                      |

Esto mantiene a Python como una capa delgada de orquestación alrededor de primitivas compartidas de validación y ejecución. El código de la UI ya no posee directamente las invariantes del grafo; los pipelines cargados, generados, ejecutados desde CLI o editados manualmente pasan por la misma ruta de validación y normalización.

---

## Estructura del proyecto

```text
NativeLab/
├── changelog.txt
├── CODE_OF_CONDUCT.md
├── comt.sh
├── CONTRIBUTING.md
├── dist
│   ├── nativelab-0.1.0-py3-none-any.whl
│   ├── nativelab-0.1.0.tar.gz
│   ├── nativelab-0.1.1-py3-none-any.whl
│   ├── nativelab-0.1.1.tar.gz
│   ├── nativelab-0.1.2-py3-none-any.whl
│   ├── nativelab-0.1.2.tar.gz
│   ├── nativelab-0.1.3-py3-none-any.whl
│   ├── nativelab-0.1.3.tar.gz
│   ├── nativelab-0.1.4-py3-none-any.whl
│   ├── nativelab-0.1.4.tar.gz
│   ├── nativelab-0.1.5-py3-none-any.whl
│   ├── nativelab-0.1.5.tar.gz
│   ├── nativelab-0.1.6-py3-none-any.whl
│   ├── nativelab-0.1.6.tar.gz
│   ├── nativelab-0.1.7-py3-none-any.whl
│   ├── nativelab-0.1.7.tar.gz
│   ├── nativelab-0.1.8-py3-none-any.whl
│   ├── nativelab-0.1.8.tar.gz
│   ├── nativelab-0.2.2-py3-none-any.whl
│   ├── nativelab-0.2.2.tar.gz
│   ├── nativelab-0.2.3-py3-none-any.whl
│   ├── nativelab-0.2.3.tar.gz
│   ├── nativelab-0.2.4-py3-none-any.whl
│   ├── nativelab-0.2.4.tar.gz
│   ├── nativelab-0.2.5-py3-none-any.whl
│   ├── nativelab-0.2.5.tar.gz
│   ├── nativelab-0.2.7-py3-none-any.whl
│   ├── nativelab-0.2.7.tar.gz
│   ├── nativelab-0.2.8-py3-none-any.whl
│   ├── nativelab-0.2.8.tar.gz
│   ├── nativelab-0.2.9-py3-none-any.whl
│   ├── nativelab-0.2.9.tar.gz
│   ├── nativelab-0.3.0-py3-none-any.whl
│   ├── nativelab-0.3.0.tar.gz
│   ├── nativelab-0.3.1-py3-none-any.whl
│   ├── nativelab-0.3.1.tar.gz
│   ├── nativelab-0.3.2-py3-none-any.whl
│   ├── nativelab-0.3.2.tar.gz
│   ├── nativelab-0.3.3-py3-none-any.whl
│   ├── nativelab-0.3.3.tar.gz
│   ├── nativelab-0.3.4-py3-none-any.whl
│   ├── nativelab-0.3.4.tar.gz
│   ├── nativelab-0.3.7-py3-none-any.whl
│   └── nativelab-0.3.7.tar.gz
├── docs
│   ├── architecture.md
│   ├── cli.md
│   ├── features.md
│   ├── installation.md
│   ├── integrations.md
│   ├── labs.md
│   ├── models.md
│   ├── README.md
│   ├── troubleshooting.md
│   ├── ui.md
│   └── workflows.md
├── .github
│   ├── ISSUE_TEMPLATE
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows
│       ├── build-linux.yml
│       ├── build-mac.yml
│       ├── build-windows.yml
│       ├── clone-count.yml
│       └── release-apps.yml
├── .gitignore
├── google81d8b06f71e45c58.html
├── images
│   ├── appearance.png
│   ├── dark_mode.png
│   ├── dev.png
│   ├── image copy.png
│   ├── light_mode.png
│   ├── pipeline.png
│   ├── server_controls.png
│   └── skill.png
├── index.html
├── LICENSE
├── MANIFEST.in
├── nativelab
│   ├── api_server
│   │   ├── catalog.py
│   │   ├── config.py
│   │   ├── __init__.py
│   │   ├── protocol.py
│   │   ├── server.py
│   │   └── tab.py
│   ├── assets
│   │   ├── icons
│   │   │   ├── blocks.svg
│   │   │   ├── book-open.svg
│   │   │   ├── brain.svg
│   │   │   ├── bug.svg
│   │   │   ├── calendar.svg
│   │   │   ├── circle-alert.svg
│   │   │   ├── circle-check.svg
│   │   │   ├── circle-pause.svg
│   │   │   ├── circle.svg
│   │   │   ├── circle-x.svg
│   │   │   ├── clipboard-list.svg
│   │   │   ├── code-2.svg
│   │   │   ├── code.svg
│   │   │   ├── combine.svg
│   │   │   ├── copy.svg
│   │   │   ├── delete.svg
│   │   │   ├── discord.svg
│   │   │   ├── download.svg
│   │   │   ├── file-code.svg
│   │   │   ├── files.svg
│   │   │   ├── file.svg
│   │   │   ├── file-text.svg
│   │   │   ├── filter.svg
│   │   │   ├── flask-conical.svg
│   │   │   ├── folder-open.svg
│   │   │   ├── folder.svg
│   │   │   ├── git-branch.svg
│   │   │   ├── globe.svg
│   │   │   ├── history.svg
│   │   │   ├── huggingface.svg
│   │   │   ├── image.svg
│   │   │   ├── import.svg
│   │   │   ├── key.svg
│   │   │   ├── lightbulb.svg
│   │   │   ├── list.svg
│   │   │   ├── loader-circle.svg
│   │   │   ├── log-in.svg
│   │   │   ├── log-out.svg
│   │   │   ├── logs.svg
│   │   │   ├── manifest.json
│   │   │   ├── map.svg
│   │   │   ├── merge.svg
│   │   │   ├── message-circle.svg
│   │   │   ├── message-square.svg
│   │   │   ├── more-horizontal.svg
│   │   │   ├── ollama.svg
│   │   │   ├── omega.svg
│   │   │   ├── palette.svg
│   │   │   ├── panel-left.svg
│   │   │   ├── panel-right-close.svg
│   │   │   ├── panel-top-close.svg
│   │   │   ├── panel-top.svg
│   │   │   ├── paperclip.svg
│   │   │   ├── pencil.svg
│   │   │   ├── pi.svg
│   │   │   ├── play.svg
│   │   │   ├── plug.svg
│   │   │   ├── plus.svg
│   │   │   ├── power-off.svg
│   │   │   ├── projector.svg
│   │   │   ├── radius.svg
│   │   │   ├── refresh-cw.svg
│   │   │   ├── regex.svg
│   │   │   ├── replace.svg
│   │   │   ├── route.svg
│   │   │   ├── save.svg
│   │   │   ├── search.svg
│   │   │   ├── section.svg
│   │   │   ├── send.svg
│   │   │   ├── server.svg
│   │   │   ├── settings.svg
│   │   │   ├── shuffle.svg
│   │   │   ├── sigma.svg
│   │   │   ├── split.svg
│   │   │   ├── square-chevron-down.svg
│   │   │   ├── square-chevron-right.svg
│   │   │   ├── stop-circle.svg
│   │   │   ├── table.svg
│   │   │   ├── tag.svg
│   │   │   ├── test-tube.svg
│   │   │   ├── text.svg
│   │   │   ├── trash-2.svg
│   │   │   ├── triangle-alert.svg
│   │   │   ├── type.svg
│   │   │   ├── upload.svg
│   │   │   ├── user.svg
│   │   │   ├── view.svg
│   │   │   ├── whatsapp.svg
│   │   │   ├── workflow.svg
│   │   │   ├── wrench.svg
│   │   │   ├── x.svg
│   │   │   └── zap.svg
│   │   └── katex
│   │       ├── auto-render.min.js
│   │       ├── fonts
│   │       │   ├── KaTeX_AMS-Regular.ttf
│   │       │   ├── KaTeX_AMS-Regular.woff
│   │       │   ├── KaTeX_AMS-Regular.woff2
│   │       │   ├── KaTeX_Caligraphic-Bold.ttf
│   │       │   ├── KaTeX_Caligraphic-Bold.woff
│   │       │   ├── KaTeX_Caligraphic-Bold.woff2
│   │       │   ├── KaTeX_Caligraphic-Regular.ttf
│   │       │   ├── KaTeX_Caligraphic-Regular.woff
│   │       │   ├── KaTeX_Caligraphic-Regular.woff2
│   │       │   ├── KaTeX_Fraktur-Bold.ttf
│   │       │   ├── KaTeX_Fraktur-Bold.woff
│   │       │   ├── KaTeX_Fraktur-Bold.woff2
│   │       │   ├── KaTeX_Fraktur-Regular.ttf
│   │       │   ├── KaTeX_Fraktur-Regular.woff
│   │       │   ├── KaTeX_Fraktur-Regular.woff2
│   │       │   ├── KaTeX_Main-BoldItalic.ttf
│   │       │   ├── KaTeX_Main-BoldItalic.woff
│   │       │   ├── KaTeX_Main-BoldItalic.woff2
│   │       │   ├── KaTeX_Main-Bold.ttf
│   │       │   ├── KaTeX_Main-Bold.woff
│   │       │   ├── KaTeX_Main-Bold.woff2
│   │       │   ├── KaTeX_Main-Italic.ttf
│   │       │   ├── KaTeX_Main-Italic.woff
│   │       │   ├── KaTeX_Main-Italic.woff2
│   │       │   ├── KaTeX_Main-Regular.ttf
│   │       │   ├── KaTeX_Main-Regular.woff
│   │       │   ├── KaTeX_Main-Regular.woff2
│   │       │   ├── KaTeX_Math-BoldItalic.ttf
│   │       │   ├── KaTeX_Math-BoldItalic.woff
│   │       │   ├── KaTeX_Math-BoldItalic.woff2
│   │       │   ├── KaTeX_Math-Italic.ttf
│   │       │   ├── KaTeX_Math-Italic.woff
│   │       │   ├── KaTeX_Math-Italic.woff2
│   │       │   ├── KaTeX_SansSerif-Bold.ttf
│   │       │   ├── KaTeX_SansSerif-Bold.woff
│   │       │   ├── KaTeX_SansSerif-Bold.woff2
│   │       │   ├── KaTeX_SansSerif-Italic.ttf
│   │       │   ├── KaTeX_SansSerif-Italic.woff
│   │       │   ├── KaTeX_SansSerif-Italic.woff2
│   │       │   ├── KaTeX_SansSerif-Regular.ttf
│   │       │   ├── KaTeX_SansSerif-Regular.woff
│   │       │   ├── KaTeX_SansSerif-Regular.woff2
│   │       │   ├── KaTeX_Script-Regular.ttf
│   │       │   ├── KaTeX_Script-Regular.woff
│   │       │   ├── KaTeX_Script-Regular.woff2
│   │       │   ├── KaTeX_Size1-Regular.ttf
│   │       │   ├── KaTeX_Size1-Regular.woff
│   │       │   ├── KaTeX_Size1-Regular.woff2
│   │       │   ├── KaTeX_Size2-Regular.ttf
│   │       │   ├── KaTeX_Size2-Regular.woff
│   │       │   ├── KaTeX_Size2-Regular.woff2
│   │       │   ├── KaTeX_Size3-Regular.ttf
│   │       │   ├── KaTeX_Size3-Regular.woff
│   │       │   ├── KaTeX_Size3-Regular.woff2
│   │       │   ├── KaTeX_Size4-Regular.ttf
│   │       │   ├── KaTeX_Size4-Regular.woff
│   │       │   ├── KaTeX_Size4-Regular.woff2
│   │       │   ├── KaTeX_Typewriter-Regular.ttf
│   │       │   ├── KaTeX_Typewriter-Regular.woff
│   │       │   └── KaTeX_Typewriter-Regular.woff2
│   │       ├── katex.min.css
│   │       ├── katex.min.js
│   │       └── LICENSE
│   ├── cli
│   │   ├── app.py
│   │   ├── chat.py
│   │   ├── cli_guide.md
│   │   ├── features.py
│   │   ├── hf_download.py
│   │   ├── __init__.py
│   │   ├── lint.py
│   │   ├── onboarding.py
│   │   ├── runtime.py
│   │   └── ui.py
│   ├── codeparser
│   │   ├── codeparser_global.py
│   │   ├── __init__.py
│   │   ├── parser
│   │   │   ├── __init__.py
│   │   │   ├── parsefinal.py
│   │   │   └── typeparser.py
│   │   ├── refrenceengine.py
│   │   └── scriptparser.py
│   ├── components
│   │   ├── components_global.py
│   │   ├── __init__.py
│   │   ├── jobhandler.py
│   │   ├── multipdf_summarise.py
│   │   ├── pdfsummarise.py
│   │   └── reason_code_pipeline.py
│   ├── core
│   │   ├── auto_setup.py
│   │   ├── context_meter.py
│   │   ├── data_portability.py
│   │   ├── engine_global.py
│   │   ├── engines
│   │   │   ├── apiengine.py
│   │   │   ├── __init__.py
│   │   │   └── llamaengine.py
│   │   ├── engine_status.py
│   │   ├── __init__.py
│   │   ├── model_loaders.py
│   │   ├── streamer_global.py
│   │   └── streamerworker
│   │       ├── apistreamer.py
│   │       ├── backendstreamer.py
│   │       ├── clistreamer.py
│   │       ├── __init__.py
│   │       └── serverstreamer.py
│   ├── GlobalConfig
│   │   ├── binaryResolve.py
│   │   ├── config_global.py
│   │   ├── config.py
│   │   ├── const.py
│   │   ├── hardwareUtil.py
│   │   ├── __init__.py
│   │   └── timeouts.py
│   ├── icon.ico
│   ├── icon.png
│   ├── imports
│   │   ├── import_global.py
│   │   ├── __init__.py
│   │   ├── optional_lib.py
│   │   ├── pyqt_lib.py
│   │   ├── qt_compat.py
│   │   └── standard_lib.py
│   ├── integrations
│   │   ├── discord_connector.py
│   │   ├── endpoints.py
│   │   ├── examples
│   │   │   ├── discord_bot.env.example
│   │   │   ├── discord_bot.py
│   │   │   ├── __init__.py
│   │   │   ├── whatsapp_bot.env.example
│   │   │   └── whatsapp_bot.py
│   │   ├── http_endpoint.py
│   │   ├── __init__.py
│   │   ├── tab.py
│   │   └── whatsapp_connector.py
│   ├── labs
│   │   ├── codeedit.py
│   │   ├── endpoints.py
│   │   ├── __init__.py
│   │   ├── labs_tab.py
│   │   └── pytodoc.py
│   ├── __main__.py
│   ├── main.py
│   ├── manual.py
│   ├── Model
│   │   ├── APImodels.py
│   │   ├── __init__.py
│   │   ├── model_family.py
│   │   ├── model_global.py
│   │   ├── ModelRegistry.py
│   │   └── templates.py
│   ├── native
│   │   ├── _core.c
│   │   ├── engine_helpers.py
│   │   ├── __init__.py
│   │   ├── pipeline_core.c
│   │   ├── pipeline_core.py
│   │   ├── rust_model.py
│   │   └── rust_model.rs
│   ├── pipelinebuilder
│   │   ├── aibuilder
│   │   │   ├── aibuilder_core.c
│   │   │   ├── aibuilder_core.rs
│   │   │   ├── context.py
│   │   │   ├── dialog.py
│   │   │   ├── engine_call.py
│   │   │   ├── __init__.py
│   │   │   └── planner.py
│   │   ├── blck_typ.py
│   │   ├── canvas.py
│   │   ├── editordialogue.py
│   │   ├── executionWorker.py
│   │   ├── execution_core.py
│   │   ├── examples
│   │   │   └── *.json
│   │   ├── flowpreview.py
│   │   ├── graph_ops.py
│   │   ├── __init__.py
│   │   ├── outrender.py
│   │   ├── pipblck.py
│   │   ├── pipebuilder.py
│   │   ├── pipefunctions.py
│   │   ├── pipe_global.py
│   │   └── validation.py
│   ├── Prefrences
│   │   ├── __init__.py
│   │   ├── ParallelLoading.py
│   │   └── prefrence_global.py
│   ├── Server
│   │   ├── hfauth.py
│   │   ├── hf_deps.py
│   │   ├── hfdwld.py
│   │   ├── __init__.py
│   │   ├── ollama_helpers.py
│   │   ├── server_global.py
│   │   └── ServerHandling.py
│   ├── skill
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   └── tab.py
│   └── UI
│       ├── buildUI.py
│       ├── effects.py
│       ├── icons.py
│       ├── __init__.py
│       ├── labs_tab.py
│       ├── mainwindow
│       │   ├── auto_setup.py
│       │   ├── chat_pipeline.py
│       │   ├── context_controls.py
│       │   ├── documents.py
│       │   ├── engine_runtime.py
│       │   ├── __init__.py
│       │   ├── labs.py
│       │   ├── models.py
│       │   ├── sessions.py
│       │   ├── shared.py
│       │   ├── status_view.py
│       │   ├── ui_build.py
│       │   └── window.py
│       ├── md_to_html.py
│       ├── Qt6widgets
│       │   ├── chatarea.py
│       │   ├── chatmodule.py
│       │   ├── __init__.py
│       │   ├── inputbar.py
│       │   ├── messagewidget.py
│       │   ├── refrencepanels.py
│       │   ├── sessionsidebar.py
│       │   └── thinkingblock.py
│       ├── qt_workers.py
│       ├── RichTextEditor.py
│       ├── tabs.py
│       ├── toggle.py
│       ├── UI_const.py
│       ├── UI_global.py
│       └── widgets.py
├── NativeLab.spec
├── .nojekyll
├── pyproject.toml
├── README.md
├── requirements.txt
├── robots.txt
├── scripts
│   └── download_svg_icons.py
├── SECURITY.md
├── setup.py
├── sitemap.xml
├── tests
│   ├── test_auto_setup.py
│   ├── test_context_meter.py
│   ├── test_hf_deps.py
│   ├── test_mainwindow_split.py
│   ├── test_native_helpers.py
│   ├── test_pipeline_canvas_ids.py
│   ├── test_pipeline_examples.py
│   ├── test_pipeline_native_core.py
│   └── test_qt_workers.py
├── uv.lock
├── .vscode
│   └── settings.json
└── web_page
    ├── compare.html
    ├── features.html
    ├── pipeline.html
    ├── setup.html
    ├── site.css
    └── site.js
```

---

## Integración multiplataforma

NativeLab y PhonoLab forman un ecosistema unificado y multiplataforma para desarrollo y uso de IA local:

### Vista general de la arquitectura

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          NativeLab Desktop                              │
│  • GUI PyQt6 con constructor de pipelines, labs y funciones avanzadas  │
│  • Cliente CLI para usuarios de terminal                                │
│  • Inferencia local con llama.cpp server o CLI                         │
│  • Integraciones API (OpenAI, Anthropic, Ollama)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          PhonoLab Android                               │
│  • Inferencia llama.cpp en el dispositivo con binarios incluidos       │
│  • Soporte para modelos de visión (LLaVA, Qwen-VL, etc.)              │
│  • Procesamiento RAG de documentos PDF, docs y archivos de texto      │
│  • Servidor API LAN para comunicación entre escritorio y móvil         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Puntos clave de integración

* **Descubrimiento de dispositivos LAN**: la pestaña “Devices” de NativeLab escanea automáticamente servidores API de PhonoLab en la misma red local
* **Autenticación inteligente**: intercambio y gestión automáticos de claves entre NativeLab y PhonoLab
* **Estándares API unificados**: ambas plataformas implementan estándares API de OpenAI y Anthropic para interoperabilidad sin fricción
* **Ecosistema de modelos compartido**: mismo formato GGUF y soporte de niveles de quantización en ambas plataformas
* **Flujos de trabajo multiplataforma**: crea pipelines que abarcan dispositivos de escritorio y móviles

### Arquitectura de integración

La integración sigue la arquitectura por capas de NativeLab:

* **Frontends**: la GUI y la CLI de NativeLab pueden usar PhonoLab como servidor API remoto
* **LabEndpoints**: la misma superficie `LabEndpoints` se usa tanto para inferencia local como remota
* **Backend centralizado**: `NativeLabBackend` soporta dispositivos PhonoLab como modelos API
* **Capa de motor**: `ApiEngine` maneja la comunicación con el servidor API de PhonoLab
* **Persistencia**: las configuraciones de dispositivos y API se almacenan en la misma capa de persistencia

Esta arquitectura asegura que los dispositivos PhonoLab aparezcan como ciudadanos de primera clase dentro del ecosistema NativeLab, con la misma experiencia de usuario y capacidades que los modelos locales e integraciones API.

---

## Modelo de hilos

* Toda la inferencia (tokens en streaming, summarization, etapas de pipeline, descargas, sondeos MCP) se ejecuta en subclases de `QThread` con señales de PyQt para actualizaciones entre hilos. El hilo principal nunca se bloquea.
* Los workers exponen `abort()`, que activa una bandera revisada en cada iteración para una cancelación limpia.
* `nativelab/UI/qt_workers.py` centraliza el apagado de workers, la desconexión de señales, el manejo de workers atascados y la limpieza segura antes de eliminar widgets.
* Los workers de resumen además soportan `request_pause()`, que escribe un snapshot de estado en `paused_jobs/` antes de salir.
* La CLI usa llamadas sincrónicas (`endpoints.call_llm`) porque no tiene una UI que deba mantenerse responsiva: el mismo backend, pero sin la plumbing de `QThread`.

---

## Limpieza de procesos huérfanos

Al cerrarse, `kill_stray_llama_servers()` termina los procesos huérfanos `llama-server` de sesiones anteriores que se hayan quedado colgadas, además de los que están gestionados actualmente. Esto evita fugas de puertos entre reinicios.
