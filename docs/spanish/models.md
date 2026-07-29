# Modelos

Todo lo relacionado con la capa de modelos de NativeLab: registros, plantillas de prompt, cuantizaciones, roles y proveedores API.

---

## Registro de modelos

`ModelRegistry` (en `nativelab/Model/ModelRegistry.py`) mantiene dos fuentes de verdad y las fusiona:

* Modelos autodetectados en `MODELS_DIR` (`./localllm/*.gguf`).
* Modelos añadidos manualmente desde la pestaña **Models** de la GUI o mediante el asistente de la CLI.

Ambos conjuntos se guardan en `custom_models.json` (rutas) y `model_configs.json` (parámetros por modelo).

```python
from nativelab.Model.model_global import get_model_registry

reg = get_model_registry()
reg.add("/path/to/some.gguf")
reg.all_models()        # → [{"path", "name", "size_mb", "family", "quant", "role", …}]
reg.get_config(path)    # → ModelConfig dataclass
```

---

## Roles

Cada modelo tiene un solo rol. El rol determina en qué ranura del motor entra e influye en el enrutamiento de prompts.

| Rol             | Icono | Uso                                                                         |
| --------------- | ----- | --------------------------------------------------------------------------- |
| `general`       | 💬    | Motor principal de chat.                                                    |
| `reasoning`     | 🧠    | Información arquitectónica en el modo pipeline; pase final de resumen.      |
| `summarization` | 📄    | Resumidor dedicado de PDF.                                                  |
| `coding`        | 💻    | Recibe prompts detectados automáticamente como solicitudes de programación. |
| `secondary`     | 🔀    | Proveedor auxiliar de contexto en ejecuciones con múltiples motores.        |

---

## Parámetros por modelo

Se almacenan en `model_configs.json` por ruta absoluta:

```python
threads        # hilos de CPU pasados a llama.cpp
ctx            # tamaño de ventana de contexto en tokens
temperature    # 0.0–2.0
top_p          # muestreo nucleus
repeat_penalty # 1.0 = desactivado
n_predict      # máximo de tokens generados por llamada
family         # clave de familia detectada automáticamente (mistral, llama3, qwen, …)
```

Se pueden editar por archivo desde la pestaña **Models** de la GUI. Los valores predeterminados provienen de `app_config.json`.

---

## Detección automática de familias

`detect_model_family()` compara substrings del nombre del archivo en orden de prioridad para elegir la plantilla `ModelFamily` correcta. `FAMILY_TEMPLATES` (en `nativelab/Model/templates.py`) incluye más de 20 familias: cada una contiene BOS/EOS, slot de sistema, prefijos/sufijos de usuario y asistente, y tokens de parada para el formato de chat de esa familia.

Cuando la biblioteca auxiliar opcional en Rust está disponible, NativeLab puede usarla para las rutas críticas de detección de familia y cuantización. Si no está presente, se usa automáticamente el matcher existente en Python.

Algunos ejemplos:

| Familia                       | Plantilla usuario / asistente                                          | BOS / EOS                                       |
| ----------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------- |
| **DeepSeek**                  | `User: {u}\n\nAssistant:`                                              | `<｜begin▁of▁sentence｜>` / `<｜end▁of▁sentence｜>` |
| **DeepSeek-R1**               | añade `<think>\n` después del prefijo del asistente                    | igual que DeepSeek                              |
| **Mistral / Mixtral**         | `[INST] {u} [/INST]`                                                   | `<s>` / `</s>`                                  |
| **LLaMA-2**                   | `[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{u} [/INST]`                    | `<s>` / `</s>`                                  |
| **LLaMA-3**                   | `<\|start_header_id\|>user<\|end_header_id\|>\n\n{u}<\|eot_id\|>`      | `<\|begin_of_text\|>`                           |
| **Phi / Phi-3**               | `<\|user\|>\n{u}<\|end\|>\n<\|assistant\|>\n`                          | -                                               |
| **Qwen / ChatML / Yi / Orca** | `<\|im_start\|>user\n{u}<\|im_end\|>\n<\|im_start\|>assistant\n`       | -                                               |
| **Gemma**                     | `<start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model\n`        | -                                               |
| **Command-R**                 | `<\|START_OF_TURN_TOKEN\|><\|USER_TOKEN\|>{u}<\|END_OF_TURN_TOKEN\|>…` | -                                               |

Lista completa: DeepSeek (+R1), Mistral, Mixtral, LLaMA-2, LLaMA-3, Phi, Phi-3, Phi-3.5, Qwen, ChatML, Gemma, CodeLlama, Falcon, Vicuna, OpenChat, Neural-Chat, Starling, Yi, Zephyr, Solar, Orca y Command-R.

La familia detectada aparece junto a cada modelo en la pestaña **Models** y en la salida `/status` de la CLI.

---

## Detección de cuantización

`detect_quant_type()` reconoce todas las cuantizaciones presentes en las compilaciones actuales de llama.cpp.

**Quantizaciones imatrix importance** — `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ2_M`, `IQ3_XXS`, `IQ3_XS`, `IQ3_S`, `IQ3_M`, `IQ4_XS`, `IQ4_NL`.

**K-quants** — `Q2_K`, `Q3_K_S/M/L`, `Q4_K_S/M`, `Q5_K_S/M`, `Q6_K`.

**Legadas** — `Q4_0/1`, `Q5_0/1`, `Q8_0`.

**Flotantes** — `F16`, `F32`, `BF16`.

Cada cuantización se asigna a un nivel de calidad con una etiqueta de color:

| Nivel              | Cuantizaciones         | Etiqueta                           |
| ------------------ | ---------------------- | ---------------------------------- |
| 🟢 Full            | F32, F16, BF16, Q8, Q6 | "Full precision" / "Near-lossless" |
| 🟣 High quality    | Q5, IQ4                | "High quality"                     |
| 🟡 Balanced        | Q4, IQ3                | "Balanced"                         |
| 🔴 Compressed      | Q3, Q2                 | "Compressed"                       |
| 🔴 Very compressed | Q2, IQ1                | "Very compressed"                  |

---

## Soporte para backend local y API

Además de los GGUF locales, NativeLab puede usar modelos Ollama en ejecución, snapshots opcionales de Hugging Face Transformers y dos formatos API. Todos aparecen en el mismo selector de modelos y usan la superficie compartida de motor/estado.

### Referencias de backend local

| Backend         | Formato de referencia        | Notas                                                                                                                                                                                                                        |
| --------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| llama.cpp GGUF  | ruta del sistema de archivos | El comportamiento existente de `.gguf` no cambia.                                                                                                                                                                            |
| Ollama          | `ollama:<model-name>`        | Requiere un servicio Ollama ya en ejecución en `http://127.0.0.1:11434`.                                                                                                                                                     |
| HF Transformers | `hf:<repo-id-or-local-dir>`  | Instala las bibliotecas de runtime desde la acción **Install Libraries** de la pestaña Download, o ejecuta el comando `pip` mostrado allí; admite directorios/repos con modelos safetensors y metadatos de config/tokenizer. |

La pestaña **Models** incluye botones para añadir modelos Ollama ya instalados y referencias HF de repositorio o directorio local. Los modelos HF de visión usan la ruta `image-text-to-text` de Transformers cuando la arquitectura del modelo la soporta.

### Soporte en la pestaña Download

La pestaña Download ahora tiene tres rutas de modelo/runtime:

* **Búsqueda GGUF en HuggingFace** descarga un archivo `.gguf` dentro de `localllm/` para llama.cpp.
* **HF Transformers snapshot** inspecciona una revisión del repo, puede instalar o comprobar las bibliotecas Python necesarias desde dentro de la app, selecciona archivos de runtime, conserva subdirectorios del repo, descarga en `localllm/hf_transformers/<namespace>/<repo>/`, reanuda con archivos `.part` y registra el resultado como `hf:<local-folder>`.
* **Ollama model pull** se conecta al host Ollama configurado, lista los modelos instalados desde `/api/tags`, transmite el progreso de `/api/pull` y registra las descargas completadas como `ollama:<model>`.

Los selectores integrados **Popular** están definidos en `POPULAR_MODEL_PRESETS` dentro de `nativelab/Model/templates.py`, agrupados como `gguf`, `hf_transformers` y `ollama`. Rellenan el campo correspondiente de repo o modelo, pero no bloquean IDs personalizados.

Para repos privados o restringidos, inicia sesión desde **Settings > Accounts > Hugging Face** con **Login with Hugging Face**. NativeLab usa su propio cliente OAuth público, guarda las credenciales localmente en `localllm/cred/huggingface.json`, oculta los tokens en la interfaz y en los logs, y utiliza ese token para la búsqueda/descarga GGUF, las descargas de snapshots HF y la carga de modelos `hf:`. La entrada manual de un token `hf_...` sigue estando disponible como alternativa avanzada, y el campo `hf_token` de App Configuration sigue existiendo como respaldo de menor prioridad cuando no hay una sesión guardada. Algunos repos restringidos, incluidos los repos de Meta Llama, también requieren aceptar términos o esperar aprobación en la página del repo de Hugging Face; un token autenticado seguirá recibiendo HTTP 403 hasta que se otorgue acceso a ese repo.

### Ajustes de HF y Ollama

Abre el botón **Settings** en la esquina superior derecha y luego **Hugging Face** u **Ollama** para editar el comportamiento de carga del backend:

| Grupo de ajustes | Campos                                                                                                                                                                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HF Transformers  | directorio de descarga/caché, token, revisión por defecto, trust remote code, local-files-only, política de safetensors, dtype de torch, device map, low CPU memory loading, attention implementation, max-memory map, modo de cuantización |
| Ollama           | URL del host y `keep_alive` por defecto                                                                                                                                                                                                     |

`LlamaEngine` lee estos ajustes antes de cargar referencias `hf:` o `ollama:`, por lo que el chat, Labs, pipelines, integraciones y el estado visible de la CLI permanecen coherentes. Los modos de cuantización HF de 8 bits y 4 bits requieren una instalación compatible de `bitsandbytes`; de lo contrario, deja la cuantización en `none`.

Las referencias y descargas Ollama requieren que el daemon de Ollama ya esté en ejecución. Si NativeLab muestra un mensaje de conexión rechazada, inicia la app de escritorio de Ollama o ejecuta `ollama serve` y vuelve a intentarlo con el host configurado.

### Formatos

| Formato     | Endpoint            | Cabecera de autenticación |
| ----------- | ------------------- | ------------------------- |
| `openai`    | `/chat/completions` | `Authorization: Bearer …` |
| `anthropic` | `/v1/messages`      | `x-api-key: …`            |

Cualquier servidor autoalojado que exponga `/chat/completions` compatible con OpenAI (LM Studio, Ollama, vLLM, llama-cpp-python) funciona con el formato `openai`.

### Campos de `ApiConfig`

```python
name             # nombre visible en la UI
provider         # "OpenAI", "Anthropic" o tu nombre personalizado
model_id         # por ejemplo "gpt-4o-mini", "claude-3-5-sonnet-20241022"
api_key          # bearer o x-api-key
base_url         # por ejemplo "https://api.openai.com/v1"
api_format       # "openai" | "anthropic"
max_tokens       # pista de contexto; el chat API normal no lo usa como límite de salida
temperature      # 0.0–2.0
use_custom_prompt, system_prompt, user_prefix, user_suffix, assistant_prefix
```

Se almacenan en `api_models.json` y se gestionan mediante `ApiRegistry`.

### `ApiEngine`

`ApiEngine` reproduce la superficie pública de `LlamaEngine` (`load`, `create_worker`, `is_loaded`, `status_text`, `shutdown`), de modo que los pipelines, el resumen y la inyección de referencias funcionen igual sin importar qué motor esté activo.

`LabEndpoints.active_engine()` devuelve el motor API cuando está cargado; si no, devuelve el local.

---

## Añadir una nueva familia de modelos

Añade una entrada a `FAMILY_TEMPLATES` (`nativelab/Model/templates.py`) y un patrón correspondiente en `detect_model_family()`. Los patrones se evalúan en orden, así que coloca los más específicos (por ejemplo `phi-3.5`) antes que los más generales (`phi`).

```python
FAMILY_TEMPLATES["myfamily"] = ModelFamily(
    name="MyFamily",
    family="myfamily",
    user_prefix="<|user|>\n",
    user_suffix="<|end|>\n<|assistant|>\n",
    assistant_prefix="",
    assistant_suffix="<|end|>\n",
    bos="",
    eos="<|end|>",
    stop_tokens=["<|end|>", "<|user|>"],
)
```

Después añade el patrón a la lista:

```python
("myfamily", FAMILY_TEMPLATES["myfamily"]),
```

La detección incorporará la nueva familia en el siguiente inicio.
