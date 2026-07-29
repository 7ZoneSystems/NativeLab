# Integraciones

NativeLab expone una pequeña superficie de integración para bots, scripts locales, puentes webhook y workers en la nube. El objetivo es permitir que herramientas externas descubran lo que la app puede hacer sin importar la ventana principal.

## Dónde encontrarlo

Abre **Dev > Integrations** en la GUI. La página está dividida en subpestañas:

* **Endpoints**: selecciona una ruta, copia respuestas JSON e inicia el puente HTTP local.
* **Discord Bot**: crea perfiles reutilizables de conector de bot y guarda credenciales localmente.
* **WhatsApp Bot**: crea perfiles reutilizables de webhook de WhatsApp Cloud API y ejecútalos localmente.

La API de Python vive en `nativelab/integrations/`.

## Rutas de endpoints

El endpoint devuelve diccionarios compatibles con JSON en texto plano.

| Ruta                          | Método | Propósito                                                                                            |
| ----------------------------- | -----: | ---------------------------------------------------------------------------------------------------- |
| `/snapshot`                   |    GET | Catálogo completo: rutas, runtime, límites, modelos, modelos API, pipelines, Labs.                   |
| `/runtime`                    |    GET | Backend activo, nombre/ruta del modelo, tamaño de contexto, puerto del servidor, estado de carga.    |
| `/models`                     |    GET | Modelos GGUF, Ollama y HF registrados con metadatos de rol, backend, familia, cuantización y visión. |
| `/v1/models`                  |    GET | Lista de modelos compatible con OpenAI para pipelines guardados expuestos como `pipeline:<name>`.    |
| `/v1/chat/completions`        |   POST | Ruta compatible con OpenAI que ejecuta un pipeline guardado cuando `model` es `pipeline:<name>`.     |
| `/api_models`                 |    GET | Configuraciones guardadas de modelos API con las claves API ocultas.                                 |
| `/limits`                     |    GET | Valores predeterminados, roles, configuración de la app y metadatos de campos de configuración.      |
| `/pipelines`                  |    GET | Pipelines visuales guardados con conteos de bloques y conexiones.                                    |
| `/pipelines/{name}`           |    GET | Definición JSON sin procesar del pipeline guardado.                                                  |
| `/labs`                       |    GET | Funciones registradas de Labs y metadatos de integración.                                            |
| `/labs/py_to_doc`             |    GET | Metadatos de la ruta del laboratorio py-to-doc.                                                      |
| `/call_llm`                   |   POST | Envía un prompt/mensajes al motor activo de NativeLab.                                               |
| `/skills`                     |    GET | Biblioteca de skills de NativeLab, incluidas descripciones e instrucciones activas.                  |
| `/integrations/discord_bots`  |    GET | Perfiles guardados del conector de Discord con tokens ocultos.                                       |
| `/integrations/whatsapp_bots` |    GET | Perfiles guardados del conector de WhatsApp con tokens ocultos.                                      |

## Iniciar el endpoint HTTP local

En la pestaña **Integrations**, define un puerto y haz clic en **Start**. La URL predeterminada es:

```text
http://127.0.0.1:8765
```

Ejemplos de solicitud:

```bash
curl http://127.0.0.1:8765/runtime
curl http://127.0.0.1:8765/pipelines
curl http://127.0.0.1:8765/v1/models
curl http://127.0.0.1:8765/labs/py_to_doc
```

Llamar al modelo activo:

```bash
curl -X POST http://127.0.0.1:8765/call_llm \
  -H "content-type: application/json" \
  -d '{"prompt":"Write a short project status update.","n_predict":300}'
```

`/call_llm` acepta:

* `prompt`: cadena del prompt del usuario.
* `messages`: lista de mensajes estilo OpenAI, opcional si se proporciona `prompt`.
* `system_prompt`: mensaje de sistema opcional.
* `n_predict`: máximo de tokens de generación.
* `temperature`, `top_p`, `repeat_penalty`: controles de muestreo.

### Pipelines guardados como modelos API

Cada pipeline visual guardado también se expone como un identificador de modelo compatible con OpenAI:

```text
pipeline:<pipeline-name>
```

Listarlos:

```bash
curl http://127.0.0.1:8765/v1/models
```

Ejecutar uno mediante el ejecutor normal de pipelines:

```bash
curl -X POST http://127.0.0.1:8765/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "model": "pipeline:research-synthesis",
    "messages": [{"role": "user", "content": "Summarize these notes."}]
  }'
```

El endpoint no crea un runner separado. Carga el JSON del pipeline guardado y lo ejecuta mediante `PipelineExecutionWorker`, por lo que validación, comportamiento de bloques, llamadas a modelos, logs y manejo de errores coinciden con las rutas de pipeline de la GUI y la CLI.

## API de Python

Para integraciones en proceso, usa `IntegrationEndpoints` directamente:

```python
from nativelab.integrations import IntegrationEndpoints

endpoints = IntegrationEndpoints()
print(endpoints.handle("/snapshot"))
print(endpoints.to_json("/models"))
```

Cuando NativeLab enlaza el endpoint de integración dentro de la GUI, también puede enrutar trabajo a través de la app en vivo:

```python
text = endpoints.call_llm(prompt="Summarize the loaded model state.")
endpoints.request_context(8192)
endpoints.request_load_model("/path/to/model.gguf")
endpoints.request_unload()
```

Para exponerlo por HTTP, usa `IntegrationHttpEndpoint`:

```python
from nativelab.integrations import IntegrationEndpoints, IntegrationHttpEndpoint

endpoints = IntegrationEndpoints()
server = IntegrationHttpEndpoint(endpoints, port=8765)
server.start()
```

Dentro de la GUI esto ya está conectado al endpoint en vivo de NativeLab.

## Endpoint de Skills

Las skills se gestionan en **Dev > Skills** y se guardan en:

```text
localllm/skill/skills.json
```

NativeLab incluye una skill integrada `edit` para ediciones estructuradas de código. Indica a los modelos que inspeccionen la estructura del archivo, nombres de funciones, valores de retorno y variables, y que prefieran operaciones precisas en lugar de reescrituras de archivo completo.

La ruta `/skills` expone la misma biblioteca a las integraciones. Las skills activas solo se inyectan en las llamadas al modelo cuando el interruptor **Skills** del chat está habilitado. Como Discord, WhatsApp, Labs y `/call_llm` de HTTP local enrutan a través del endpoint compartido de NativeLab, heredan el mismo contexto de skills mientras ese interruptor esté activado.

## Conector del bot de Discord

La subpestaña Discord Bot guarda perfiles reutilizables del conector en:

```text
localllm/integrations/discord_bots.json
```

Crea `bot1`, guárdalo, y luego crea `bot2` del mismo modo. Cada perfil almacena:

* token del bot, application ID y guild ID opcional
* URL del endpoint de NativeLab
* comportamiento de respuesta, incluidas respuestas efímeras y respuestas por mención directa `@Bot`
* prompt de sistema editable, con el prompt de Discord de NativeLab mantenido como preset
* configuración de cola de solicitudes
* permisos de Discord requeridos por el bot
* controles de acceso de NativeLab para comandos de modelo, runtime, pipeline, lab y model-list

El archivo del bot en tiempo de ejecución se guarda en:

```text
nativelab/integrations/examples/discord_bot.py
```

Lee el perfil guardado y expone comandos slash:

* `/help`: muestra los comandos habilitados para este perfil.
* `/ask`: envía el prompt a `POST /call_llm`.
* `/status`: lee `/runtime`.
* `/pipelines` y `/pipeline`: leen metadatos de pipelines guardados.
* `/labs` y `/lab`: leen metadatos de rutas de Labs como `/labs/py_to_doc`.
* `/models`: lee las rutas de catálogo de modelos locales y API.
* `@Bot <message>`: modo opcional de respuesta por mención directa, habilitado por perfil guardado.

Configuración:

```bash
python -m pip install discord.py aiohttp
```

Las instalaciones empaquetadas incluyen estas dependencias. Al ejecutar desde un clon nuevo, instálalas con el comando anterior o con `python -m pip install -e .`.

Inicia NativeLab, abre **Integrations > Endpoints**, pulsa **Start**, y luego abre **Discord Bot** y guarda un perfil. Usa **Start Bot** / **Stop Bot** en esa subpestaña para ejecutar el perfil seleccionado desde la app. El panel **Bot Logs** muestra el arranque, la conexión con Discord, llamadas al endpoint, actividad de cola y errores de comandos.

También puedes ejecutar el mismo perfil guardado desde una terminal:

```bash
export DISCORD_BOT_PROFILE="bot1"
python nativelab/integrations/examples/discord_bot.py
```

Puedes sobrescribir el token o el endpoint para una ejecución sin cambiar el perfil guardado:

```bash
export DISCORD_BOT_TOKEN="your-token"
export NATIVELAB_INTEGRATION_URL="http://127.0.0.1:8765"
```

## Conector del bot de WhatsApp

La subpestaña WhatsApp Bot guarda perfiles reutilizables de WhatsApp Cloud API en:

```text
localllm/integrations/whatsapp_bots.json
```

Cada perfil almacena el token de acceso de Meta, el ID del número de teléfono, el ID opcional de la cuenta de empresa, el verify token, el host/puerto/ruta local del webhook, límites de cola, límites de respuestas, el prompt de sistema editable y los controles de acceso de NativeLab.

El archivo del webhook en tiempo de ejecución se guarda en:

```text
nativelab/integrations/examples/whatsapp_bot.py
```

El bot expone estos comandos de texto de WhatsApp:

* `/help`: muestra los comandos habilitados para este perfil.
* `/ask <prompt>`: envía el prompt a `POST /call_llm`.
* `/status`: lee `/runtime`.
* `/pipelines` y `/pipeline <name>`: leen metadatos de pipelines guardados.
* `/labs` y `/lab <name>`: leen metadatos de rutas de Labs como `/labs/py_to_doc`.
* `/models`: lee las rutas de catálogo de modelos locales y API.
* mensajes de texto normales: modo opcional de pregunta directa, habilitado por perfil guardado.

Configuración:

```bash
python -m pip install aiohttp
```

Las instalaciones empaquetadas incluyen esta dependencia. Al ejecutar desde un clon nuevo, instálala con el comando anterior o con `python -m pip install -e .`.

Inicia NativeLab, abre **Integrations > Endpoints**, pulsa **Start**, y luego abre **WhatsApp Bot** y guarda un perfil. Usa **Start Bot** / **Stop Bot** en esa subpestaña para ejecutar el webhook seleccionado desde dentro de la app. El panel **Bot Logs** muestra el arranque, la verificación del webhook, mensajes entrantes, llamadas al endpoint, actividad de cola y errores de envío.

Meta requiere una URL pública HTTPS de callback. Para desarrollo local, expón la URL local de callback del perfil, por ejemplo `http://127.0.0.1:8770/webhook`, con un túnel como ngrok o cloudflared. Coloca la URL pública HTTPS en el campo de callback del webhook de Meta y usa el verify token del perfil.

También puedes ejecutar el mismo perfil guardado desde una terminal:

```bash
export WHATSAPP_BOT_PROFILE="whatsapp1"
python nativelab/integrations/examples/whatsapp_bot.py
```

Puedes sobrescribir las credenciales o el endpoint para una ejecución sin cambiar el perfil guardado:

```bash
export WHATSAPP_ACCESS_TOKEN="your-token"
export WHATSAPP_PHONE_NUMBER_ID="your-phone-number-id"
export NATIVELAB_INTEGRATION_URL="http://127.0.0.1:8765"
```

## Notas de seguridad

* El endpoint HTTP integrado se enlaza solo a `127.0.0.1`.
* Las claves API se ocultan en `/api_models`.
* Los tokens del bot de Discord se guardan localmente en `localllm/integrations/discord_bots.json`.
* Los tokens de acceso de WhatsApp se guardan localmente en `localllm/integrations/whatsapp_bots.json`.
* Las respuestas por mención directa requieren que el Message Content Intent esté activado en el Developer Portal de Discord para esa aplicación de bot.
* Los webhooks de WhatsApp necesitan un túnel HTTPS público para la entrega del callback por parte de Meta.
* No expongas el endpoint directamente a Internet.
* Trata `/call_llm` como una capacidad local de confianza, porque puede enviar prompts al modelo o backend API que esté cargado en NativeLab.
