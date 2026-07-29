# CLI - `nativelab --cli`

NativeLab ahora cuenta con un centro de control de terminal de primera clase. Utiliza los mismos motores, registros de modelos, perfiles de API, endpoints de Labs, skills, pipelines y perfiles de integración que la GUI, pero mantiene todo automatizable desde una terminal.

La primera ejecución sigue iniciando el proceso de configuración cuando no existen preferencias de la CLI. Después de la configuración, `nativelab --cli` abre el menú interactivo:

```text
Chat
Modelos
Modelos API
Skills
Labs
Pipelines
Integraciones
Estado
Configuración
Salir
```

Para una guía paso a paso para principiantes, consulta
[`nativelab/cli/cli_guide.md`](../nativelab/cli/cli_guide.md).

## Comandos directos

```bash
nativelab --cli                         # configuración si es necesario, luego menú
nativelab --cli setup [--reset]         # asistente de configuración inicial
nativelab --cli chat [--model PATH_OR_REF] [--ctx N] [--system TEXT]
nativelab --cli status
nativelab --cli lint <file.py> ...

nativelab --cli models [list|load|default] [target] [--ctx N] [--json]
nativelab --cli api-models [list|load|default] [target] [--json]
nativelab --cli skills [list|create|edit|delete|enable|disable|chat-on|chat-off] [name] [--json]

nativelab --cli labs list [--json]
nativelab --cli labs code-edit [--file path] [--prompt text] [--save] [--save-as path] [--edit-response] [--no-diff]
nativelab --cli labs py-to-doc --mode single --file path.py --out-dir docs/generated
nativelab --cli labs py-to-doc --mode queue --file a.py --file b.py
nativelab --cli labs py-to-doc --mode project --project ./src --out-dir docs/generated
nativelab --cli labs py-to-doc --mode project --project ./src --out-dir docs/generated --resume
nativelab --cli labs py-to-doc --mode project --project ./src --context-policy auto --context-budget 8192

nativelab --cli pipeline list [--json]
nativelab --cli pipeline show <name> [--json]
nativelab --cli pipeline run <name> [--text text | --file input.txt]

nativelab --cli endpoint [/snapshot|/runtime|/models|/api_models|/limits|/pipelines|/labs|/skills] [--json]
nativelab --cli serve --port 8765

nativelab --cli integrations list [--json]
nativelab --cli integrations discord list|show|create|edit|delete|run [name] [--json]
nativelab --cli integrations whatsapp list|show|create|edit|delete|run [name] [--json]
```

`--json` está disponible en los comandos de inspección, listado y visualización para que los scripts de shell puedan consumir el estado de NativeLab sin necesidad de analizar tablas.

## Comandos Slash del Chat

Dentro de `nativelab --cli chat`, los mensajes normales se envían al modelo cargado. Las líneas que comienzan con `/` controlan el entorno de ejecución compartido:

| Comando                                                | Efecto                                                                                       |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| `/help`                                                | Muestra la lista de comandos.                                                                |
| `/status`                                              | Muestra el backend, el modelo cargado, el contexto (ctx) y el estado de inyección de skills. |
| `/models`                                              | Lista los modelos GGUF, Ollama y Hugging Face registrados.                                   |
| `/api-models`                                          | Lista los perfiles de modelos API y sus referencias guardadas.                               |
| `/load <path\|ollama:name\|hf:repo-or-dir\|@api/name>` | Carga un perfil de modelo local, backend o API.                                              |
| `/unload`                                              | Descarga el modelo o perfil API activo.                                                      |
| `/ctx <n>`                                             | Cambia el contexto del modelo local y lo recarga.                                            |
| `/skills on\|off\|list`                                | Activa, desactiva o inspecciona la inyección de skills para el chat.                         |
| `/pipelines`                                           | Lista los pipelines visuales guardados.                                                      |
| `/pipeline <name>`                                     | Muestra un pipeline guardado.                                                                |
| `/pipeline run <name> [text]`                          | Ejecuta un pipeline guardado con texto de entrada.                                           |
| `/labs`                                                | Lista las rutas disponibles de Labs.                                                         |
| `/py-to-doc <file.py> [...]`                           | Genera documentación para uno o varios archivos Python.                                      |
| `/py-to-doc project <dir>`                             | Genera documentación de un proyecto con soporte de checkpoints y reanudación.                |
| `/code-edit [file] -- <request>`                       | Ejecuta el Lab de edición estructurada y muestra un diff.                                    |
| `/endpoint <route>`                                    | Inspecciona una ruta de endpoint de integración.                                             |
| `/serve [port]`                                        | Inicia el endpoint de integración hasta presionar Ctrl+C.                                    |
| `/system <text>`                                       | Define el prompt del sistema actual.                                                         |
| `/reset`                                               | Borra el historial de la conversación.                                                       |
| `/lint <file...>`                                      | Ejecuta linting de archivos Python.                                                          |
| `/save <file>`                                         | Guarda la conversación en formato JSON.                                                      |
| `/quit`                                                | Sale del chat.                                                                               |

Las referencias a archivos siguen funcionando dentro de los mensajes normales del chat:

```text
Explain @nativelab/labs/endpoints.py in simple terms.
```

La CLI incrusta el contenido de archivos de texto legibles dentro del prompt y limita cada archivo a aproximadamente 60 KB.

## Modelos y Modelos API

Los modelos locales y los modelos API comparten el mismo selector y entorno de ejecución. Puedes utilizar referencias API en cualquier lugar donde se acepte un objetivo de modelo:

```bash
nativelab --cli api-models list
nativelab --cli api-models load "@api/grok4.1%28no%20reasoning%29"
nativelab --cli models default /path/to/model.gguf --ctx 8192
```

Los valores predeterminados se almacenan en `localllm/cli_prefs.json`. Las claves API aparecen ocultas en todos los comandos de visualización.

## Skills

Las skills se almacenan en `localllm/skill/skills.json`. La CLI garantiza que exista la skill integrada `edit`, permite crear, editar y eliminar skills, y guarda el estado del interruptor de skills para el chat en `localllm/cli_prefs.json`.

```bash
nativelab --cli skills list
nativelab --cli skills create refactor-helper
nativelab --cli skills enable edit
nativelab --cli skills chat-on
```

Cuando las skills del chat están activadas, los nombres, descripciones e instrucciones de las skills activas se inyectan mediante el endpoint compartido de Labs. Las integraciones que utilizan el mismo endpoint LLM también pueden beneficiarse indirectamente de ese contexto.

## Labs

`code-edit` utiliza las mismas operaciones de edición estructurada que el Lab de la GUI. Escribe el código temporal actual en `localllm/temp_code_edit_file` y el historial de edición en `localllm/temp_code_edit.json`.

```bash
nativelab --cli labs code-edit --file app.py --prompt "add input validation"
nativelab --cli labs code-edit --prompt "create a tiny Flask app"
nativelab --cli labs code-edit --file app.py --prompt "simplify parser" --edit-response --save-as app_v2.py
```

`--edit-response` escribe el JSON sin procesar generado por el modelo en `localllm/temp_code_edit_response.json` y abre `$EDITOR` antes de aplicar las operaciones.

`py-to-doc` admite los modos **single file**, **queue** y **project**. El modo **project** utiliza el mismo worker de la GUI y sus archivos de checkpoint/reanudación ubicados en `localllm/temp`, permitiendo continuar tras un reinicio desde los archivos o funciones ya procesados. Este modo recorre los subdirectorios, ignora las rutas definidas en el `.gitignore` de la raíz del proyecto y crea previamente la misma estructura de directorios de salida antes de generar la documentación.

Utiliza `--resume` para exigir un checkpoint existente compatible en lugar de iniciar una ejecución nueva.

El contexto persistente puede controlarse mediante `--context-policy none|fixed|auto`.

El modo `auto` utiliza un presupuesto aproximado de tokens (`--context-budget`) únicamente para el historial de conversación que mantiene `py-to-doc`. Si una sección supera ese presupuesto, esa sección se completa y la siguiente clase, función o archivo comienza con un contexto limpio.

Para modelos GGUF locales, el modo automático recarga el modelo o servidor local con el presupuesto seleccionado para que `py-to-doc` y `llama.cpp` utilicen la misma ventana de contexto.

Añade `--auto-model-reload --reload-free-ram-gb N --reload-free-ram-mb N` para reiniciar automáticamente el modelo local activo al finalizar la sección actual cuando la memoria RAM disponible caiga por debajo del umbral indicado. La recarga se ejecuta antes de comenzar la siguiente sección del LLM.

## Pipelines

La CLI puede inspeccionar y ejecutar pipelines creados desde la GUI:

```bash
nativelab --cli pipeline list
nativelab --cli pipeline show direct
cat prompt.txt | nativelab --cli pipeline run direct
```

La creación de pipelines, la edición de ejemplos y el AI Pipeline Builder siguen siendo exclusivos de la GUI en esta versión. La CLI utiliza los mismos archivos JSON guardados y el mismo flujo de validación al ejecutar un pipeline.

## Integraciones

El explorador de endpoints expone el estado de NativeLab para bots y scripts externos:

```bash
nativelab --cli endpoint /snapshot --json
nativelab --cli endpoint /skills --json
nativelab --cli serve --port 8765
```

Los comandos de perfiles de Discord y WhatsApp utilizan los mismos archivos JSON que la GUI:

```bash
nativelab --cli integrations discord create bot1
nativelab --cli integrations discord run bot1
nativelab --cli integrations whatsapp create wa1
nativelab --cli integrations whatsapp run wa1
```

Las ejecuciones de bots en primer plano muestran registros visibles y se detienen con **Ctrl+C**.

## Archivos generados

| Ruta                                    | Propósito                                                                     |
| --------------------------------------- | ----------------------------------------------------------------------------- |
| `localllm/cli_prefs.json`               | Modelo predeterminado de la CLI, contexto y estado del interruptor de skills. |
| `localllm/custom_models.json`           | Rutas de modelos locales adicionales.                                         |
| `localllm/model_configs.json`           | Roles de modelos y configuración de ejecución.                                |
| `localllm/api_models.json`              | Perfiles de modelos API guardados.                                            |
| `localllm/skill/skills.json`            | Biblioteca compartida de skills para modelos.                                 |
| `localllm/temp_code_edit.json`          | Historial del Lab Code Edit.                                                  |
| `localllm/temp_code_edit_response.json` | Última respuesta JSON editable del modelo de Code Edit.                       |
| `localllm/temp_code_edit_file`          | Archivo de trabajo del Lab Code Edit.                                         |
| `localllm/temp/`                        | Checkpoints de proyectos generados por py-to-doc.                             |
| `localllm/integrations/*.json`          | Perfiles de conectores de Discord y WhatsApp.                                 |

## Notas para desarrolladores

El entorno de ejecución de la CLI se encuentra en [`nativelab/cli/runtime.py`](../nativelab/cli/runtime.py).

Construye un único par compartido de `LabEndpoints` e `IntegrationEndpoints` y los enlaza con el motor local o API cargado. Los comandos de Argparse, el menú interactivo, el chat REPL, Labs, los pipelines y `serve` reutilizan el mismo entorno de ejecución.
