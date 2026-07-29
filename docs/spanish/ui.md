# Interfaz de Usuario (UI)

Recorrido por la interfaz gráfica, temas, persistencia, atajos de teclado y gestión dinámica de la memoria RAM.

---

## Pestañas

| Pestaña                                 | Para qué sirve                                                                                                                         |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **💬 Chat**                             | Ventana principal de conversación. Barra lateral con sesiones, burbujas de mensajes y barra de entrada con selector de modelos.        |
| **📚 Models**                           | Biblioteca de modelos GGUF. Permite asignar roles, editar parámetros específicos de cada modelo y buscar nuevos archivos.              |
| **🌐 API Models**                       | Configuración de endpoints OpenAI, Anthropic y servidores autoalojados.                                                                |
| **⬇️ Download**                         | Descarga de modelos GGUF desde Hugging Face, snapshots de HF Transformers, descargas de Ollama e instalador de versiones de llama.cpp. |
| **Settings (esquina superior derecha)** | Ventana lateral de configuración con las páginas General, Docs, Hugging Face, Ollama, Server, Appearance y Accounts.                   |
| **Dev**                                 | Oculta hasta habilitar **Settings > General > Developer Mode**. Contiene Labs, Logs, API Server, Integrations, Pipeline, MCP y Skills. |

La subpestaña **Dev > API Server** puede alojar el motor activo de NativeLab, un modelo registrado o un pipeline visual guardado seleccionado desde **Hosted model**.

Las entradas de pipelines utilizan identificadores `pipeline:<name>` y se ejecutan mediante el ejecutor estándar de pipelines.

---

## Interfaz del Pipeline Builder

La página **Pipeline**, dentro de **Dev**, utiliza un diseño de tres paneles:

* **Barra lateral izquierda:** ejemplos, botones para añadir bloques, lista de modelos y controles para guardar, cargar y previsualizar.
* **Centro:** lienzo desplazable con edición de bloques, dibujo de conexiones y desplazamiento mediante arrastre sobre zonas vacías.
* **Barra lateral derecha:** pestañas **Execution** y **AI Builder**.

Las barras laterales izquierda y derecha pueden redimensionarse.

Si alguna se reduce por debajo de su tamaño mínimo útil, se retrae automáticamente y aparece un botón circular para volver a abrirla en el borde central del lienzo, evitando que desaparezca permanentemente.

El texto y los controles se ajustan dinámicamente al ancho disponible.

Cuando la barra derecha es estrecha, la pestaña **AI Builder** cambia automáticamente a un diseño compacto con controles apilados.

Consulta [pipeline-builder.md](pipeline-builder.md) para obtener información detallada sobre su funcionamiento.

---

## Componentes del chat

### `ChatArea` y `MessageWidget`

Un `QScrollArea` que contiene una pila vertical de instancias de `MessageWidget`.

Cada widget incorpora un `RichTextEdit` (subclase de `QTextBrowser`) que convierte Markdown a HTML mediante un renderizador personalizado `_md_to_html()` (mejorado en el commit `65efa045` con:

* Mejor formato para bloques de código.
* Representación consistente de elementos en línea.
* Correcciones de diseño específicas para Android.

Incluye soporte para:

* Bloques de código delimitados → tabla de dos filas (barra superior con el lenguaje, número de líneas y enlace **⧉ Copy**, seguida del bloque de código con fuente monoespaciada y resaltado de sintaxis).
* Código en línea.
* Encabezados.
* Texto en negrita y cursiva.
* Líneas horizontales.
* Listas con viñetas.
* Listas numeradas.

Los enlaces **Copy** utilizan anclas `copy://BLOCK_ID`.

Al hacer clic, `mousePressEvent` de `RichTextEdit` busca el código original en un diccionario interno y lo copia al portapapeles.

Los mensajes largos (más de 260 px) muestran automáticamente el botón desplegable:

**▼ Show more / ▲ Show less**

---

### `ThinkingBlock`

Durante la generación de resúmenes aparece un widget desplegable sobre la burbuja del resumen mostrando un registro en tiempo real con la vista previa de entrada y la salida de cada sección.

Al finalizar el proceso cambia a color verde e incluye el icono ✅.

---

### `InputBar`

Incluye:

* Selector de modelo (muestra familia, cuantización y nivel de calidad).
* Campo de texto multilínea (Enter envía; Shift+Enter inserta un salto de línea).
* Interruptor de modo código.
* Indicador del modo pipeline.
* Botones **Send** y **Stop**.

---

### `SessionSidebar`

Las sesiones se agrupan por fecha.

Incluye:

* Búsqueda incremental.
* Sesión activa resaltada en negrita y color púrpura.
* Menú contextual (clic derecho) para:

  * Cambiar el nombre.
  * Exportar en Markdown.
  * Eliminar la sesión.

---

## Temas

La aplicación incluye temas claro y oscuro.

Puedes cambiar entre ellos desde:

**View → Toggle Theme**

o mediante:

`Ctrl+T`

El tema activo se guarda en `app_config.json`.

Cuando cambia el tema, todas las pestañas se reconstruyen completamente en una única operación síncrona para actualizar correctamente todos los colores integrados (tarjetas con degradados, resaltado de sintaxis, etc.).

### Paletas personalizadas

La página **Settings > Appearance** permite modificar todos los colores de la interfaz.

Haz clic sobre cualquier muestra de color para abrir el selector del sistema; los cambios se aplican inmediatamente.

Las paletas clara y oscura se almacenan por separado como:

* `custom_light_palette`
* `custom_dark_palette`

dentro de `app_config.json`, por lo que personalizar una no afecta a la otra.

---

## Atajos de teclado

| Atajo         | Acción                                         |
| ------------- | ---------------------------------------------- |
| `Ctrl+N`      | Nueva sesión                                   |
| `Ctrl+Q`      | Salir                                          |
| `Ctrl+B`      | Mostrar u ocultar la barra lateral de sesiones |
| `Ctrl+L`      | Cambiar a **Dev > Logs**                       |
| `Ctrl+M`      | Cambiar a la pestaña **Models**                |
| `Ctrl+T`      | Alternar entre tema claro y oscuro             |
| `Enter`       | Enviar mensaje                                 |
| `Shift+Enter` | Insertar un salto de línea                     |

El menú **File** permite exportar sesiones en formato JSON, Markdown o texto plano.

El menú **Model** dispone de una opción para recargar el modelo con un solo clic.

El menú **View** agrupa los accesos relacionados con la navegación entre pestañas y barras laterales.

---

## Barra de estado

Situada en la parte inferior de la ventana.

Muestra:

* Modelo actual, familia y cuantización.
* Estado del motor activo (puerto del servidor, modo CLI o API).
* Indicador en tiempo real del uso de RAM (cuando `psutil` está disponible).
* Barra de uso del contexto procedente del medidor de contexto centralizado.

El chat, Labs, el editor de código, los pipelines y otros flujos de trabajo con LLM pueden informar del uso real del contexto de la solicitud activa en lugar de mostrar únicamente el historial del chat.

---

## Persistencia de datos

Todo el estado almacenado en disco se guarda dentro del directorio de trabajo del proyecto.

| Ruta                                          | Contenido                                                         |
| --------------------------------------------- | ----------------------------------------------------------------- |
| `sessions/{id}.json`                          | Historial de conversación de cada sesión.                         |
| `localllm/custom_models.json`                 | Rutas de modelos añadidos manualmente.                            |
| `localllm/model_configs.json`                 | Configuración específica de cada modelo.                          |
| `localllm/parallel_prefs.json`                | Preferencias de carga paralela y configuración de pipelines.      |
| `app_config.json`                             | Umbrales, valores predeterminados, tema y paletas personalizadas. |
| `localllm/server_config.json`                 | Rutas de binarios, host, puerto y GPU offload.                    |
| `localllm/auto_setup_state.json`              | Estado reanudable de la configuración automática inicial.         |
| `localllm/api_models.json`                    | Configuración de modelos API.                                     |
| `localllm/cli_prefs.json`                     | Último modelo y contexto utilizados por la CLI.                   |
| `localllm/pipeline_builder_history/{id}.json` | Historial de chat y contexto del AI Pipeline Builder.             |
| `mcp_config.json`                             | Definiciones de servidores MCP.                                   |
| `paused_jobs/{id}.json`                       | Instantáneas de trabajos de resumen pausados.                     |
| `ref_cache/{id}_raw.txt`                      | Texto original de los archivos de referencia adjuntos.            |
| `ref_cache/{id}.pkl`                          | Cachés serializadas de referencias volcadas al disco.             |
| `ref_index/{sid}_refs.json`                   | Índice de metadatos de referencias por sesión.                    |
| `localllm/pipelines/{name}.json`              | Pipelines visuales guardados.                                     |

---

## Watchdog de RAM

La clase `RamWatchdog` y la función `_ram_free_mb()` (basada en `psutil`) evitan errores por falta de memoria durante el procesamiento de documentos largos.

Se activan cuando:

* Se añade un nuevo archivo de referencia.
* Periódicamente durante el procesamiento de múltiples PDF (cada cinco fragmentos).
* Justo antes de la consolidación final del resumen.

Cuando se activa, `SessionReferenceStore.flush_ram()` llama a `_spill_to_disk()` para cada referencia cargada, serializando todos los fragmentos de texto y limpiando `_hot`.

Los fragmentos se vuelven a cargar bajo demanda utilizando caché LRU.

Antes del resumen final, el proceso de **reactive reload** vuelve a cargar en memoria los fragmentos más relevantes para la consulta cuando la memoria disponible lo permite.

---

## Configuración de la aplicación

Todos los umbrales de ejecución se almacenan en `app_config.json` y pueden editarse completamente desde la pestaña **⚙️ Config**.

| Configuración           | Valor predeterminado | Descripción                                                                 |
| ----------------------- | -------------------- | --------------------------------------------------------------------------- |
| `ram_watchdog_mb`       | 800                  | Umbral de RAM libre (MB) que activa el volcado al disco.                    |
| `chunk_index_size`      | 400                  | Tamaño en caracteres de los fragmentos indexados.                           |
| `max_ram_chunks`        | 120                  | Número máximo de fragmentos por referencia mantenidos en memoria.           |
| `summary_chunk_chars`   | 3000                 | Caracteres por fragmento durante el resumen.                                |
| `summary_ctx_carry`     | 600                  | Caracteres del resumen anterior que se conservan como contexto.             |
| `summary_n_pred_sect`   | 380                  | Máximo de tokens por resumen de sección.                                    |
| `summary_n_pred_final`  | 700                  | Máximo de tokens para la consolidación final.                               |
| `multipdf_n_pred_sect`  | 380                  | Tokens por sección durante trabajos con múltiples PDF.                      |
| `multipdf_n_pred_final` | 900                  | Tokens para la consolidación final entre documentos.                        |
| `ref_top_k`             | 6                    | Número de fragmentos mejor puntuados recuperados por referencia y consulta. |
| `ref_max_context_chars` | 3000                 | Cantidad máxima de texto de referencia que puede inyectarse.                |
| `pause_after_chunks`    | 2                    | Número de fragmentos antes de sugerir una pausa automática.                 |
| `default_threads`       | 12                   | Número predeterminado de hilos de CPU para llama.cpp.                       |
| `default_ctx`           | 4096                 | Ventana de contexto predeterminada.                                         |
| `default_n_predict`     | 512                  | Número máximo predeterminado de nuevos tokens.                              |
| `auto_spill_on_start`   | false                | Vuelca todas las cachés de referencias al iniciar la aplicación.            |

Las configuraciones específicas de cada modelo tienen prioridad sobre estos valores predeterminados.
