# Características

NativeLab está construido sobre cuatro pilares: **inferencia local (local-first)**, **pipelines con múltiples motores**, **flujos de trabajo avanzados para documentos** y una **capa de experimentación** para nuevas ideas.

---

## Novedades de la versión v0.3.7

### Cliente Android PhonoLab

El cliente oficial de NativeLab para Android ya está disponible. Ejecuta modelos llama.cpp directamente en el dispositivo con funciones avanzadas específicas para móviles:

* **Inferencia en el dispositivo:** Utiliza binarios integrados de llama-server con un wrapper JNI para evitar las restricciones W^X de Android.
* **Compatibilidad con modelos de visión:** Detección automática de modelos de visión (LLaVA, Qwen-VL, InternVL, etc.) con emparejamiento automático de archivos mmproj.
* **Procesamiento RAG de documentos:** Extracción y fragmentación nativa de PDF, texto y documentos para chats con contexto.
* **Servidor API LAN:** Servidor HTTP compatible con OpenAI y Anthropic para comunicación entre escritorio y móvil.
* **Integración multiplataforma:** NativeLab Desktop puede descubrir y utilizar dispositivos PhonoLab como servidores remotos de IA.
* **Ecosistema de modelos compartido:** El mismo formato GGUF y los mismos niveles de cuantización funcionan tanto en escritorio como en dispositivos móviles.

### Mejoras en AI Builder y el editor de pipelines

El constructor de pipelines ahora incluye una pestaña **AI Builder** junto a **Execution**. Un modelo cargado puede generar archivos JSON de pipelines de NativeLab a partir de una solicitud en lenguaje natural, guardarlos mediante el sistema normal de pipelines y volver a cargarlos en el lienzo para realizar pruebas.

El AI Builder incluye comprobaciones previas de contexto, reintentos automáticos cuando se espera únicamente JSON, soporte para `/get_data` y `/context`, historial local del AI Builder y prompts inteligentes con conocimiento del estado del lienzo para realizar ediciones iterativas.

El editor de pipelines también incorpora ejemplos predefinidos, expansión automática del lienzo cuando se cargan o generan bloques lejanos, desplazamiento mediante arrastre sobre lienzos vacíos, barras laterales redimensionables, flechas flotantes para reabrir paneles y escalado dinámico de texto y controles en barras laterales estrechas.

Consulta [pipeline-builder.md](pipeline-builder.md).

### Núcleo nativo del pipeline

Las rutas críticas deterministas del pipeline disponen ahora de una capa opcional de aceleración en C, mientras que Python continúa siendo responsable de la interfaz de usuario, la orquestación, los plugins, las llamadas a modelos y el manejo de errores.

Los helpers nativos cubren:

* Normalización de identificadores de bloques.
* Reasignación de conexiones.
* Detección de ciclos.
* Funciones auxiliares de transformación y combinación.
* Selección de rutas.
* Límites de iteración en bucles.
* Registros de validación.

Los wrappers de Python mantienen el mismo comportamiento cuando la extensión nativa no está disponible.

### Mejoras de la interfaz y del proceso de configuración

El punto de entrada principal de la interfaz se dividió en módulos especializados dentro de `nativelab/UI/mainwindow/`, el cierre de `QThread` se centralizó, los errores por límite de contexto ahora se muestran mediante diálogos normales para el usuario y el proceso de configuración inicial puede reanudarse, ofreciendo opciones adaptadas al hardware entre llama.cpp y Hugging Face Transformers.

### Funciones incorporadas en la v0.3.7 que siguen vigentes

### Labs: la capa de experimentación

Nuevo paquete `nativelab/labs/` junto con una pestaña dedicada en la interfaz gráfica.

Cada función experimental recibe una única instancia de `LabEndpoints` y la utiliza para consultar el estado del motor, cambiar modelos, modificar el contexto y realizar llamadas síncronas al LLM (con enrutamiento automático API → servidor → CLI).

Añadir una nueva función consiste simplemente en agregar un archivo y registrarlo.

Consulta [labs.md](labs.md).

### CLI de terminal — `nativelab --cli`

Cliente de terminal inspirado en Claude Code.

El asistente interactivo de configuración descarga un modelo desde Hugging Face, selecciona un tamaño de contexto, guarda las preferencias y abre un REPL con:

* Inserción de archivos mediante `@file`.
* Comandos slash.
* Herramienta de lint integrada.
* Renderizado de iconos en terminales compatibles como iTerm2 y Kitty.

Consulta [cli.md](cli.md).

### Superficie de endpoints compartida con la CLI

Los mismos `LabEndpoints` que utilizan los paneles de Labs también son utilizados por el REPL de la CLI.

Los hooks de enrutamiento inverso (`request_load_model`, `request_context`, `request_unload`) están conectados de forma uniforme. Un comando `/load` desde el REPL se comporta exactamente igual que una función de Labs solicitando cambiar de modelo.

Los pipelines visuales guardados también se exponen mediante el puente HTTP de integración como identificadores de modelo compatibles con OpenAI (`pipeline:<name>`).

Al realizar una llamada a `/v1/chat/completions` utilizando uno de esos identificadores, se ejecuta el pipeline guardado mediante el `PipelineExecutionWorker` habitual.

La página **Dev → API Server** utiliza el mismo catálogo. El menú **Hosted model** incluye los pipelines guardados, de modo que al seleccionar `pipeline:<name>` las solicitudes API se procesan mediante el ejecutor del pipeline en lugar de cargar un ejecutor de modelos independiente.

---

# Catálogo (todo lo incluido en la v0.3.7)

## Inferencia

* **llama.cpp local:** `LlamaEngine` inicia `llama-server` para streaming HTTP real con el modelo residente en memoria RAM y utiliza `llama-cli` por solicitud cuando el servidor no está disponible.
* **Modelos API:** `ApiEngine` actúa como reemplazo directo compatible con APIs OpenAI (`/chat/completions`, autenticación Bearer) y Anthropic (`/v1/messages`, `x-api-key`). Funciona tanto con servicios alojados como con servidores propios como LM Studio, Ollama y vLLM. Consulta [models.md#local-and-api-backend-support](models.md#local-and-api-backend-support).
* **Descarga de trabajo a GPU:** La sección **Settings → Server** permite configurar `ngl`, `main_gpu` y `tensor_split` para sistemas con varias GPU.
* **Motores paralelos:** Carga simultáneamente motores para razonamiento, resumen, programación y tareas secundarias, cada uno ejecutándose en su propio puerto de llama-server. NativeLab muestra advertencias sobre el uso de RAM antes de activarlos.
* **Modo Pipeline:** Las solicitudes de programación se procesan primero mediante motores especializados en análisis estructural y después sus resultados alimentan al modelo de programación, obteniendo respuestas de mayor calidad.

## Modelos

* **Detección automática de familias:** Reconocimiento de más de 20 plantillas de chat según el nombre del archivo (DeepSeek, DeepSeek-R1, Mistral, Mixtral, LLaMA-2/3, Phi-3/3.5, Qwen/ChatML, Gemma, CodeLlama, Falcon, Vicuna, OpenChat, Neural-Chat, Starling, Yi, Zephyr, Solar, Orca y Command-R).
* **Detección de cuantización:** Compatibilidad con todas las cuantizaciones de llama.cpp, desde `Q4_0` hasta K-Quants (`Q2_K`–`Q6_K`) y variantes Imatrix (`IQ1_S`–`IQ4_XS`), mostrando niveles de calidad codificados por colores.
* **Parámetros por modelo:** Configuración individual de `threads`, `ctx`, `temperature`, `top_p`, `repeat_penalty` y `n_predict`, almacenados en `model_configs.json`.
* **Descargadores:** Búsqueda de repositorios GGUF, selección de plantillas populares desde `templates.py`, descarga de snapshots completos de Hugging Face Transformers para `hf:<local-folder>`, descarga de modelos Ollama y obtención de compilaciones de llama.cpp.

## Flujos de trabajo para documentos y código

* **Motor de referencias:** Permite adjuntar PDF, texto plano o archivos fuente a una sesión. Los fragmentos más relevantes se insertan automáticamente como bloques `[REFERENCE: ...]` antes de cada prompt.
* **Analizador de scripts:** Análisis AST para Python y analizadores mediante expresiones regulares para más de 20 lenguajes (JavaScript, TypeScript, Rust, Go, C/C++, Java, Kotlin, Ruby, SQL, Bash, YAML, JSON, TOML, Lua, Swift, C#, PHP, R, Julia y Markdown), extrayendo importaciones, clases, funciones, constantes, tipos, objetos SQL y claves de configuración.
* **Resumen por fragmentos:** Los PDF largos se dividen, resumen sección por sección manteniendo contexto acumulado y posteriormente se consolidan. Es posible pausar y reanudar el proceso desde disco.
* **Síntesis de múltiples PDF:** Resume varios documentos y genera un informe con los temas comunes entre ellos.
* **Monitor de RAM:** Descarga automáticamente las cachés de referencias al disco cuando la memoria libre cae por debajo de un umbral configurable y las recarga antes de la consolidación final.

## Constructor visual de pipelines

Editor basado en nodos con más de 20 tipos de bloques.

* Entrada, Salida, Modelo e Intermedio (salida en streaming).
* Contexto: Reference, Knowledge y PDF (con resumen automático para documentos grandes).
* Lógica: IF/ELSE, SWITCH, FILTER, TRANSFORM, MERGE, SPLIT y Custom Code.
* Lógica basada en LLM: LLM-IF, LLM-SWITCH, LLM-FILTER, LLM-TRANSFORM y LLM-SCORE.
* Bucles con contador de iteraciones (por ejemplo, redacción → crítica × 3).
* Ejemplos predefinidos incluidos.
* Pestaña AI Builder para generación y edición iterativa de pipelines mediante IA.
* Barras laterales redimensionables con controles flotantes.
* Lienzo de tamaño dinámico.
* Expresiones Python ejecutadas en un entorno aislado (sin importaciones ni acceso a archivos o red).
* Registro de ejecución en tiempo real y streaming por bloque.
* Guardado y carga en formato JSON.

Consulta [pipeline-builder.md](pipeline-builder.md).

## Integración MCP

Gestiona servidores **Model Context Protocol (MCP)** desde una pestaña dedicada.

Permite configurar:

* Servidores `stdio` ejecutados como procesos hijos.
* Servidores SSE mediante endpoints HTTP.
* Variables de entorno personalizadas.
* Estado en tiempo real.

Consulta [workflows.md#mcp](workflows.md#mcp).

## Mejoras de experiencia de usuario

* Temas claro y oscuro.
* Paletas de colores totalmente personalizables.
* Renderizado Markdown con resaltado de sintaxis y botón **Copy** en todos los bloques de código.
* Barra lateral de sesiones organizada por fecha con opciones para renombrar, exportar a Markdown y eliminar.
* Indicadores en tiempo real del uso de RAM y del contexto.
* Pausa y reanudación de trabajos largos de resumen.

## Cliente Android PhonoLab

* **Inferencia local:** Ejecuta modelos llama.cpp directamente en Android mediante binarios integrados y un wrapper JNI.
* **Compatibilidad con modelos de visión:** Detección automática de modelos de visión y emparejamiento automático de archivos mmproj.
* **Procesamiento RAG:** Extracción y fragmentación nativa de PDF, texto y documentos.
* **Servidor API LAN:** Compatible con OpenAI y Anthropic para comunicación entre escritorio y móvil.
* **Gestión de almacenamiento:** Selección de almacenamiento mediante SAF con ubicación predeterminada para modelos.
* **Sistema de temas:** Cambio entre modo claro y oscuro con gestión adecuada del ciclo de vida.
* **Renderizado matemático:** Visualización de fórmulas mediante KaTeX.
* **Manejo avanzado de errores:** Sistema multinivel que distingue entre errores fatales y no fatales.
* **Flujos de trabajo multiplataforma:** Creación de pipelines que combinan escritorio y dispositivos móviles.

---

## Hoja de ruta

La capa **Labs** está diseñada como el lugar donde se desarrollarán los próximos experimentos. Es el punto de menor fricción para añadir nuevas funciones que interactúan con el motor.

Si deseas contribuir al proyecto, **[labs.md](labs.md)** es el mejor punto de partida.
