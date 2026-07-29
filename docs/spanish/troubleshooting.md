# Solución de problemas

Errores comunes y sus soluciones. Si lo que estás viendo no coincide con ninguno de los casos descritos aquí, abre un issue en https://github.com/7zonesystems/NativeLab/issues.

---

## Instalación / inicio

### `command not found: nativelab`

El paquete se instaló correctamente, pero el script no está en tu `$PATH`. Ejecútalo como un módulo:

```bash id="q17gkx"
python -m nativelab          # GUI
python -m nativelab --cli    # CLI
```

Si utilizaste `pip install --user`, asegúrate también de que el directorio de binarios del usuario esté incluido en `$PATH`:

```bash id="v7nk2g"
# Linux / macOS
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Windows
# Añade %APPDATA%\Python\Python3xx\Scripts a la variable de entorno PATH
```

### `ModuleNotFoundError: No module named 'PyQt6'`

`pip install nativelab` debería instalar esta dependencia automáticamente. Si no fue así:

```bash id="qqfh9f"
pip install PyQt6
```

Si `pip` indica problemas con los wheels, probablemente tu versión de Python sea demasiado antigua. Los wheels de PyQt6 están disponibles a partir de Python 3.10.

### `ModuleNotFoundError: No module named 'nativelab'`

Probablemente falta la instalación o estás ejecutando Python desde un directorio donde existe una carpeta `nativelab/` al mismo nivel y Python está importando esa carpeta en lugar del paquete instalado.

Ejecuta el programa desde otro directorio o utiliza:

```bash id="u3a75r"
python -m nativelab
```

---

## Binarios de llama.cpp

### "No llama.cpp binary found."

La GUI o la CLI no pudieron localizar `llama-server` o `llama-cli`.

Tres soluciones:

1. **La más sencilla:** abre la GUI, ve a **⬇️ Download** e instala una versión.
2. **Manual:** coloca los binarios en `./llama/bin/` (relativo al directorio de trabajo).
3. **Ruta personalizada:** configura las rutas manualmente desde la pestaña **🖥️ Server** de la GUI. Se guardarán en `./localllm/server_config.json`.

### "Server start failed - falling back to llama-cli mode"

No es un error crítico. El modelo seguirá funcionando, pero cada prompt reiniciará el proceso (lo que ralentiza considerablemente las conversaciones largas).

Las causas más comunes son:

* El puerto elegido ya está siendo utilizado → ajusta el rango de puertos en la pestaña **Server**.
* Falta el binario del servidor y únicamente está disponible `llama-cli`.
* Quedó un proceso `llama-server` de un cierre inesperado ocupando el puerto. La aplicación ejecuta `kill_stray_llama_servers()` al iniciarse, pero si no funciona, usa `pkill llama-server` (Linux/macOS) o finaliza el proceso desde el Administrador de tareas de Windows.

### Permiso denegado en Linux/macOS

```bash id="qyyqk8"
chmod -R +x ./llama/bin/
```

---

## Modelos

### "Model not found"

`LlamaEngine.load()` no pudo resolver la ruta.

Comprueba:

* Que la ruta del registro de modelos coincida con la ubicación real del archivo (puede haber cambiado si el archivo fue movido).
* Que el archivo no tenga tamaño cero (descarga interrumpida). Ejecuta nuevamente el asistente o vuelve a descargarlo desde la pestaña **Download**. Los archivos `.part` continúan automáticamente las descargas interrumpidas.

### El modelo carga pero responde con texto incoherente, repite contenido o ignora instrucciones

Casi siempre se trata de una **plantilla de prompt incorrecta**.

NativeLab detecta automáticamente la familia del modelo a partir del nombre del archivo, pero si este fue renombrado o utiliza un nombre poco habitual, la detección puede recurrir a una plantilla genérica.

Solución rápida:

En la pestaña **Models**, edita el modelo y cambia el campo `family` para que coincida con la familia real del modelo.

Si es necesario, añade una nueva familia siguiendo la guía de [models.md#adding-a-new-model-family](models.md#adding-a-new-model-family).

### "Model load: Failed" sin más detalles

Consulta **Dev > Logs**.

Las causas más habituales son:

* Memoria RAM insuficiente → utiliza una cuantización más pequeña o reduce el contexto.
* Archivo GGUF corrupto → vuelve a descargarlo.
* La versión de llama.cpp es demasiado antigua para ese formato de cuantización (por ejemplo, los IQ-quants requieren versiones recientes).

---

## Rendimiento

### Las respuestas son muy lentas

* Reduce el tamaño del contexto (`/ctx 2048` en la CLI o mediante el control deslizante en la GUI).
* Utiliza una cuantización más pequeña (Q4_K_M suele ofrecer el mejor equilibrio).
* Activa GPU Offload desde la pestaña **Server** si dispones de GPU.
* Si estás utilizando el modo de compatibilidad con `llama-cli`, intenta hacer funcionar el servidor. El modo servidor es considerablemente más rápido para conversaciones largas.

### Las respuestas terminan a mitad de una frase

Se agotó el límite `n_predict` de esa llamada.

Puedes:

* Aumentar `default_n_predict` en **Config** o el parámetro `n_predict` específico del modelo en la pestaña **Models**.
* Pedir simplemente al modelo que continúe escribiendo.

### Uso elevado de RAM durante el resumen de documentos

Es un comportamiento esperado con documentos largos.

El watchdog mueve las cachés de referencias al disco cuando la RAM libre cae por debajo de `ram_watchdog_mb` (800 MB por defecto).

Puedes:

* Reducir `max_ram_chunks` en **Config**.
* Aumentar `ram_watchdog_mb` para comenzar el volcado antes.
* Activar `auto_spill_on_start: true` para iniciar con un uso de memoria más reducido.

---

## CLI

### El asistente de configuración termina con "Aborted: cannot proceed without binaries or API mode"

Respondiste **no** cuando se preguntó si deseabas continuar sin binarios locales.

Ejecuta nuevamente:

```bash id="i17uc2"
nativelab --cli setup
```

e instala previamente los binarios o responde **y** para utilizar únicamente el modo API.

### La CLI no muestra el icono al iniciar

El icono solo aparece en iTerm2, WezTerm, el terminal de VS Code, Hyper, mintty y Kitty.

En otros terminales (gnome-terminal, xterm o una sesión SSH estándar) únicamente se mostrará el banner ASCII.

Es el comportamiento previsto.

### La referencia `@file` es ignorada

No debe haber espacios ni comillas entre `@` y la ruta:

```text id="2nl31t"
✗  @ ./foo.py            # el espacio rompe la referencia
✗  @"./foo.py"           # las comillas también
✓  @./foo.py
✓  @/abs/path/to/foo.py
```

La ruta se resuelve de forma relativa al directorio de trabajo actual.

---

## Pipelines

### "Pipeline doesn't progress past the first block"

Comprueba lo siguiente:

* El bloque **Input** debe estar conectado con otro bloque.
* Un bloque **FILTER** puede haber detenido el pipeline; consulta el registro de ejecución buscando líneas con `"drop"`.
* Los bloques que no son de lógica solo admiten una conexión saliente por puerto. Si necesitas múltiples ramas, utiliza un bloque **SPLIT**.

### Un bucle nunca termina

El contador del bucle representa el número máximo de *visitas* a esa conexión.

Si se supera mediante otra ruta del grafo, el comportamiento puede no ser el esperado.

Añade una condición clara de salida o utiliza un número fijo de iteraciones.

### El Pipeline o AI Builder indica que el límite de contexto es demasiado pequeño

La solicitud más los tokens reservados para la salida superan la ventana de contexto del modelo cargado.

Aumenta el contexto del modelo y vuelve a cargarlo, o reduce el tamaño del prompt o del historial del lienzo.

En AI Builder, la comprobación previa bloquea la solicitud antes de enviarla al modelo.

Si el servidor devuelve un error similar a:

```text id="bjdvn5"
request (...) exceeds the available context size (...)
```

NativeLab debería mostrar un cuadro de diálogo indicando que el prompt es demasiado grande y sugiriendo aumentar el contexto o reducir la entrada.

El error seguirá registrándose para fines de depuración.

### AI Builder: "The model response did not contain a JSON object"

AI Builder realiza automáticamente un segundo intento utilizando un prompt que exige únicamente JSON.

Si vuelve a fallar, formula una petición más corta y directa, por ejemplo:

```text id="nztmhv"
Make an input -> model -> output pipeline.
```

Evita pedir explicaciones dentro de la misma solicitud.

El modelo debe responder únicamente con un objeto JSON, no con Markdown ni texto descriptivo.

### AI Builder generó un pipeline pero no puede guardarlo

Los pipelines generados también pasan por el validador estándar.

Corrige manualmente el error indicado o solicita a AI Builder que modifique el grafo.

Las causas más comunes son:

* Falta un bloque **Input** o **Output**.
* Extremos de conexión inválidos.
* Conexiones directas entre modelos.
* Bloques de lógica LLM sin instrucciones.
* Código inseguro en un bloque **Custom Code**.

### La barra lateral del Pipeline desapareció

Las barras laterales del constructor de pipelines se retraen cuando se reducen demasiado.

Busca el botón con la flecha circular en el borde central izquierdo o derecho del lienzo para volver a abrir la barra correspondiente.

### Bloque Custom Code: "name X is not defined"

El entorno aislado restringe los elementos integrados de Python.

La lista completa de funciones disponibles está documentada en [workflows.md#custom-code](workflows.md#deterministic-logic-no-model-calls).

Si necesitas acceso más amplio, ejecuta el código fuera del sandbox (por ejemplo, como un script Python independiente invocado mediante un subprocess desde un bloque **Custom Code**).

---

## Modelos API

### "API test failed: 401 Unauthorized"

La clave API es incorrecta o no tiene acceso al modelo especificado.

Verifica ambos campos en la pestaña **API Models**.

### "API test failed: connection refused"

La URL base es incorrecta o el servidor no está en ejecución.

Si utilizas servidores autoalojados (Ollama, LM Studio o vLLM), asegúrate de que estén escuchando en la dirección y puerto configurados.

### API de Anthropic: "credit balance is too low"

Anthropic devuelve errores HTTP 400 cuando no hay saldo suficiente.

El mensaje mostrado en **Dev > Logs** incluye la respuesta completa del servidor.

---

## Restablecer a un estado limpio

Si el comportamiento es extraño y deseas comenzar desde cero sin desinstalar la aplicación:

```bash id="f2f9u6"
# Desde el directorio de trabajo de nativelab
rm -rf localllm/cli_prefs.json localllm/server_config.json
rm -rf paused_jobs/ ref_cache/ ref_index/ sessions/
rm -f app_config.json
```

Los modelos almacenados en `./localllm/*.gguf` se conservan.

Al volver a iniciar NativeLab, el asistente o la GUI recrearán automáticamente los archivos necesarios.

---

## ¿Sigues teniendo problemas?

* **Dev > Logs** en la GUI contiene toda la cadena de errores y suele ser el mejor punto para comenzar el diagnóstico.
* Ejecuta `python -u -m nativelab --cli` para ver toda la salida estándar de errores en tiempo real.
* Abre un issue incluyendo las líneas relevantes del registro junto con tu sistema operativo y la versión de Python: https://github.com/7zonesystems/NativeLab/issues.
