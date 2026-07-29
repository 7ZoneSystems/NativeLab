# Solución de problemas

Errores comunes y sus soluciones. Si algo aquí no coincide con lo que estás viendo, abre un issue en https://github.com/7zonesystems/NativeLab/issues.

---

## Instalación / inicio

### `command not found: nativelab`

El paquete se instaló, pero el script no está en tu `$PATH`. Ejecútalo como módulo:

```bash id="g8m2wk"
python -m nativelab          # GUI
python -m nativelab --cli    # CLI
```

Si usaste `pip install --user`, asegúrate también de que el directorio de binarios de usuario esté en `$PATH`:

```bash id="0a4v2m"
# Linux / macOS
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Windows
# Añade %APPDATA%\Python\Python3xx\Scripts a tu variable de entorno PATH
```

### `ModuleNotFoundError: No module named 'PyQt6'`

`pip install nativelab` debería incluir esto. Si no lo hizo:

```bash id="p3q7zh"
pip install PyQt6
```

Si `pip` se queja de los wheels, probablemente tu Python sea demasiado antiguo. Los wheels de PyQt6 comienzan en Python 3.10.

### `ModuleNotFoundError: No module named 'nativelab'`

O bien falta instalarlo, o estás ejecutando Python desde una carpeta donde `nativelab/` existe como directorio hermano y Python importa la *carpeta* en lugar del paquete instalado. Ejecuta desde otro directorio de trabajo o usa `python -m nativelab`.

---

## Binarios de llama.cpp

### "No llama.cpp binary found."

La CLI / GUI no pudo localizar `llama-server` o `llama-cli`. Hay tres soluciones:

1. **La más fácil**: abre la GUI, ve a **⬇️ Download** e instala una release.
2. **Manual**: coloca los binarios en `./llama/bin/` (relativo a tu directorio de trabajo).
3. **Ruta personalizada**: define rutas explícitas en la pestaña **🖥️ Server** de la GUI. Se guardan en `./localllm/server_config.json`.

### "Server start failed - falling back to llama-cli mode"

No es fatal. El modelo sigue funcionando, pero cada prompt reinicia el proceso (lento para conversaciones con varios turnos). Causas comunes:

* El puerto elegido ya está en uso → ajusta el rango de puertos en la pestaña **Server**.
* Falta el binario del servidor: solo está `llama-cli` en disco.
* Un `llama-server` anterior quedó colgado tras un cierre inesperado y está ocupando el puerto. La app llama a `kill_stray_llama_servers()` al iniciar, pero si no basta, `pkill llama-server` (Linux/macOS) o el Administrador de tareas (Windows) lo resuelven.

### Permiso denegado en Linux/macOS

```bash id="x0v1rg"
chmod -R +x ./llama/bin/
```

---

## Modelos

### "Model not found"

`LlamaEngine.load()` no pudo resolver la ruta. Revisa:

* Que la ruta en el registro de modelos coincida con el archivo real en disco (las rutas pueden cambiar si el archivo se movió).
* Que el archivo no tenga 0 bytes (descarga interrumpida). Vuelve a ejecutar el asistente o descárgalo otra vez desde la pestaña **Download**; los archivos `.part` reanudan automáticamente.

### El modelo carga pero responde con galimatías / repite / ignora instrucciones

Casi siempre es un **desajuste de plantilla de prompt**. NativeLab detecta automáticamente la familia a partir del nombre del archivo, pero si el archivo fue renombrado o usa un nombre poco común, la detección puede caer en una plantilla genérica.

Solución rápida: en la pestaña **Models**, edita el modelo y cambia el campo `family` para que coincida con la familia real. Si hace falta, añade una nueva familia; consulta [models.md#adding-a-new-model-family](models.md#adding-a-new-model-family).

### "Model load: Failed" sin más detalle

Revisa **Dev > Logs**. Las causas más comunes son:

* Falta de RAM → usa una cuantización más pequeña o baja el contexto.
* GGUF corrupto → vuelve a descargarlo.
* La versión de llama.cpp es demasiado antigua para ese formato de cuantización (por ejemplo, los IQ-quants requieren una compilación reciente).

---

## Rendimiento

### Las respuestas son muy lentas

* Baja el tamaño del contexto (`/ctx 2048` en la CLI, deslizador en la GUI).
* Usa una cuantización más pequeña (Q4_K_M suele ser el mejor equilibrio en la mayoría del hardware).
* Activa GPU offload en la pestaña **Server** si tienes GPU.
* Si estás en modo de fallback de CLI, haz que funcione el binario del servidor: el modo servidor es muchísimo más rápido en conversaciones con varios turnos.

### Las respuestas se cortan a mitad de frase

Se agotó el presupuesto `n_predict` de esa llamada. Puedes:

* Aumentar `default_n_predict` en **Config**, o `n_predict` por modelo en la pestaña **Models**.
* Pedirle al modelo que “continue” en el siguiente mensaje.

### Uso alto de RAM durante el resumen

Es lo esperado para documentos largos. El watchdog vuelca las cachés de referencias al disco cuando la RAM libre cae por debajo de `ram_watchdog_mb` (por defecto, 800 MB). Puedes:

* Bajar `max_ram_chunks` en **Config**.
* Subir `ram_watchdog_mb` para volcar antes.
* Activar `auto_spill_on_start: true` para arrancar más ligero.

---

## CLI

### El asistente de configuración sale con "Aborted: cannot proceed without binaries or API mode"

Respondísteis “no” cuando se preguntó si querías continuar sin binarios locales. Vuelve a ejecutar `nativelab --cli setup` e instala primero los binarios o responde “y” para usar solo el modo API.

### La CLI no muestra icono al iniciar

El icono solo se renderiza en iTerm2 / WezTerm / el terminal de VS Code / Hyper / mintty / Kitty. En otros terminales (gnome-terminal, xterm, SSH simple) verás solo el banner ASCII: es la ruta de respaldo silenciosa, no un error.

### La referencia `@file` se ignora

Asegúrate de que no haya espacios ni comillas entre `@` y la ruta:

```text id="j4n8sm"
✗  @ ./foo.py            # el espacio rompe esto
✗  @"./foo.py"           # las comillas rompen esto
✓  @./foo.py
✓  @/abs/path/to/foo.py
```

La ruta se resuelve relativa al directorio de trabajo actual.

---

## Pipelines

### "Pipeline doesn't progress past the first block"

* Comprueba que **Input** esté conectado a algo aguas abajo.
* Un bloque **FILTER** puede haber terminado el pipeline; revisa el log de ejecución buscando líneas de “drop”.
* Cada puerto solo puede tener una conexión saliente en bloques no lógicos. Si necesitas ramificar, usa un bloque **SPLIT**.

### Un loop nunca termina

El conteo del loop es el máximo de *visitas* a esa arista. Si se supera por otra ruta, no volverá a buclear. Añade una condición de rama clara o usa un número fijo de iteraciones.

### Pipeline o AI Builder dice que el límite de contexto es demasiado pequeño

La solicitud más los tokens reservados de salida superan la ventana de contexto del modelo cargado. Aumenta el contexto del modelo y recárgalo, o acorta el prompt / historial del lienzo. En AI Builder, la comprobación previa bloquea la solicitud antes de enviarla al modelo.

Si el servidor upstream devuelve un error como:

```text id="r5q1nx"
request (...) exceeds the available context size (...)
```

NativeLab debería mostrar un diálogo normal explicando que el prompt es demasiado grande y sugiriendo un contexto mayor o una entrada más corta. El error sigue quedando registrado para depuración.

### AI Builder: "The model response did not contain a JSON object"

AI Builder reintenta una vez con un prompt más estricto que solo acepta JSON. Si sigue fallando, haz la solicitud más corta y directa, por ejemplo:

```text id="c2v9kp"
Make an input -> model -> output pipeline.
```

Evita pedir explicaciones en la misma solicitud. El modelo debe devolver un objeto JSON, no Markdown ni prosa.

### AI Builder generó un pipeline pero no se puede guardar

Los pipelines generados siguen pasando por el validador normal. Corrige manualmente el problema indicado o pídele al AI Builder que revise el grafo. Las causas comunes son bloques **Input/Output** ausentes, endpoints de conexión inválidos, enlaces directos entre modelos, instrucciones faltantes en bloques LLM o código inseguro en **Custom Code**.

### La barra lateral del Pipeline desapareció

Las barras laterales del constructor se retraen cuando se arrastran demasiado pequeñas. Mira el borde medio de la izquierda o derecha del lienzo y haz clic en el botón circular para volver a abrir la barra.

### Bloque Custom Code: "name X is not defined"

El sandbox restringe los builtins. La lista completa disponible en custom code está documentada en [workflows.md#custom-code](workflows.md#deterministic-logic-no-model-calls). Para acceso más amplio, el código debería ejecutarse fuera del sandbox (por ejemplo, como un script Python normal invocado vía subprocess desde un bloque Custom Code).

---

## Modelos API

### "API test failed: 401 Unauthorized"

La clave API es incorrecta, o la clave no tiene acceso al modelo que especificaste. Revisa ambos campos en la pestaña **API Models**.

### "API test failed: connection refused"

La base URL es incorrecta o el servidor no está en ejecución. Para servidores autoalojados (Ollama, LM Studio, vLLM), asegúrate de que estén escuchando en la dirección y puerto que configuraste.

### Anthropic API: "credit balance is too low"

Anthropic devuelve errores 400 cuando te quedas sin crédito. El mensaje de error en **Dev > Logs** incluye el cuerpo completo de la respuesta upstream.

---

## Restablecer a un estado limpio

Si todo está raro y quieres empezar de cero sin desinstalar:

```bash id="d9x5cf"
# Desde tu directorio de trabajo de nativelab
rm -rf localllm/cli_prefs.json localllm/server_config.json
rm -rf paused_jobs/ ref_cache/ ref_index/ sessions/
rm -f app_config.json
```

Los modelos en `./localllm/*.gguf` se conservan. Al volver a abrir NativeLab, el asistente / GUI recreará todo lo necesario.

---

## ¿Todavía atascado?

* **Dev > Logs** en la GUI contiene toda la cadena de errores; es el punto más útil para mirar primero.
* Ejecuta `python -u -m nativelab --cli` para ver todo stderr en tiempo real.
* Abre un issue con las líneas de log relevantes y tu sistema operativo / versión de Python: https://github.com/7zonesystems/NativeLab/issues.
