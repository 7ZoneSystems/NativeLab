# Labs - la capa de experimentación

El paquete `nativelab/labs/` es donde llegan las nuevas funciones. Existe para permitirte desarrollar experimentos sin modificar `MainWindow`, los componentes internos del motor o los workers de streaming.

> **TL;DR** — coloca un panel `QWidget` en `nativelab/labs/<feature>.py`, regístralo en `LAB_FEATURES` y obtendrás gratuitamente estado del motor, cambio de modelo, cambio de contexto y llamadas síncronas al LLM.

---

## ¿Qué incluye?

```text
nativelab/labs/
├── __init__.py        reexporta LabEndpoints, LabsTab, LAB_FEATURES, …
├── endpoints.py       LabEndpoints - la superficie compartida
├── labs_tab.py        barra lateral + paneles apilados (LabsTab + lista LAB_FEATURES)
└── pytodoc.py         primera función: generador de README con py-to-doc
```

Labs ahora se encuentra en **Dev > Labs**. Activa **Settings > General > Developer Mode** si la pestaña **Dev** está oculta. La pestaña **Dev** sustituye la barra lateral normal del historial de chat por una barra lateral vertical para desarrolladores que contiene **Labs**, **Logs**, **Integrations**, **Pipeline**, **MCP** y **Skills**.

### Recuperación de proyectos de py-to-doc

En el modo **Project** de py-to-doc, NativeLab escribe un checkpoint seguro para reinicios en:

```text
localllm/temp
```

El checkpoint se actualiza después de cada sección de archivo generada, incluyendo la documentación de clases y funciones.

Si la aplicación se pausa, se cierra o el equipo se apaga, selecciona la misma raíz del proyecto y la misma carpeta de salida, conserva la misma configuración y vuelve a pulsar **Generate**. NativeLab verifica el checkpoint y los archivos Markdown existentes, y continúa desde el último paso completado.

El botón **Resume Project** requiere un checkpoint coincidente y muestra un mensaje claro si el proyecto, la carpeta de salida o la configuración seleccionados no coinciden con el estado guardado.

El panel de py-to-doc también muestra las tareas de proyecto guardadas en `localllm/temp` y `localllm/pytodoc_jobs`. Al seleccionar una de ellas, se restauran el proyecto, la carpeta de salida y la configuración cuando están disponibles, y la ejecución continúa automáticamente desde el checkpoint guardado.

El modo **Project** analiza recursivamente los archivos Python, aplica el `.gitignore` de la raíz del proyecto seleccionada y crea previamente la misma estructura de directorios dentro de la carpeta de salida antes de escribir los archivos Markdown generados.

El contexto de py-to-doc puede ejecutarse en tres modos:

* Sin reinicio del contexto.
* Reinicio fijo después de clases o funciones.
* Reinicio automático según un presupuesto de tokens.

El reinicio automático utiliza un número aproximado de tokens para el historial acumulado de py-to-doc. En modelos GGUF locales, seleccionar un presupuesto automático recarga el modelo o servidor local con la misma ventana de contexto antes de comenzar la generación.

El modo automático también puede recargar el modelo local activo después de finalizar la sección actual cuando la memoria RAM disponible desciende por debajo del umbral configurado en GB o MB, limpiando la caché del backend antes de comenzar la siguiente sección del LLM.

Si la generación de una clase, función o archivo supera el presupuesto establecido, dicha generación finaliza normalmente y la siguiente sección comienza con un contexto acumulado completamente nuevo.

### Laboratorio Structured Edit

El laboratorio **structured-edit** se encuentra en **Dev > Labs**.

Puede adjuntar un archivo de código existente o comenzar desde un espacio de trabajo temporal vacío y solicitar al modelo activo que devuelva operaciones de edición estructuradas en lugar de reescribir el archivo completo.

La copia de trabajo se guarda continuamente en:

```text
localllm/temp_code_edit.json
localllm/temp_code_edit_file
```

El archivo original no se modifica hasta pulsar **Save** o **Save As**.

El laboratorio muestra la estructura analizada del archivo, incluyendo funciones detectadas, rangos de líneas, argumentos, expresiones de retorno y variables locales cuando están disponibles.

---

## El contrato para un panel de laboratorio

Un panel es cualquier subclase de `QWidget` que:

1. Defina los atributos de clase `LAB_NAME` y `LAB_ICON`.
2. Implemente `set_endpoints(endpoints: LabEndpoints)`.
3. Utilice `endpoints` para todas las lecturas del motor, llamadas al LLM y enrutamiento inverso.

Eso es todo.

No es necesario importar `MainWindow`, clases del motor ni configurar manualmente los workers de streaming.

---

## La superficie de endpoints

```python
from nativelab.labs import LabEndpoints
```

### Leer el estado

```python
endpoints.status_text     # "🟢 Server  :8612"
endpoints.is_loaded       # bool
endpoints.mode            # "server" | "cli" | "api" | "unloaded"
endpoints.model_path      # ruta absoluta del GGUF
endpoints.model_name      # solo el nombre del archivo
endpoints.ctx_value       # int
endpoints.server_port     # int
endpoints.is_api_active   # bool
endpoints.is_local_active # bool
endpoints.is_loading      # bool
endpoints.snapshot()      # todo lo anterior como un dict
endpoints.model_family()  # plantilla ModelFamily (BOS/EOS, prefijos, stops)
```

### Llamada síncrona al LLM

Realiza el enrutamiento automáticamente hacia API → servidor → CLI según el motor activo.

Es seguro llamarlo desde un `QThread`.

```python
reply = endpoints.call_llm(
    messages=[
        {"role": "user", "content": "Summarise this:\n" + code},
    ],
    system_prompt="You are a senior engineer.",
    n_predict=512,
    temperature=0.3,
)
```

### Enrutamiento inverso: solicitar cambios al host

```python
endpoints.request_load_model("/path/to/other.gguf")  # → True/False
endpoints.request_context(8192)                      # → True/False
endpoints.request_active_model_reload()              # → True/False
endpoints.wait_until_loaded()                        # → True/False
endpoints.request_unload()                           # → None
endpoints.ensure_server(log_cb=lambda m: print(m))  # → True/False
```

El host (`MainWindow` en la GUI y `cli/chat.py` en la CLI) conecta estos hooks durante el inicio.

El mismo panel funciona en ambos entornos.

### Señales

```python
endpoints.engine_changed.connect(self._refresh)
endpoints.status_changed.connect(self._update_label)   # str
endpoints.log_msg.connect(self._on_log)                # (level, message)
```

---

## Cómo añadir una nueva función a Labs

### 1. Crear el módulo

```text
nativelab/labs/codereview.py
```

```python
from __future__ import annotations
from typing import Optional

from nativelab.imports.import_global import (
    QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton,
)
from .endpoints import LabEndpoints


class CodeReviewPanel(QWidget):
    LAB_NAME = "code-review"
    LAB_ICON = "🔍"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._endpoints: Optional[LabEndpoints] = None
        self._build()

    def set_endpoints(self, endpoints: LabEndpoints):
        self._endpoints = endpoints
        endpoints.status_changed.connect(self.lbl_status.setText)
        self.lbl_status.setText(endpoints.status_text)

    def _build(self):
        root = QVBoxLayout(self)
        root.addWidget(QLabel(f"{self.LAB_ICON}  {self.LAB_NAME}"))
        self.lbl_status = QLabel("…")
        root.addWidget(self.lbl_status)

        self.input = QTextEdit()
        root.addWidget(self.input, 1)

        self.btn = QPushButton("Review")
        self.btn.clicked.connect(self._run)
        root.addWidget(self.btn)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        root.addWidget(self.output, 1)

    def _run(self):
        if not self._endpoints or not self._endpoints.is_loaded:
            self.output.setPlainText("No engine loaded.")
            return
        code = self.input.toPlainText()
        try:
            reply = self._endpoints.call_llm(
                prompt=f"Review this code and list issues:\n\n{code}",
                system_prompt="You are a senior code reviewer.",
            )
        except Exception as exc:
            reply = f"[error: {exc}]"
        self.output.setPlainText(reply)
```

### 2. Registrarlo

En `nativelab/labs/labs_tab.py`:

```python
from .codereview import CodeReviewPanel

LAB_FEATURES: list[Type[QWidget]] = [
    PyToDocPanel,
    CodeReviewPanel,    # ← aquí
]
```

### 3. Eso es todo el cambio

Reinicia la aplicación.

Tu panel aparecerá en **Dev > Labs** utilizando el icono y el nombre definidos en los atributos de la clase.

Recibirá automáticamente una instancia de `LabEndpoints`.

---

## ¿Por qué existe?

Antes de Labs, añadir una nueva función implicaba modificar `MainWindow` (referencias al motor), `tabs.py` (definición de la interfaz), los workers de streaming (llamadas personalizadas) y, probablemente, también el registro de modelos.

Eliminar una función significaba seguir referencias distribuidas por todas las capas.

La superficie de Labs es deliberadamente pequeña:

* **Lectura de estado:** sin modificar estado durante las consultas.
* **`call_llm`:** una llamada síncrona, un único valor de retorno y sin coordinar manualmente `QThread`.
* **Enrutamiento inverso:** tres hooks explícitos; nunca se exponen referencias directas al motor.
* **Señales:** los cambios del motor notifican automáticamente a los paneles sin necesidad de sondeo continuo.

Si alguna vez sientes la necesidad de importar una clase del motor, un worker de streaming o `MainWindow` desde una función de Labs, detente.

La función debería funcionar mediante `endpoints.call_llm` o la superficie necesita un nuevo método.

Las Pull Requests que amplíen `LabEndpoints` son bienvenidas y fáciles de revisar precisamente **porque esta superficie permanece pequeña**.

---

## Cómo utiliza la CLI la misma superficie

`nativelab/cli/chat.py` construye un objeto `LabEndpoints` exactamente igual que `MainWindow`.

La diferencia está únicamente en los hooks de enrutamiento inverso (comportamiento síncrono de la CLI en lugar de diálogos de la GUI), mientras que la semántica de las llamadas permanece idéntica.

Por eso `/load`, `/ctx` y `/unload` dentro del REPL se comportan exactamente igual que una función de Labs solicitando el mismo cambio desde la GUI: **literalmente recorren el mismo camino de ejecución**.

```python
# nativelab/cli/chat.py - fragmento relevante
endpoints.bind_engines(
    llama_provider=lambda: eng,
    api_provider  =lambda: api,
)
endpoints.bind_reverse_routes(
    on_context=on_context,
    on_model  =on_model,
    on_unload =on_unload,
)
```

---

## Inventario

| Función   | Estado     | Módulo            |
| --------- | ---------- | ----------------- |
| py-to-doc | ✅ incluida | `labs/pytodoc.py` |

Se están desarrollando más funciones.

Si añades una nueva, incorpora también una fila a esta tabla.
