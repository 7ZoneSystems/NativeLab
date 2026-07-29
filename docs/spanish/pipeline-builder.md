# Constructor de Pipelines

El constructor de pipelines de NativeLab es el espacio de trabajo visual para crear flujos de trabajo repetibles con LLM. Admite edición manual mediante nodos, ejemplos predefinidos incluidos, pipelines JSON guardados, utilidades de grafos aceleradas de forma nativa y un constructor asistido por IA capaz de crear o modificar un pipeline a partir de una solicitud en lenguaje natural.

Ábrelo desde **Dev > Pipeline**. El **Developer Mode** debe estar habilitado en **Settings** si la pestaña **Dev** está oculta.

---

## Diseño

El constructor dispone de tres paneles:

| Área                    | Propósito                                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| Barra lateral izquierda | Ejemplos predefinidos, botones de bloques, lista de modelos y controles para guardar, cargar y previsualizar. |
| Lienzo                  | Editor desplazable de grafos de bloques.                                                                      |
| Barra lateral derecha   | Pestañas **Execution** y **AI Builder**.                                                                      |

Ambas barras laterales son redimensionables. Si una barra se reduce demasiado, se retrae en lugar de colapsar completamente. Haz clic en la flecha circular situada en el borde central del lienzo para volver a abrirla.

El texto, los botones y los controles del **AI Builder** escalan automáticamente según el ancho actual de la barra lateral.

---

## Ejemplos predefinidos

NativeLab incluye ejemplos de pipelines en:

```text
nativelab/pipelinebuilder/examples/
```

Estos ejemplos se distribuyen junto con la aplicación y aparecen en el menú desplegable **Example Presets**.

Los ejemplos actuales incluyen:

* `quick-answer`
* `clean-summarize`
* `draft-review`
* `triage-router`
* `llm-classify-and-respond`
* `llm-quality-gate`
* `research-synthesis-fanout`
* `briefing-pack-builder`

Selecciona un modelo en la lista antes de elegir un ejemplo si deseas que los bloques de modelo vacíos se rellenen automáticamente.

Si no se selecciona ningún modelo, el ejemplo seguirá cargándose y dejará los bloques de modelo preparados para asignarlos manualmente.

Cuando un ejemplo o un JSON cargado contiene bloques fuera del área visible del lienzo, NativeLab amplía automáticamente el lienzo para que todo el grafo permanezca accesible.

---

## Edición del lienzo

### Acciones básicas

| Acción                     | Cómo hacerlo                                                                  |
| -------------------------- | ----------------------------------------------------------------------------- |
| Añadir un bloque           | Haz clic en un botón de bloque de la barra lateral izquierda.                 |
| Añadir un bloque de modelo | Haz doble clic o arrastra un modelo desde la lista de modelos.                |
| Mover un bloque            | Arrástralo; al soltarlo se ajustará automáticamente a la cuadrícula de 20 px. |
| Desplazar el lienzo        | Haz clic y mantén pulsado sobre un área vacía del lienzo, luego arrastra.     |
| Conectar bloques           | Arrastra desde un puerto hasta otro puerto.                                   |
| Eliminar o configurar      | Haz clic derecho sobre un bloque o una conexión.                              |
| Previsualizar el flujo     | Haz clic en **Preview Flow** después de conectar los bloques.                 |
| Guardar o cargar           | Usa **Save Pipeline...** y **Load Pipeline...**.                              |

### Reglas de seguridad

* Todo pipeline necesita al menos un bloque **Input** y un bloque **Output**.
* Las conexiones directas entre modelos están bloqueadas; utiliza un bloque **Intermediate**, **Transform**, **Filter**, **Merge** u otro bloque lógico entre llamadas a modelos.
* Las conexiones duplicadas se ignoran.
* Los identificadores de bloques se normalizan cuando es necesario para evitar reutilizar accidentalmente un ID existente al pegar, generar o cargar un JSON.
* Las conexiones de bucle poseen límites explícitos de visitas.
* Los bloques **Custom Code** se ejecutan en un espacio de nombres restringido. No están disponibles las importaciones, el acceso al sistema de archivos, la red ni los subprocesos.

---

## Tipos de bloques

### Entrada, salida y modelos

| Bloque       | Uso                                                    |
| ------------ | ------------------------------------------------------ |
| Input        | Punto inicial obligatorio para el texto del usuario.   |
| Output       | Destino final obligatorio para el resultado.           |
| Model        | Ejecuta un modelo local, API, Ollama o HF cargado.     |
| Intermediate | Captura y transmite la salida intermedia del pipeline. |

### Contexto

| Bloque      | Uso                                                           |
| ----------- | ------------------------------------------------------------- |
| Reference   | Inserta texto de referencia estático.                         |
| Knowledge   | Inserta conocimiento reutilizable.                            |
| PDF Summary | Carga un PDF y lo resume o lo inserta según la configuración. |

### Lógica determinista

| Bloque      | Uso                                                                                                                         |
| ----------- | --------------------------------------------------------------------------------------------------------------------------- |
| IF / ELSE   | Evalúa una expresión Python segura sobre `text`; dirige la ejecución según verdadero o falso.                               |
| SWITCH      | Una expresión segura devuelve una clave que coincide con salidas etiquetadas.                                               |
| FILTER      | Permite continuar o detener el pipeline.                                                                                    |
| TRANSFORM   | Prefijos, sufijos, reemplazos, cambios de mayúsculas/minúsculas, eliminación de espacios, truncado y expresiones regulares. |
| MERGE       | Combina múltiples textos procedentes de bloques anteriores.                                                                 |
| SPLIT       | Divide una entrada en múltiples ramas.                                                                                      |
| Custom Code | Bloque Python determinista ejecutado en un entorno restringido.                                                             |

### Lógica basada en LLM

| Bloque        | Uso                                                                            |
| ------------- | ------------------------------------------------------------------------------ |
| LLM IF / ELSE | Enrutamiento sí/no mediante lenguaje natural.                                  |
| LLM SWITCH    | Clasificación mediante LLM hacia salidas etiquetadas.                          |
| LLM FILTER    | Decisión PASS/STOP.                                                            |
| LLM TRANSFORM | Reescribe o reformatea texto.                                                  |
| LLM SCORE     | Asigna una puntuación del 1 al 10 y dirige la ejecución hacia bajo/medio/alto. |

Los modelos pequeños suelen ser ideales para los bloques de enrutamiento LLM porque las respuestas esperadas son cortas y estructuradas.

---

## Pestaña Execution

La pestaña **Execution** ejecuta el lienzo actual utilizando exactamente la misma ruta de validación empleada por los pipelines guardados y por la ejecución de pipelines desde la CLI.

Antes de ejecutar, NativeLab verifica:

* Existencia de bloques **Input** y **Output** obligatorios.
* Referencias válidas para los bloques de modelo.
* Configuración correcta del contexto, PDF y metadatos lógicos.
* Conexiones válidas entre bloques.
* Existencia y sintaxis del bloque **Custom Code**.
* Asociación de modelos e instrucciones para los bloques de lógica LLM.

El registro de ejecución muestra:

* Inicio de bloques.
* Decisiones de ramificación.
* Transformaciones.
* Fusiones.
* Detenciones por filtros.
* División de ramas.
* Iteraciones de bucles.
* Resultado final.

Los errores del LLM o del runtime se muestran mediante el cuadro de diálogo centralizado de errores de NativeLab, por lo que los errores de ventana de contexto o fallos del servidor/API aparecen como mensajes claros para el usuario y no únicamente en los registros.

---

## Pestaña AI Builder

La pestaña **AI Builder** convierte una solicitud escrita en lenguaje natural en un pipeline JSON de NativeLab, lo valida, lo guarda y permite cargarlo para realizar pruebas.

### Flujo básico

1. Carga primero un modelo.
2. Abre **Dev > Pipeline > AI Builder**.
3. Introduce un nombre para el archivo JSON de salida.
4. Describe el pipeline que deseas crear.
5. Haz clic en **Build & Save**.
6. Haz clic en **Load / Test** para colocar el pipeline generado en el lienzo.

El modelo activo recibe:

* Una guía compacta para construir pipelines.
* El nombre solicitado para el archivo.
* La etiqueta del modelo activo o seleccionado.
* Tu solicitud.

La respuesta debe ser un único objeto JSON.

NativeLab extrae el JSON, lo normaliza, asigna automáticamente el modelo activo a los bloques de modelo vacíos cuando es posible, lo valida y lo guarda utilizando el sistema normal de pipelines.

Si la primera respuesta del modelo no contiene un JSON válido, NativeLab realiza automáticamente un segundo intento utilizando un prompt más estricto que solo acepta JSON y registra una vista previa de la respuesta inválida.

### Comprobación previa del contexto

Antes de enviar la solicitud, AI Builder estima:

* Tokens de entrada correspondientes a la guía más tu solicitud.
* Tokens reservados para el JSON generado.
* Total estimado frente al límite de contexto del modelo cargado.

Si la solicitud supera el contexto disponible, NativeLab bloquea el envío antes de llamar al modelo y solicita aumentar el contexto, recargar el modelo o acortar la solicitud.

### Contexto inteligente

AI Builder admite edición iterativa.

* Historial vacío + lienzo vacío: tu prompt se envía tal cual.
* Historial vacío + lienzo existente: NativeLab incluye el JSON del lienzo actual para que el modelo modifique el grafo existente en lugar de crear uno nuevo.
* Con historial existente: NativeLab incluye un resumen compacto del contexto previo, los turnos recientes y los datos actuales del lienzo.

Comandos disponibles dentro del cuadro de solicitud:

| Comando     | Efecto                                                                                                                             |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `/get_data` | Muestra el estado actual del lienzo en formato JSON.                                                                               |
| `/context`  | Compacta el historial de AI Builder utilizando el modelo activo cuando es posible o un método determinista local como alternativa. |

El historial se guarda en:

```text
localllm/pipeline_builder_history/default.json
```

### Reglas del JSON generado

AI Builder acepta exactamente el mismo esquema persistente utilizado por los pipelines guardados:

```json
{
  "version": 2,
  "title": "short name",
  "description": "short purpose",
  "blocks": [
    {
      "bid": 1,
      "btype": "input",
      "x": 80,
      "y": 120,
      "w": 148,
      "h": 76,
      "model_path": "",
      "role": "general",
      "label": "Input",
      "metadata": {}
    }
  ],
  "connections": [
    {
      "from_block_id": 1,
      "from_port": "E",
      "to_block_id": 2,
      "to_port": "W",
      "is_loop": false,
      "loop_times": 1
    }
  ]
}
```

Los bloques respaldados por modelos pueden dejar vacío el campo `model_path`.

NativeLab lo completará automáticamente utilizando el modelo seleccionado o el motor activo cuando sea posible.

---

## Núcleo nativo del Pipeline

Python continúa siendo la capa responsable de la interfaz y la orquestación.

Las rutas deterministas críticas se ejecutan mediante utilidades nativas cuando están disponibles:

* `nativelab/native/pipeline_core.c` gestiona la normalización de identificadores de bloques, reasignación de conexiones, detección de ciclos, utilidades de transformación y combinación, selección de rutas, límites de iteraciones y registros de validación.
* `nativelab/native/pipeline_core.py` actúa como wrapper y alternativa en Python cuando la extensión nativa no está disponible.
* `nativelab/pipelinebuilder/graph_ops.py` centraliza el comportamiento de los grafos.
* `nativelab/pipelinebuilder/execution_core.py` centraliza los componentes deterministas de ejecución.
* `nativelab/pipelinebuilder/validation.py` centraliza los mensajes de validación y mantiene la misma lógica en GUI, CLI y pipelines generados.

Las utilidades de AI Builder también disponen de implementaciones nativas en C y Rust para estimación de tokens y detección de objetos JSON, con comportamiento alternativo en Python.

La capa nativa es completamente opcional.

Mejora el rendimiento de las operaciones deterministas del pipeline sin controlar widgets Qt, llamadas a modelos, subprocesos, plugins ni cuadros de diálogo visibles para el usuario.

---

## Solución de problemas

### AI Builder indica que el contexto es demasiado pequeño

Aumenta el límite de contexto del modelo y vuelve a cargarlo, o reduce el tamaño de la solicitud.

El JSON generado reserva tokens de salida, por lo que una solicitud puede superar el contexto incluso si el prompt visible parece corto.

### El modelo no devolvió JSON

NativeLab realiza automáticamente un segundo intento.

Si vuelve a fallar, utiliza una solicitud más directa, por ejemplo:

```text
Make a 3 block input -> model -> output pipeline.
```

### El pipeline generado no supera la validación

El validador rechaza grafos inseguros o incompletos.

Revisa el mensaje de error y solicita al AI Builder que corrija el pipeline o modifica manualmente los bloques.

### El contenido de la barra lateral desaparece al redimensionarla

Las barras laterales ajustan automáticamente el tamaño del texto y de los controles.

Si la barra lateral derecha se vuelve demasiado estrecha, las etiquetas cambian a `Exec` y `AI`.

Si continúas reduciendo el ancho, la barra se retrae y aparece una flecha en el centro del lienzo para volver a abrirla.
