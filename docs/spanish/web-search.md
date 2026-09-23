# Búsqueda Web e Integración con SearXNG

El cliente de escritorio de NativeLab incluye un subsistema de búsqueda web local e integrado. Permite que los modelos de lenguaje (LLM), los pipelines y los flujos de trabajo autónomos consulten información actualizada en tiempo real e incorporen citas de investigación directamente desde internet, sin necesidad de suscripciones a API de búsqueda externas, claves propietarias ni telemetría de terceros en la nube.

---

## Visión general de la arquitectura

La funcionalidad de búsqueda web en NativeLab está impulsada por una integración embebida e *in-process* del motor de metabúsqueda de código abierto **SearXNG**, ubicado en `nativelab/web_search/`.

```
nativelab/web_search/
├── __init__.py          # API pública de Python y funciones de búsqueda
└── searxng/             # Núcleo embebido de SearXNG
    ├── requirements.txt # Dependencias de Python
    ├── SUBSYSTEM.md     # Documentación del subsistema
    └── searx/           # Motores internos, analizadores y configuración
        └── settings.yml # Definición de motores y parámetros de búsqueda
```

### Diseño *In-Process* (Embebido en el proceso)
A diferencia de los despliegues tradicionales de SearXNG que requieren contenedores Docker, servidores Redis, proxies Nginx y plantillas de interfaz web, NativeLab integra un núcleo de SearXNG optimizado exclusivamente en Python dentro del propio proceso de la aplicación:
- **Sin sobrecarga de demonios/servicios**: Se ejecuta directamente dentro del runtime de Python de NativeLab sin requerir contenedores Docker ni servicios externos en segundo plano.
- **Inicialización diferida (*Lazy Initialization*)**: El subsistema se inicializa en la primera búsqueda (`_ensure_initialized()`), garantizando un inicio de aplicación instantáneo.
- **Contexto de solicitud aislado**: Utiliza un contexto de aplicación Flask interno (`test_request_context`) para gestionar el ciclo de vida de cada consulta.
- **Registro limpio de eventos**: Los mensajes y tiempos de espera de motores de terceros individuales se aíslan y filtran para mantener los registros de NativeLab limpios y comprensibles.

---

## API de Python

El módulo `nativelab.web_search` expone dos funciones principales:

### 1. `web_search(query, ...)`
Ejecuta una metabúsqueda a través de las categorías de motores seleccionadas y devuelve diccionarios estructurados de Python.

```python
from nativelab.web_search import web_search

results = web_search(
    query="últimos avances en computación cuántica",
    categories=["science", "general"],
    language="es",
    max_results=5,
    safesearch=1,
    time_range="month", # "day", "week", "month", "year", o None
    timeout=10.0,
)

for item in results:
    print(f"Título:    {item['title']}")
    print(f"URL:       {item['url']}")
    print(f"Fragmento: {item['content']}")
    print(f"Motor:     {item['engine']}")
    print(f"Puntaje:   {item['score']}\n")
```

**Esquema del objeto de resultado**:
| Campo | Tipo | Descripción |
|---|---|---|
| `title` | `str` | Título de la página web o resultado de búsqueda. |
| `url` | `str` | URL de destino. |
| `content` | `str` | Fragmento de texto o resumen extraído por el analizador del motor. |
| `engine` | `str` | Motor de búsqueda de origen (por ejemplo: `duckduckgo`, `wikipedia`, `brave`, `bing`, `google`). |
| `score` | `float` | Puntuación de relevancia y clasificación de SearXNG. |
| `category` | `str` | Categoría asignada (por ejemplo: `general`, `science`, `news`). |

### 2. `web_search_text(query, ...)`
Función utilitaria que realiza la búsqueda y formatea los resultados en texto legible tanto por personas como por modelos LLM, facilitando la inyección directa en prompts y contextos.

```python
from nativelab.web_search import web_search_text

formatted_context = web_search_text(
    "guía de aceleración metal en llama.cpp",
    categories=["general", "it"],
    max_results=5,
)
print(formatted_context)
```

**Ejemplo de salida**:
```text
Web search results for: guía de aceleración metal en llama.cpp

1. Build with Metal - llama.cpp Documentation
   URL: https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md
   Para habilitar el soporte de GPU Metal en Apple Silicon macOS, configure GGML_METAL=ON...

2. Ejecución de modelos locales en macOS
   URL: https://example.org/guides/macos-llm
   Guía paso a paso para ejecutar modelos GGUF cuantizados aprovechando la aceleración Metal...
```

---

## Integración en el Cliente de Escritorio y Constructor de Pipelines

NativeLab integra la búsqueda web de forma nativa en el constructor visual de pipelines y en los flujos del AI Builder.

### 1. Bloque de Pipeline "Web Search"
En el constructor visual (**Dev > Pipeline**), puedes añadir el bloque **Web Search** (de color naranja `#f97316` y con icono de lupa) a cualquier flujo de trabajo.

- **Enrutamiento dinámico de consultas**: El bloque toma el texto recibido en su puerto de entrada (proveniente del usuario, de un prompt intermedio o de un bloque de transformación) y lo utiliza dinámicamente como término de búsqueda.
- **Configuración del bloque**: Haz clic derecho sobre el bloque y selecciona **Configure block...**:
  - **Categorías (`Categories`)**: Selecciona una o más categorías: `general`, `images`, `videos`, `news`, `science`, `it`, `files`, `music`, `social media`.
  - **Idioma (`Language`)**: Código de idioma ISO (por ejemplo: `en`, `es`, `fr`, `de`, `all`).
  - **Resultados máximos (`Max Results`)**: Número máximo de resultados (1–50, por defecto: 10).
  - **Tiempo de espera (`Timeout`)**: Límite de tiempo en segundos por motor (por defecto: 10).
  - **Formato de salida (`Output Format`)**:
    - `text`: Lista formateada en texto/Markdown con títulos, enlaces y fragmentos (recomendado para prompts de modelos LLM).
    - `json`: Estructura JSON válida para bloques de código o procesadores downstream.
  - **Botón de prueba ("Test Search")**: Permite probar la conectividad de búsqueda y comprobar los resultados en vivo directamente desde el diálogo de configuración antes de ejecutar el pipeline.

### 2. Soporte en el AI Pipeline Builder
Al utilizar la pestaña **AI Builder** para generar pipelines a partir de descripciones en lenguaje natural (por ejemplo: *"Crea un pipeline que tome un tema, busque noticias científicas recientes en la web y genere un resumen con el modelo"*), el planificador de IA reconoce la necesidad de búsqueda y añade automáticamente bloques `web_search` configurados con las categorías pertinentes.

### 3. Ejecución y tolerancia a fallos
Durante la ejecución del pipeline (`executionWorker.py`):
- Si la consulta de entrada está vacía, el bloque registra una advertencia y deja pasar el contexto original sin interrumpir el flujo.
- Si no se obtienen resultados (debido a filtros o desconexión de red), emite un mensaje de respaldo informativo (`[Web search returned no results for: ...]`) para evitar fallos en los modelos posteriores.
- Si las dependencias del módulo no están instaladas, muestra un diagnóstico claro indicando cómo instalarlas.

---

## Instalación y requisitos previos

Para habilitar las funciones de búsqueda web, instala las dependencias de Python del subsistema SearXNG:

```bash
# Desde la raíz del repositorio
pip install -r nativelab/web_search/searxng/requirements.txt
```

Las dependencias principales incluyen:
- `httpx` / `requests` / `urllib3` (Envío síncrono y asíncrono de solicitudes a motores)
- `lxml` / `beautifulsoup4` (Procesamiento de HTML y extracción de fragmentos)
- `pyyaml` (Carga y lectura de configuración de motores)
- `flask` (Gestión del contexto interno de peticiones)
- `fasttext-wheel` (Identificación automática de idiomas)
- `babel` / `certifi` / `dateutil` / `jinja2`

---

## Configuración de motores de búsqueda

Es posible personalizar los motores de búsqueda que consulta SearXNG editando:
`nativelab/web_search/searxng/searx/settings.yml`

Ajustes comunes:
- **Activar/Desactivar motores**: Modifica la sección `engines:` (DuckDuckGo, Wikipedia, Qwant, Brave, Google, Bing, Startpage, arXiv, GitHub, etc.).
- **Idioma predeterminado**: Ajusta la clave `search.default_lang`.
- **Tiempos de espera**: Configura `outgoing.request_timeout` y `outgoing.max_request_timeout`.

---

## Créditos y licenciamiento honesto

### Atribución a SearXNG
El motor de búsqueda web de NativeLab incorpora y adapta código del proyecto **SearXNG**:
- **Proyecto**: SearXNG (Un motor de metabúsqueda hackeable y respetuoso con la privacidad)
- **Repositorio fuente**: [https://github.com/searxng/searxng](https://github.com/searxng/searxng)
- **Predecesor original**: Searx, creado por Adam Tauber (asciimoo) y la comunidad de colaboradores.

Extendemos nuestro sincero agradecimiento y pleno reconocimiento a la comunidad de SearXNG y a todos sus colaboradores por su extraordinario trabajo en la creación de herramientas de búsqueda abiertas y privadas.

### Licenciamiento honesto y cumplimiento
- **Licencia de SearXNG**: SearXNG se distribuye bajo la licencia **GNU Affero General Public License v3.0 (AGPL-3.0)**.
- **Licencia de NativeLab**: NativeLab se distribuye bajo la licencia **GNU Affero General Public License v3.0 (AGPL-3.0)**.
- **Integridad Copyleft**: Ambos proyectos comparten la misma licencia de software libre (AGPLv3), garantizando plena compatibilidad legal y protegiendo los derechos de todos los usuarios.
- **Transparencia en las modificaciones**: NativeLab aloja el núcleo de SearXNG en `nativelab/web_search/searxng/` tras haber retirado las plantillas HTML de interfaz web, las páginas de traducción y los servidores web independientes para permitir una ejecución embebida y sin interfaz gráfica (*headless*). Toda la lógica de búsqueda en Python y sus adaptaciones permanecen completamente abiertas bajo los términos de la AGPLv3.
