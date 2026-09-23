# Web Search & SearXNG Integration

NativeLab PC Client features a built-in, local-first web search subsystem. It enables LLMs, pipelines, and autonomous workflows to query the live internet for up-to-date information, real-time facts, and research citations—without requiring external search API subscriptions, proprietary keys, or third-party cloud telemetry.

---

## Architecture Overview

The web search functionality in NativeLab is powered by an embedded, in-process integration of the open-source **SearXNG** metasearch engine located at `nativelab/web_search/`.

```
nativelab/web_search/
├── __init__.py          # Public Python API & search wrappers
└── searxng/             # Embedded SearXNG core
    ├── requirements.txt # Python dependencies
    ├── SUBSYSTEM.md     # Subsystem documentation
    └── searx/           # SearXNG engine internals, parsers, and settings
        └── settings.yml # Engine definitions and search configuration
```

### In-Process Design
Unlike traditional SearXNG deployments that require Docker containers, Redis servers, Nginx reverse proxies, and web UI templates, NativeLab embeds a streamlined Python-only SearXNG core directly into the application process:
- **Zero Daemon Overhead**: Runs directly within the NativeLab Python runtime without needing a separate daemon or Docker container.
- **Lazy Initialization**: The search subsystem initializes on the first search request (`_ensure_initialized()`), keeping initial application launch instant.
- **Scoped Request Context**: Uses an internal Flask application context (`test_request_context`) for request lifecycle handling.
- **Clean Logging**: Per-engine parser noise and network timeouts from individual search providers are isolated and suppressed to keep NativeLab logs readable.

---

## Python API

The `nativelab.web_search` package provides two high-level functions:

### 1. `web_search(query, ...)`
Executes a metasearch across selected engine categories and returns structured Python dictionaries.

```python
from nativelab.web_search import web_search

results = web_search(
    query="latest discoveries in quantum computing",
    categories=["science", "general"],
    language="en",
    max_results=5,
    safesearch=1,
    time_range="month", # "day", "week", "month", "year", or None
    timeout=10.0,
)

for item in results:
    print(f"Title:   {item['title']}")
    print(f"URL:     {item['url']}")
    print(f"Snippet: {item['content']}")
    print(f"Engine:  {item['engine']}")
    print(f"Score:   {item['score']}\n")
```

**Returned Object Schema**:
| Field | Type | Description |
|---|---|---|
| `title` | `str` | Title of the web page or search hit. |
| `url` | `str` | Destination URL. |
| `content` | `str` | Text snippet or summary extracted by the engine parser. |
| `engine` | `str` | Originating engine (e.g., `duckduckgo`, `wikipedia`, `brave`, `bing`, `google`). |
| `score` | `float` | SearXNG relevance and ranking score. |
| `category` | `str` | Matched search category (e.g., `general`, `science`, `news`). |

### 2. `web_search_text(query, ...)`
Convenience wrapper that searches and formats the results into human- and LLM-readable text, ideal for direct prompt and context injection.

```python
from nativelab.web_search import web_search_text

formatted_context = web_search_text(
    "llama.cpp metal acceleration guide",
    categories=["general", "it"],
    max_results=5,
)
print(formatted_context)
```

**Sample Output**:
```text
Web search results for: llama.cpp metal acceleration guide

1. Build with Metal - llama.cpp Documentation
   URL: https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md
   To enable Metal GPU support on Apple Silicon macOS, pass GGML_METAL=ON...

2. Running Local Models on macOS
   URL: https://example.org/guides/macos-llm
   Step-by-step walkthrough of running quantized GGUF models using Metal acceleration...
```

---

## PC Client & Pipeline Builder Integration

NativeLab natively integrates Web Search into the visual pipeline builder and AI builder workflows.

### 1. Web Search Pipeline Block
In the **Dev > Pipeline** builder, you can add a dedicated **Web Search** block (colored orange `#f97316` with the search icon) to any workflow.

- **Dynamic Query Routing**: The Web Search block takes whatever text is passed into its input port (from user input, an intermediate LLM prompt, or a transform block) and executes a web search with that string as the query.
- **Configurable Block Settings**: Right-click the block and choose **Configure block...**:
  - **Categories**: Check one or more categories: `general`, `images`, `videos`, `news`, `science`, `it`, `files`, `music`, `social media`.
  - **Language**: Set ISO language code (e.g. `en`, `es`, `fr`, `de`, `all`).
  - **Max Results**: Set maximum number of hits (1–50, default: 10).
  - **Timeout**: Per-engine network timeout limit in seconds (default: 10).
  - **Output Format**: 
    - `text`: Pre-formatted Markdown/text list with titles, URLs, and snippet excerpts (recommended for downstream LLM prompts).
    - `json`: Formatted JSON array for downstream code, scripts, or structural extractors.
  - **Test Search Button**: An interactive button in the configuration dialog allows testing search connectivity and verifying results live before running the pipeline.

### 2. AI Pipeline Builder Support
When using the **AI Builder** tab to generate pipelines from natural language (e.g., *"Create a pipeline that takes a user topic, searches the web for recent science news, and summarizes the findings with a model"*), the AI planner recognizes the research intent and automatically scaffolds `web_search` blocks configured with relevant categories.

### 3. Execution & Fault Tolerance
During execution (`executionWorker.py`):
- If the incoming query is empty, the block logs a warning and cleanly passes through the context.
- If no results are returned (due to strict filters or network issues), it passes through a fallback message (`[Web search returned no results for: ...]`) preventing downstream model crashes.
- If dependencies are missing, a clear diagnostic message directs the user to the installation instructions.

---

## Installation & Prerequisites

To enable web search functionality, install the required dependencies for the embedded SearXNG subsystem:

```bash
# From the repository root
pip install -r nativelab/web_search/searxng/requirements.txt
```

Key dependencies include:
- `httpx` / `requests` / `urllib3` (Async & sync HTTP engine dispatch)
- `lxml` / `beautifulsoup4` (HTML parsing and snippet extraction)
- `pyyaml` (Engine configuration loader)
- `flask` (Internal request context management)
- `fasttext-wheel` (Language identification)
- `babel` / `certifi` / `dateutil` / `jinja2`

---

## Engine Configuration

You can customize which search engines SearXNG queries by editing:
`nativelab/web_search/searxng/searx/settings.yml`

Common customizations:
- **Enable/Disable Engines**: Toggle engines under the `engines:` section (e.g. DuckDuckGo, Wikipedia, Qwant, Brave, Google, Bing, Startpage, arXiv, GitHub, etc.).
- **Default Language**: Change `search.default_lang`.
- **Outgoing Timeouts & Limits**: Adjust `outgoing.request_timeout` and `outgoing.max_request_timeout`.

---

## Credits & Licensing

### Attribution to SearXNG
NativeLab's web search engine embeds and adapts code from the **SearXNG** project:
- **Project**: SearXNG (A privacy-respecting, hackable metasearch engine)
- **Source Repository**: [https://github.com/searxng/searxng](https://github.com/searxng/searxng)
- **Original Predecessor**: Searx by Adam Tauber (asciimoo) and community contributors.

We extend our sincere gratitude and full credit to the SearXNG community and all upstream contributors for their pioneering work on open-source, privacy-preserving metasearch.

### Honest Licensing & Compliance
- **SearXNG License**: SearXNG is distributed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
- **NativeLab License**: NativeLab is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
- **Copyleft Integrity**: Both projects share the same copyleft license (AGPLv3), ensuring full legal alignment and preserving the rights of all end users.
- **Modifications Disclosed**: NativeLab embeds the SearXNG core in `nativelab/web_search/searxng/` with stripped UI templates, web server endpoints, and translation catalogs to enable efficient, headless, in-process search execution. All Python search core logic and adaptations remain fully open and accessible under AGPLv3.
