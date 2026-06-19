# SearXNG Embedded Core

SearXNG search engine stripped for in-process use inside NativeLab.
No Docker, no web UI, no templates, no translations - pure Python search.

## Install

```bash
cd nativelab/web_search/searxng
pip install -r requirements.txt
```

## Usage (from NativeLab)

```python
from nativelab.web_search import web_search, web_search_text

# Returns list of dicts
results = web_search("python async tutorial", max_results=5)
for r in results:
    print(r["title"], r["url"])

# Returns formatted text (for pipeline context injection)
text = web_search_text("machine learning basics")
```

## Pipeline Builder Integration

Use `web_search` or `web_search_text` in Custom Code blocks:

```python
from nativelab.web_search import web_search_text
result = web_search_text(text, max_results=5)
```

## Key Settings (searx/settings.yml)

- `engines` - enable/disable individual search engines
- `search.default_lang` - default language
- `outgoing.request_timeout` - HTTP timeout for engine requests
