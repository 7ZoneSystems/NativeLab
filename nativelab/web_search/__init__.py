"""
NativeLab Web Search - in-process SearXNG wrapper for pipeline builder.

Usage:
    from nativelab.web_search import web_search

    results = web_search("python async tutorial")
    for r in results:
        print(r["title"], r["url"])
"""
from __future__ import annotations

import typing as t

_initialized = False
_flask_app = None
_path_patched = False


def _ensure_initialized():
    """Lazy-initialize SearXNG search engine on first call."""
    global _initialized, _flask_app, _path_patched
    if _initialized:
        return

    # Defer sys.path modification until actually needed
    if not _path_patched:
        import sys
        import os
        searxng_dir = os.path.join(os.path.dirname(__file__), "searxng")
        if searxng_dir not in sys.path:
            sys.path.insert(0, searxng_dir)
        _path_patched = True

    try:
        # Suppress noisy SearXNG engine loading errors
        import logging
        searx_logger = logging.getLogger("searx.engines")
        old_level = searx_logger.level
        searx_logger.setLevel(logging.CRITICAL)

        from flask import Flask
        _flask_app = Flask("nativelab_searxng")
        _flask_app.config["SECRET_KEY"] = "nativelab-embedded"

        from searx.search import initialize as searx_initialize
        searx_initialize()

        searx_logger.setLevel(old_level)
        _initialized = True
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize SearXNG search engine: {e}\n"
            f"Ensure nativelab/web_search/searxng/ exists and "
            f"requirements.txt is installed."
        ) from e


def _build_engineref_list(categories: list[str] | None) -> list:
    """Build EngineRef list from category names."""
    from searx.engines import categories as engine_categories
    from searx.search.models import EngineRef

    if not categories:
        categories = ["general"]

    result = []
    for cat in categories:
        if cat in engine_categories:
            for engine in engine_categories[cat]:
                if not getattr(engine, "disabled", False):
                    result.append(EngineRef(engine.name, cat))
    return result


def web_search(
    query: str,
    *,
    categories: list[str] | None = None,
    language: str = "en",
    max_results: int = 20,
    safesearch: int = 0,
    time_range: str | None = None,
    timeout: float = 10,
) -> list[dict[str, t.Any]]:
    """
    Search the web using SearXNG in-process.

    Args:
        query: Search query string.
        categories: Engine categories (e.g. ["general"], ["images"], ["news"]).
                    None defaults to ["general"].
        language: Search language code (e.g. "en", "de", "all").
        max_results: Maximum number of results to return.
        safesearch: Safe search level (0=off, 1=moderate, 2=strict).
        time_range: Time range filter ("day", "week", "month", "year", None).
        timeout: Per-engine timeout in seconds.

    Returns:
        List of result dicts with keys: title, url, content, engine, score, ...
    """
    _ensure_initialized()

    from searx.search import Search
    from searx.search.models import SearchQuery

    engineref_list = _build_engineref_list(categories)

    search_query = SearchQuery(
        query=query,
        engineref_list=engineref_list,
        lang=language,
        safesearch=safesearch,
        pageno=1,
        time_range=time_range,
        timeout_limit=timeout,
    )

    try:
        import logging
        # Suppress noisy per-engine errors during search (Google parser breakage, etc.)
        searx_engines_logger = logging.getLogger("searx.engines")
        old_level = searx_engines_logger.level
        searx_engines_logger.setLevel(logging.CRITICAL)

        if _flask_app is None:
            searx_engines_logger.setLevel(old_level)
            return []
        with _flask_app.test_request_context():
            result_container = Search(search_query).search()

        searx_engines_logger.setLevel(old_level)
    except Exception as e:
        import logging
        logging.getLogger("nativelab.web_search").warning("Search failed: %s", e)
        return []

    ordered = result_container.get_ordered_results()

    results: list[dict[str, t.Any]] = []
    for r in ordered[:max_results]:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "engine": r.get("engine", ""),
            "score": r.get("score", 0),
            "category": r.get("category", ""),
        })

    return results


def web_search_text(query: str, **kwargs) -> str:
    """
    Search and return results as formatted text (for pipeline builder context injection).
    """
    results = web_search(query, **kwargs)
    if not results:
        return f"No results found for: {query}"

    lines = [f"Web search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "")
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {url}")
        if content:
            lines.append(f"   {content[:200]}")
        lines.append("")

    return "\n".join(lines)
