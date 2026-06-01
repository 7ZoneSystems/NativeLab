from __future__ import annotations

import json
import urllib.error


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


def normalize_ollama_host(host: str = "") -> str:
    value = str(host or DEFAULT_OLLAMA_HOST).strip().rstrip("/")
    if not value:
        value = DEFAULT_OLLAMA_HOST
    if "://" not in value:
        value = f"http://{value}"
    return value.rstrip("/")


def normalize_ollama_exception(exc: Exception, host: str = "", action: str = "connect to") -> str:
    base = normalize_ollama_host(host)
    if isinstance(exc, urllib.error.HTTPError):
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        detail = raw[:500] or str(getattr(exc, "reason", "") or "")
        try:
            payload = json.loads(raw or "{}")
            detail = str(payload.get("error") or detail)
        except Exception:
            pass
        return f"Ollama HTTP {exc.code} at {base}: {detail}"

    reason = getattr(exc, "reason", None)
    text = str(reason or exc)
    lowered = text.lower()
    if "connection refused" in lowered or isinstance(reason, ConnectionRefusedError):
        return (
            f"Could not reach Ollama at {base}. Start the Ollama app or run "
            "`ollama serve`, then retry. If Ollama uses another address, update "
            "Settings > App Configuration > Ollama Host or the Download tab Host field."
        )
    if "timed out" in lowered or isinstance(reason, TimeoutError):
        return f"Ollama did not respond at {base}. Check that the daemon is healthy, then retry."
    if "name or service not known" in lowered or "temporary failure in name resolution" in lowered:
        return f"Ollama host could not be resolved: {base}. Check the configured Ollama Host."
    return f"Could not {action} Ollama at {base}: {text}"
