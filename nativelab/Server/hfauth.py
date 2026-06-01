from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from nativelab.GlobalConfig.timeouts import LONG_TIMEOUT_SECONDS

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except Exception:
    class QThread:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    class _MissingSignal:
        def connect(self, *args, **kwargs):
            pass
        def emit(self, *args, **kwargs):
            pass

    def pyqtSignal(*args, **kwargs):  # type: ignore[no-redef]
        return _MissingSignal()


HF_CRED_FILE = Path("./localllm/cred/huggingface.json")
APP_CONFIG_FILE = Path("app_config.json")
HF_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
HF_DEVICE_URL = "https://huggingface.co/oauth/device"
HF_TOKEN_URL = "https://huggingface.co/oauth/token"
HF_TOKEN_SETTINGS_URL = "https://huggingface.co/settings/tokens"
HF_REQUIRED_MESSAGE = "Hugging Face authorization required. Sign in from Accounts > Hugging Face."
HF_DEFAULT_SCOPE = "openid profile read-repos gated-repos"
HF_BUILTIN_OAUTH_CLIENT_ID = "bdc0955b-9345-41a0-8077-29422220600a"
HF_OAUTH_CLIENT_ID_ENV = "NATIVELAB_HF_OAUTH_CLIENT_ID"
HF_BAD_CLIENT_ID_MESSAGE = (
    "Previous Hugging Face setup used an access token as an OAuth client ID. "
    "Use Login with Hugging Face, or paste the token into Advanced access token."
)

_SAFE_KEYS = {
    "client_id", "token_type", "scope", "expires_at", "username",
    "last_validated_at", "last_error",
}


def _now() -> int:
    return int(time.time())


def _default_creds() -> Dict[str, Any]:
    return {
        "client_id": "",
        "access_token": "",
        "refresh_token": "",
        "token_type": "",
        "scope": "",
        "expires_at": 0,
        "username": "",
        "last_validated_at": 0,
        "last_error": "",
    }


def load_hf_credentials() -> Dict[str, Any]:
    if not HF_CRED_FILE.exists():
        return _default_creds()
    try:
        raw = json.loads(HF_CRED_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return _default_creds()
    except Exception:
        return _default_creds()
    data = _default_creds()
    data.update(raw)
    if _cleanup_bad_client_id_state(data):
        _write_hf_credentials(data)
    return data


def _write_hf_credentials(data: Dict[str, Any]) -> Dict[str, Any]:
    HF_CRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    HF_CRED_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        HF_CRED_FILE.chmod(0o600)
    except Exception:
        pass
    return data


def _cleanup_bad_client_id_state(data: Dict[str, Any]) -> bool:
    client_id = str(data.get("client_id", "") or "").strip()
    token = str(data.get("access_token", "") or "").strip()
    if not token and client_id.startswith("hf_"):
        data["client_id"] = ""
        data["last_error"] = HF_BAD_CLIENT_ID_MESSAGE
        return True
    last_error = str(data.get("last_error", "") or "")
    if "invalid_client" in last_error:
        data["last_error"] = HF_BAD_CLIENT_ID_MESSAGE
        return True
    return False


def save_hf_credentials(data: Dict[str, Any]) -> Dict[str, Any]:
    current = load_hf_credentials()
    current.update(data or {})
    _cleanup_bad_client_id_state(current)
    return _write_hf_credentials(current)


def get_hf_oauth_client_id() -> str:
    return str(os.environ.get(HF_OAUTH_CLIENT_ID_ENV) or HF_BUILTIN_OAUTH_CLIENT_ID).strip()


def clear_hf_credentials(keep_client_id: bool = True) -> Dict[str, Any]:
    current = load_hf_credentials()
    client_id = current.get("client_id", "") if keep_client_id else ""
    data = _default_creds()
    data["client_id"] = client_id
    return save_hf_credentials(data)


def mask_hf_token(token: str) -> str:
    token = str(token or "").strip()
    if not token:
        return ""
    if len(token) <= 10:
        return token[:2] + "***"
    return token[:6] + "..." + token[-4:]


def redacted_credentials(data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    src = load_hf_credentials() if data is None else dict(data)
    out = {k: src.get(k, "") for k in _SAFE_KEYS}
    out["access_token"] = mask_hf_token(src.get("access_token", ""))
    out["refresh_token"] = "***" if src.get("refresh_token") else ""
    return out


def get_hf_access_token() -> str:
    token = str(load_hf_credentials().get("access_token", "") or "").strip()
    if token:
        return token
    try:
        data = json.loads(APP_CONFIG_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return str(data.get("hf_token", "") or "").strip()
    except Exception:
        pass
    return ""


def hf_auth_headers(token: Optional[str] = None, user_agent: str = "NativeLabPro/2") -> Dict[str, str]:
    headers = {"User-Agent": user_agent}
    tok = str(token or get_hf_access_token() or "").strip()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers


def hf_repo_url(repo_id: str = "") -> str:
    repo = str(repo_id or "").strip().strip("/")
    return f"https://huggingface.co/{repo}" if repo else "https://huggingface.co/models"


def _hf_access_denied_message(repo_id: str = "") -> str:
    if get_hf_access_token():
        repo_txt = str(repo_id or "this repository").strip()
        return (
            f"Hugging Face access denied for {repo_txt}. You are signed in, but this "
            "account may not have accepted the gated model terms, the access request may "
            "still be pending, or the request may have been rejected. Open "
            f"{hf_repo_url(repo_id)} in your browser, accept/request access, wait for "
            "approval if required, then retry. If access was just granted, logout and "
            "login again from Accounts > Hugging Face to refresh the token."
        )
    return HF_REQUIRED_MESSAGE


def _http_error_text(exc: urllib.error.HTTPError, repo_id: str = "") -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:
        raw = ""
    if exc.code == 400:
        try:
            payload = json.loads(raw or "{}")
        except Exception:
            payload = {}
        if payload.get("error") == "invalid_client":
            return HF_BAD_CLIENT_ID_MESSAGE
    if exc.code == 401:
        return f"{HF_REQUIRED_MESSAGE} HTTP {exc.code}."
    if exc.code == 403:
        return f"{_hf_access_denied_message(repo_id)} HTTP {exc.code}."
    return f"Hugging Face HTTP {exc.code}: {raw[:300] or exc.reason}"


def normalize_hf_exception(exc: Exception, repo_id: str = "") -> str:
    if isinstance(exc, urllib.error.HTTPError):
        msg = _http_error_text(exc, repo_id=repo_id)
        if exc.code in (401, 403):
            save_hf_credentials({"last_error": msg})
        return msg
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code == 401:
        msg = f"{HF_REQUIRED_MESSAGE} HTTP {status_code}."
        save_hf_credentials({"last_error": msg})
        return msg
    if status_code == 403:
        msg = f"{_hf_access_denied_message(repo_id)} HTTP {status_code}."
        save_hf_credentials({"last_error": msg})
        return msg
    text = str(exc)
    if "403 Client Error" in text or "Forbidden" in text:
        msg = _hf_access_denied_message(repo_id)
        save_hf_credentials({"last_error": msg})
        return msg
    if "401 Client Error" in text or "Unauthorized" in text:
        msg = HF_REQUIRED_MESSAGE
        save_hf_credentials({"last_error": msg})
        return msg
    return text


def validate_hf_token(token: Optional[str] = None) -> Dict[str, Any]:
    tok = str(token or get_hf_access_token() or "").strip()
    if not tok:
        raise RuntimeError("No Hugging Face token saved.")
    req = urllib.request.Request(HF_WHOAMI_URL, headers=hf_auth_headers(tok))
    try:
        with urllib.request.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        raise RuntimeError(normalize_hf_exception(exc)) from exc
    username = (
        data.get("name")
        or data.get("preferred_username")
        or data.get("fullname")
        or data.get("sub")
        or ""
    )
    saved = save_hf_credentials({
        "username": username,
        "last_validated_at": _now(),
        "last_error": "",
    })
    result = dict(data)
    result["_saved"] = redacted_credentials(saved)
    return result


def save_hf_access_token(token: str) -> Dict[str, Any]:
    tok = str(token or "").strip()
    if not tok:
        raise RuntimeError("Paste a Hugging Face access token first.")
    validate_hf_token(tok)
    saved = save_hf_credentials({
        "access_token": tok,
        "refresh_token": "",
        "token_type": "bearer",
        "scope": "manual-token",
        "expires_at": 0,
        "last_error": "",
    })
    return redacted_credentials(saved)


def request_device_code(client_id: str) -> Dict[str, Any]:
    body = urllib.parse.urlencode({
        "client_id": client_id,
        "scope": HF_DEFAULT_SCOPE,
    }).encode("utf-8")
    req = urllib.request.Request(
        HF_DEVICE_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "NativeLabPro/2"},
    )
    with urllib.request.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def poll_device_token(client_id: str, device_code: str, interval: int, expires_in: int, status_cb=None, abort_cb=None) -> Dict[str, Any]:
    deadline = time.time() + int(expires_in or 900)
    wait = max(2, int(interval or 5))
    while time.time() < deadline:
        if abort_cb and abort_cb():
            raise RuntimeError("Hugging Face login cancelled.")
        body = urllib.parse.urlencode({
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code,
            "client_id": client_id,
        }).encode("utf-8")
        req = urllib.request.Request(
            HF_TOKEN_URL,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "NativeLabPro/2"},
        )
        try:
            with urllib.request.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8", errors="replace") or "{}")
            except Exception:
                payload = {}
            err = str(payload.get("error") or "")
            if err == "authorization_pending":
                if status_cb:
                    status_cb("Waiting for Hugging Face authorization...")
                time.sleep(wait)
                continue
            if err == "slow_down":
                wait += 5
                if status_cb:
                    status_cb("Hugging Face asked to slow down; waiting...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Hugging Face OAuth failed: {payload.get('error_description') or err or exc.code}") from exc
    raise RuntimeError("Hugging Face login expired before authorization completed.")


class HfOAuthLoginWorker(QThread):
    code_ready = pyqtSignal(str, str)
    status = pyqtSignal(str)
    done = pyqtSignal(dict)
    err = pyqtSignal(str)

    def __init__(self, client_id: str = ""):
        super().__init__()
        self.client_id = (client_id or get_hf_oauth_client_id()).strip()
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            if not self.client_id:
                raise RuntimeError("Hugging Face OAuth client ID is not configured.")
            save_hf_credentials({"client_id": self.client_id, "last_error": ""})
            self.status.emit("Requesting Hugging Face device login...")
            device = request_device_code(self.client_id)
            user_code = str(device.get("user_code") or "")
            verification_uri = str(
                device.get("verification_uri_complete")
                or device.get("verification_uri")
                or ""
            )
            if not device.get("device_code") or not verification_uri:
                raise RuntimeError("Hugging Face did not return a device login URL.")
            self.code_ready.emit(user_code, verification_uri)
            token_data = poll_device_token(
                self.client_id,
                str(device.get("device_code")),
                int(device.get("interval") or 5),
                int(device.get("expires_in") or 900),
                status_cb=self.status.emit,
                abort_cb=lambda: self._abort,
            )
            expires_in = int(token_data.get("expires_in") or 0)
            save_hf_credentials({
                "client_id": self.client_id,
                "access_token": token_data.get("access_token", ""),
                "refresh_token": token_data.get("refresh_token", ""),
                "token_type": token_data.get("token_type", ""),
                "scope": token_data.get("scope", ""),
                "expires_at": (_now() + expires_in) if expires_in else 0,
                "last_error": "",
            })
            self.status.emit("Validating Hugging Face token...")
            who = validate_hf_token(token_data.get("access_token", ""))
            self.done.emit(redacted_credentials(who.get("_saved", {})))
        except Exception as exc:
            msg = normalize_hf_exception(exc)
            save_hf_credentials({"last_error": msg})
            self.err.emit(msg)


class HfDirectTokenWorker(QThread):
    done = pyqtSignal(dict)
    err = pyqtSignal(str)

    def __init__(self, token: str):
        super().__init__()
        self.token = str(token or "").strip()

    def run(self):
        try:
            self.done.emit(save_hf_access_token(self.token))
        except Exception as exc:
            msg = normalize_hf_exception(exc)
            save_hf_credentials({"last_error": msg})
            self.err.emit(msg)


class HfValidateWorker(QThread):
    done = pyqtSignal(dict)
    err = pyqtSignal(str)

    def run(self):
        try:
            validate_hf_token()
            self.done.emit(redacted_credentials())
        except Exception as exc:
            msg = normalize_hf_exception(exc)
            save_hf_credentials({"last_error": msg})
            self.err.emit(msg)
