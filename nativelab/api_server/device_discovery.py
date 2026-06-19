"""
Network discovery for PhonoLab devices on the local LAN.

Scans the local subnet for PhonoLab API servers (default port 8787),
queries their /health and /device endpoints, and returns device info.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from nativelab.api_server.config import detect_lan_ip

DISCOVERY_PORT = 8787
DISCOVERY_TIMEOUT = 1.5  # seconds per host
DISCOVERY_CACHE_FILE = Path("./localllm/discovered_devices.json")


@dataclass
class DiscoveredDevice:
    """A PhonoLab device found on the network."""
    ip: str
    port: int = 8787
    name: str = ""              # Device name (manufacturer + model)
    model: str = ""             # Currently loaded model
    status: str = "unknown"     # idle/ready/generating/error
    is_vision: bool = False
    cpu_cores: int = 0
    ram_mb: int = 0
    android_version: str = ""
    api_key: str = ""           # LAN API key (if known)
    last_seen: float = 0.0      # timestamp
    device_info: dict = field(default_factory=dict)
    auth_required: bool = True  # Does device require API key?
    auth_status: str = "unknown"  # unknown/ok/failed/not_tested
    require_key_changed: float = 0.0  # timestamp when key change detected

    @property
    def base_url(self) -> str:
        return f"http://{self.ip}:{self.port}"

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/v1"

    @property
    def display_name(self) -> str:
        if self.name:
            return f"{self.name} ({self.ip})"
        return self.ip

    @property
    def status_emoji(self) -> str:
        return {
            "ready": "[ok]",
            "idle": "[idle]",
            "generating": "[gen]",
            "loading": "[load]",
            "reloading": "[load]",
            "error": "[err]",
        }.get(self.status, "[?]")

    @property
    def status_icon_name(self) -> str:
        """SVG icon name for the device status."""
        return {
            "ready": "circle-check",
            "idle": "circle",
            "generating": "loader-circle",
            "loading": "loader-circle",
            "reloading": "loader-circle",
            "error": "circle-x",
        }.get(self.status, "circle")

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def to_dict(self) -> dict:
        return {
            "ip": self.ip,
            "port": self.port,
            "name": self.name,
            "model": self.model,
            "status": self.status,
            "is_vision": self.is_vision,
            "cpu_cores": self.cpu_cores,
            "ram_mb": self.ram_mb,
            "android_version": self.android_version,
            "api_key": self.api_key,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DiscoveredDevice":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _get_local_subnet() -> str:
    """Get the local subnet prefix (e.g., '192.168.1')."""
    ip = detect_lan_ip()
    parts = ip.split(".")
    if len(parts) == 4:
        return ".".join(parts[:3])
    return "192.168.1"


def _probe_host(ip: str, port: int = DISCOVERY_PORT, timeout: float = DISCOVERY_TIMEOUT) -> Optional[DiscoveredDevice]:
    """Probe a single host for a PhonoLab API server."""
    try:
        # Quick TCP connect check
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        if result != 0:
            return None

        # Query /health
        url = f"http://{ip}:{port}/health"
        req = urllib.request.Request(url, headers={"User-Agent": "NativeLab-Discovery/1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())

        if not data.get("ok"):
            return None

        # It's a PhonoLab device
        runtime = data.get("runtime", {})
        device_summary = data.get("device", {})
        auth_info = data.get("auth", {})

        device = DiscoveredDevice(
            ip=ip,
            port=port,
            name=device_summary.get("model", ""),
            model=runtime.get("model", "none"),
            status=data.get("status", "unknown"),
            is_vision=runtime.get("is_vision", False),
            cpu_cores=device_summary.get("cpu_cores", 0),
            ram_mb=device_summary.get("ram_mb", 0),
            last_seen=time.time(),
            auth_required=auth_info.get("required", True),
            auth_status="not_tested",
        )

        # Try to get more details from /device
        try:
            device_url = f"http://{ip}:{port}/device"
            req2 = urllib.request.Request(device_url, headers={"User-Agent": "NativeLab-Discovery/1"})
            with urllib.request.urlopen(req2, timeout=timeout) as resp2:
                device_data = json.loads(resp2.read().decode())
                device.device_info = device_data
                dev = device_data.get("device", {})
                device.name = f"{dev.get('manufacturer', '')} {dev.get('model', '')}".strip()
                device.android_version = dev.get("android_version", "")
                mem = device_data.get("memory", {})
                device.ram_mb = mem.get("system_total_mb", device.ram_mb)
                cpu = device_data.get("cpu", {})
                device.cpu_cores = cpu.get("cores", device.cpu_cores)
        except Exception:
            pass

        return device

    except Exception:
        return None


def scan_network(
    subnet: str = "",
    port: int = DISCOVERY_PORT,
    timeout: float = DISCOVERY_TIMEOUT,
    on_found: Optional[Callable[[DiscoveredDevice], None]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    max_threads: int = 64,
) -> List[DiscoveredDevice]:
    """
    Scan the local subnet for PhonoLab devices.

    Args:
        subnet: Subnet to scan (e.g., "192.168.1"). Auto-detected if empty.
        port: Port to probe (default 8787).
        timeout: Per-host timeout in seconds.
        on_found: Callback when a device is found.
        progress_cb: Callback with (scanned_count, total_count).
        max_threads: Maximum concurrent threads.

    Returns:
        List of discovered devices.
    """
    if not subnet:
        subnet = _get_local_subnet()

    devices: List[DiscoveredDevice] = []
    lock = threading.Lock()
    scanned = [0]
    total = 254

    def _check(ip: str):
        device = _probe_host(ip, port, timeout)
        with lock:
            scanned[0] += 1
            if progress_cb:
                progress_cb(scanned[0], total)
            if device:
                devices.append(device)
                if on_found:
                    on_found(device)

    # Use thread pool
    threads = []
    for i in range(1, 255):
        ip = f"{subnet}.{i}"
        t = threading.Thread(target=_check, args=(ip,), daemon=True)
        threads.append(t)
        t.start()

        # Throttle to max_threads
        if len(threads) >= max_threads:
            for t in threads:
                t.join(timeout=timeout + 0.5)
            threads = []

    # Wait for remaining
    for t in threads:
        t.join(timeout=timeout + 0.5)

    return devices


def save_devices(devices: List[DiscoveredDevice]) -> None:
    """Cache discovered devices to disk."""
    DISCOVERY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    DISCOVERY_CACHE_FILE.write_text(
        json.dumps([d.to_dict() for d in devices], indent=2),
        encoding="utf-8",
    )


def load_cached_devices() -> List[DiscoveredDevice]:
    """Load previously discovered devices from cache."""
    if not DISCOVERY_CACHE_FILE.exists():
        return []
    try:
        data = json.loads(DISCOVERY_CACHE_FILE.read_text(encoding="utf-8"))
        return [DiscoveredDevice.from_dict(d) for d in data]
    except Exception:
        return []


def refresh_device(device: DiscoveredDevice) -> Optional[DiscoveredDevice]:
    """Refresh a single device's status."""
    return _probe_host(device.ip, device.port, timeout=2.0)


def test_connection(device: DiscoveredDevice, api_key: str = "") -> tuple[bool, str]:
    """
    Test connection to a PhonoLab device with optional API key.
    Returns (success, message).
    Auto-detects if key is needed or has changed.
    """
    key = api_key or device.api_key or ""
    try:
        url = f"{device.base_url}/health"
        headers = {"User-Agent": "NativeLab-Discovery/1"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode())
        if data.get("ok"):
            device.auth_status = "ok"
            return True, f"Connected to {device.display_name}"
        return False, "Device responded but not OK"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            device.auth_status = "failed"
            device.require_key_changed = time.time()
            if not key:
                return False, "Authentication required - enter API key from PhonoLab"
            return False, "API key rejected - key may have changed on device"
        return False, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        return False, f"Connection failed: {e}"


def auto_connect_device(device: DiscoveredDevice) -> tuple[bool, str]:
    """
    Smart connect: tries stored key first, then no key.
    Returns (success, message).
    Updates device.auth_status and device.auth_required.
    """
    # Try with stored key first
    if device.api_key:
        ok, msg = test_connection(device, device.api_key)
        if ok:
            return True, msg
        if device.auth_status == "failed":
            return False, "key_changed"

    # Try without key (device might not require auth)
    ok, msg = test_connection(device, "")
    if ok:
        device.auth_required = False
        device.auth_status = "ok"
        return True, msg

    # If 401 without key, device requires auth
    if device.auth_status == "failed":
        device.auth_required = True
        return False, "auth_required"

    return False, msg


def get_device_status(device: DiscoveredDevice, api_key: str = "") -> Optional[dict]:
    """Get full status from a PhonoLab device."""
    key = api_key or device.api_key or ""
    try:
        url = f"{device.base_url}/status"
        headers = {"User-Agent": "NativeLab-Discovery/1"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def get_device_runtime_config(device: DiscoveredDevice, api_key: str = "") -> Optional[dict]:
    """Get live runtime config (temperature, top_k, etc.) from a PhonoLab device."""
    key = api_key or device.api_key or ""
    try:
        url = f"{device.base_url}/runtime"
        headers = {"User-Agent": "NativeLab-Discovery/1"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode())
            return {
                "loaded": data.get("loaded", False),
                "model": data.get("model", "none"),
                "status": data.get("status", "unknown"),
                "ctx": data.get("ctx", 2048),
                "is_vision": data.get("is_vision", False),
            }
    except Exception:
        return None


def get_device_models(device: DiscoveredDevice, api_key: str = "") -> Optional[list]:
    """Get list of models from a PhonoLab device."""
    key = api_key or device.api_key or ""
    try:
        url = f"{device.base_url}/v1/models"
        headers = {"User-Agent": "NativeLab-Discovery/1"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode())
            return data.get("data", [])
    except Exception:
        return None


def load_model_on_device(device: DiscoveredDevice, model_path: str, api_key: str = "") -> tuple[bool, str]:
    """Trigger model load on a PhonoLab device."""
    key = api_key or device.api_key or ""
    try:
        url = f"{device.base_url}/load"
        body = json.dumps({"model_path": model_path}).encode()
        headers = {
            "User-Agent": "NativeLab-Discovery/1",
            "Content-Type": "application/json",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode())
        if data.get("ok"):
            return True, data.get("message", "Model loading started")
        return False, data.get("error", {}).get("message", "Unknown error")
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode())
            return False, err.get("error", {}).get("message", f"HTTP {e.code}")
        except Exception:
            return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)


def update_device_config(device: DiscoveredDevice, config: dict, api_key: str = "") -> tuple[bool, str]:
    """Update model config on a PhonoLab device (temperature, top_k, etc.)."""
    key = api_key or device.api_key or ""
    try:
        url = f"{device.base_url}/config"
        body = json.dumps(config).encode()
        headers = {
            "User-Agent": "NativeLab-Discovery/1",
            "Content-Type": "application/json",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode())
        if data.get("ok"):
            return True, "Config updated"
        return False, data.get("error", {}).get("message", "Unknown error")
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode())
            return False, err.get("error", {}).get("message", f"HTTP {e.code}")
        except Exception:
            return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)


def get_device_queue(device: DiscoveredDevice, api_key: str = "") -> Optional[dict]:
    """Get queue status from a PhonoLab device."""
    key = api_key or device.api_key or ""
    try:
        url = f"{device.base_url}/queue"
        headers = {"User-Agent": "NativeLab-Discovery/1"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None
