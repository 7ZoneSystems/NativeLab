from imports.import_global import HAS_PSUTIL, json, psutil, socket, _platform, subprocess, Path, dataclass, field, datetime, List
from GlobalConfig.config_global import SERVER_CONFIG_FILE, SESSIONS_DIR
from Model.model_global import detect_model_family, FAMILY_TEMPLATES, ModelFamily

@dataclass
class ServerConfig:
    cli_path:    str = ""
    server_path: str = ""
    host:        str = "127.0.0.1"
    port_range_lo: int = 8600
    port_range_hi: int = 8700
    extra_cli_args:    str = ""
    extra_server_args: str = ""
    # GPU settings
    enable_gpu:   bool = False
    ngl:          int  = -1      # -1 = offload all layers
    main_gpu:     int  = 0       # primary GPU device index
    tensor_split: str  = ""      # e.g. "0.6,0.4" for two GPUs

    def save(self):
        SERVER_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        SERVER_CONFIG_FILE.write_text(json.dumps({
            "cli_path":          self.cli_path,
            "server_path":       self.server_path,
            "host":              self.host,
            "port_range_lo":     self.port_range_lo,
            "port_range_hi":     self.port_range_hi,
            "extra_cli_args":    self.extra_cli_args,
            "extra_server_args": self.extra_server_args,
            "enable_gpu":        self.enable_gpu,
            "ngl":               self.ngl,
            "main_gpu":          self.main_gpu,
            "tensor_split":      self.tensor_split,
        }, indent=2))

    @classmethod
    def load(cls) -> "ServerConfig":
        if SERVER_CONFIG_FILE.exists():
            try:
                d = json.loads(SERVER_CONFIG_FILE.read_text())
                return cls(**{k: v for k, v in d.items()
                               if k in cls.__dataclass_fields__})
            except Exception:
                pass
        return cls()

    @property
    def detected_os(self) -> str:
        s = _platform.system()
        return {"Windows": "Windows", "Darwin": "macOS"}.get(s, "Linux")

    @property
    def default_cli_name(self) -> str:
        return "llama-cli.exe" if self.detected_os == "Windows" else "llama-cli"

    @property
    def default_server_name(self) -> str:
        return "llama-server.exe" if self.detected_os == "Windows" else "llama-server"


SERVER_CONFIG = ServerConfig.load()


def detect_gpus() -> list:
    """
    Detect available GPUs. Returns list of dicts:
      { idx, name, vram_mb, type }   type = 'cuda' | 'metal' | 'vulkan'
    """
    gpus = []
    # ── NVIDIA / AMD via nvidia-smi ───────────────────────────────────────────
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL).decode().strip()
        for i, line in enumerate(out.splitlines()):
            parts = [p.strip() for p in line.split(",")]
            vram  = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
            gpus.append({"idx": i, "name": parts[0] if parts else f"GPU {i}",
                         "vram_mb": vram, "type": "cuda"})
    except Exception:
        pass
    # ── Apple Metal (macOS) ───────────────────────────────────────────────────
    if not gpus and _platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                timeout=6, stderr=subprocess.DEVNULL).decode()
            for line in out.splitlines():
                stripped = line.strip()
                if stripped.startswith("Chipset Model:") or stripped.startswith("Model:"):
                    name = stripped.split(":", 1)[-1].strip()
                    if name:
                        gpus.append({"idx": 0, "name": name,
                                     "vram_mb": 0, "type": "metal"})
                        break
        except Exception:
            pass
        if not gpus:
            gpus.append({"idx": 0, "name": "Apple GPU (Metal)",
                         "vram_mb": 0, "type": "metal"})
    # ── Vulkan fallback probe ─────────────────────────────────────────────────
    if not gpus:
        try:
            out = subprocess.check_output(
                ["vulkaninfo", "--summary"], timeout=5,
                stderr=subprocess.DEVNULL).decode()
            for line in out.splitlines():
                if "deviceName" in line:
                    name = line.split("=", 1)[-1].strip()
                    gpus.append({"idx": len(gpus), "name": name,
                                 "vram_mb": 0, "type": "vulkan"})
        except Exception:
            pass
    return gpus


# MCP config file path
MCP_CONFIG_FILE = Path("./localllm/mcp_config.json")


# ═════════════════════════════ DATA MODELS ══════════════════════════════════

@dataclass
class Message:
    role:      str
    content:   str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M"))


@dataclass
class Session:
    id:       str
    title:    str
    created:  str
    messages: List[Message] = field(default_factory=list)

    @classmethod
    def new(cls, title: str = "New Chat") -> "Session":
        now = datetime.now()
        return cls(id=now.strftime("%Y-%m-%d_%H%M%S"),
                   title=title,
                   created=now.strftime("%Y-%m-%d"))

    @classmethod
    def load(cls, path: Path) -> "Session":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            id=d["id"], title=d["title"], created=d["created"],
            messages=[Message(**m) for m in d.get("messages", [])]
        )

    def save(self):
        path = SESSIONS_DIR / f"{self.id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "id": self.id, "title": self.title, "created": self.created,
                "messages": [{"role": m.role, "content": m.content,
                               "timestamp": m.timestamp} for m in self.messages]
            }, f, indent=2, ensure_ascii=False)

    def add_message(self, role: str, content: str) -> Message:
        m = Message(role=role, content=content)
        self.messages.append(m)
        return m

    def build_prompt(self, model_path: str = "", max_chars: int = 8000) -> str:
        """Build prompt using the correct template for the detected model family."""
        fam = detect_model_family(model_path) if model_path else FAMILY_TEMPLATES["mistral"]
        return self._build_with_template(fam, max_chars)

    def _build_with_template(self, fam: ModelFamily, max_chars: int) -> str:
        recent: List[Message] = []
        used = 0
        for m in reversed(self.messages):
            used += len(m.content)
            if used > max_chars:
                break
            recent.insert(0, m)

        parts: List[str] = []
        i = 0
        first_turn = True
        while i < len(recent):
            m = recent[i]
            if m.role == "user":
                reply = (recent[i + 1].content
                         if i + 1 < len(recent) and recent[i + 1].role == "assistant"
                         else "")
                # BOS only on the very first turn to avoid token duplication
                bos = fam.bos if first_turn else ""
                first_turn = False
                user_part = fam.user_prefix + m.content + fam.user_suffix
                if reply:
                    parts.append(bos + user_part +
                                 fam.assistant_prefix + reply + fam.assistant_suffix)
                    i += 2
                else:
                    parts.append(bos + user_part + fam.assistant_prefix)
                    i += 1
            else:
                i += 1

        return "".join(parts)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}\n\n*{self.created}*\n\n---\n\n"]
        for m in self.messages:
            icon = "**You**" if m.role == "user" else "**Assistant**"
            lines.append(f"{icon} · {m.timestamp}\n\n{m.content}\n\n---\n\n")
        return "".join(lines)

    def to_txt(self) -> str:
        lines = [f"{self.title}\n{'='*40}\n{self.created}\n\n"]
        for m in self.messages:
            lines.append(f"[{m.role.upper()}] {m.timestamp}\n{m.content}\n\n")
        return "".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "id": self.id, "title": self.title, "created": self.created,
            "messages": [{"role": m.role, "content": m.content,
                           "timestamp": m.timestamp} for m in self.messages]
        }, indent=2, ensure_ascii=False)

    @property
    def approx_tokens(self) -> int:
        return sum(len(m.content) for m in self.messages) // 4


# ═════════════════════════════ ENGINE LAYER ═════════════════════════════════

def free_port(lo: int = 0, hi: int = 0) -> int:
    if lo == 0: lo = SERVER_CONFIG.port_range_lo
    if hi == 0: hi = SERVER_CONFIG.port_range_hi
    for p in range(lo, hi):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return lo


def kill_stray_llama_servers(keep_pids: set | None = None):
    """
    Kill every llama-server process on this machine whose PID is NOT in keep_pids.
    Uses psutil process tree kill when available; falls back to pkill/taskkill.
    Returns count of killed processes (-1 if count unknown via pkill).
    """
    keep_pids = keep_pids or set()
    import platform
    killed = 0
    if HAS_PSUTIL:
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                name = proc.info.get("name") or ""
                cmd  = " ".join(proc.info.get("cmdline") or [])
                if ("llama-server" in name or "llama-server" in cmd) \
                        and proc.pid not in keep_pids:
                    try:
                        # Kill whole subtree first
                        try:
                            for child in proc.children(recursive=True):
                                child.kill()
                        except Exception:
                            pass
                        proc.kill()
                        killed += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    else:
        try:
            if __import__("platform").system() == "Windows":
                subprocess.call(
                    ["taskkill", "/F", "/IM", "llama-server.exe"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.call(
                    ["pkill", "-9", "-f", "llama-server"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            killed = -1   # count unknown
        except Exception:
            pass
    return killed