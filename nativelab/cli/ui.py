"""Terminal UI helpers - ANSI colors, prompts, progress bars, icon display."""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

from nativelab.GlobalConfig.timeouts import LONG_TIMEOUT_SECONDS


# ─────────────────────────────────────────────────────────────────────────────
#  Colors
# ─────────────────────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(s: str)    -> str: return _c("1",  s)
def dim(s: str)     -> str: return _c("2",  s)
def red(s: str)     -> str: return _c("31", s)
def green(s: str)   -> str: return _c("32", s)
def yellow(s: str)  -> str: return _c("33", s)
def blue(s: str)    -> str: return _c("34", s)
def magenta(s: str) -> str: return _c("35", s)
def cyan(s: str)    -> str: return _c("36", s)
def gray(s: str)    -> str: return _c("90", s)


# ─────────────────────────────────────────────────────────────────────────────
#  Logging primitives
# ─────────────────────────────────────────────────────────────────────────────

def info(msg: str)    -> None: print(f"{cyan('ℹ')}  {msg}")
def ok(msg: str)      -> None: print(f"{green('OK')}  {msg}")
def warn(msg: str)    -> None: print(f"{yellow('WARN')}  {msg}")
def err(msg: str)     -> None: print(f"{red('ERR')}  {msg}", file=sys.stderr)
def hr() -> None:
    width = shutil.get_terminal_size((80, 20)).columns
    print(gray("─" * width))


def banner(title: str, subtitle: str = "") -> None:
    width = shutil.get_terminal_size((80, 20)).columns
    bar   = "═" * width
    print(magenta(bar))
    show_icon()
    print(bold(f"  {title}"))
    if subtitle:
        print(dim(f"     {subtitle}"))
    print(magenta(bar))


# ─────────────────────────────────────────────────────────────────────────────
#  Icon
# ─────────────────────────────────────────────────────────────────────────────

def _icon_path() -> Optional[Path]:
    p = Path(__file__).resolve().parent.parent / "icon.png"
    return p if p.exists() else None


def _supports_iterm_image() -> bool:
    """iTerm2 inline-image protocol is honoured by iTerm2, WezTerm, mintty,
    VS Code's terminal and Hyper. Detect them via standard env vars."""
    if not sys.stdout.isatty():
        return False
    if os.environ.get("LC_TERMINAL") == "iTerm2":              return True
    if os.environ.get("ITERM_SESSION_ID"):                     return True
    if os.environ.get("WEZTERM_EXECUTABLE"):                   return True
    tp = (os.environ.get("TERM_PROGRAM") or "").lower()
    if tp in ("iterm.app", "wezterm", "mintty", "vscode", "hyper"):
        return True
    return False


def _supports_kitty_image() -> bool:
    if not sys.stdout.isatty():
        return False
    if os.environ.get("KITTY_WINDOW_ID"):
        return True
    if (os.environ.get("TERM") or "").startswith("xterm-kitty"):
        return True
    return False


def show_icon() -> None:
    """Render the NativeLab icon inline if the terminal supports it.

    Tries, in order:
      • Kitty graphics protocol via `kitten icat` (best fidelity)
      • iTerm2 inline-image protocol (works in iTerm2, WezTerm, mintty,
        VS Code, Hyper)
      • Silent no-op otherwise - the surrounding ASCII banner still renders.
    """
    icon = _icon_path()
    if icon is None:
        return

    if _supports_kitty_image() and shutil.which("kitten"):
        try:
            subprocess.run(
                ["kitten", "icat", "--align=left",
                 "--place=8x4@2x0", str(icon)],
                check=False, timeout=LONG_TIMEOUT_SECONDS,
            )
            return
        except Exception:
            pass

    if _supports_iterm_image():
        try:
            data    = icon.read_bytes()
            b64data = base64.b64encode(data).decode("ascii")
            b64name = base64.b64encode(icon.name.encode()).decode("ascii")
            sys.stdout.write(
                "  "  # left padding to match banner indent
                f"\033]1337;File=name={b64name};size={len(data)};"
                f"inline=1;width=4;height=2;preserveAspectRatio=1:"
                f"{b64data}\a\n"
            )
            sys.stdout.flush()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────────────────────────────────────

def ask(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        s = input(f"{cyan('?')} {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(130)
    return s or (default or "")


def ask_yesno(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        s = ask(f"{prompt} ({d})").lower()
        if not s:
            return default
        if s in ("y", "yes"): return True
        if s in ("n", "no"):  return False
        warn("Please answer y or n.")


def ask_int(prompt: str, default: Optional[int] = None,
            lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    while True:
        s = ask(prompt, str(default) if default is not None else None)
        try:
            v = int(s)
        except ValueError:
            warn("Enter a whole number.")
            continue
        if lo is not None and v < lo: warn(f"Must be ≥ {lo}."); continue
        if hi is not None and v > hi: warn(f"Must be ≤ {hi}."); continue
        return v


def choose(prompt: str, options: list[str], default: int = 0) -> int:
    """Numbered single-select prompt. Returns the chosen index."""
    print(cyan("?") + f" {prompt}")
    for i, opt in enumerate(options):
        marker = green(" ●") if i == default else gray(" ○")
        print(f"  {marker} {i + 1}. {opt}")
    while True:
        s = ask("Choose", str(default + 1)).strip()
        try:
            idx = int(s) - 1
        except ValueError:
            warn("Enter a number.")
            continue
        if 0 <= idx < len(options):
            return idx
        warn(f"Pick a number between 1 and {len(options)}.")


# ─────────────────────────────────────────────────────────────────────────────
#  Progress
# ─────────────────────────────────────────────────────────────────────────────

class ProgressBar:
    """Single-line byte/percent progress bar that overwrites itself."""

    def __init__(self, total: int = 0, prefix: str = "", width: int = 32):
        self.total  = max(total, 0)
        self.prefix = prefix
        self.width  = width
        self._last  = 0.0
        self._t0    = time.time()

    def update(self, done: int, total: Optional[int] = None) -> None:
        if total is not None and total > 0:
            self.total = total
        now = time.time()
        # Throttle to 10 Hz to keep terminals snappy
        if now - self._last < 0.1 and done < self.total:
            return
        self._last = now

        if self.total > 0:
            pct  = min(done / self.total, 1.0)
            fill = int(pct * self.width)
            bar  = "█" * fill + "░" * (self.width - fill)
            elapsed = max(now - self._t0, 0.001)
            speed   = done / elapsed
            eta     = (self.total - done) / speed if speed > 0 else 0
            line = (
                f"{self.prefix}  [{bar}] "
                f"{_human(done)}/{_human(self.total)}  "
                f"{pct*100:5.1f}%  "
                f"{_human(speed)}/s  "
                f"eta {_secs(eta)}"
            )
        else:
            line = f"{self.prefix}  {_human(done)} downloaded"

        sys.stdout.write("\r" + line[: shutil.get_terminal_size((120, 20)).columns - 1])
        sys.stdout.flush()

    def done(self, msg: str = "") -> None:
        sys.stdout.write("\r\033[K")
        if msg:
            ok(msg)


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:6.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _secs(s: float) -> str:
    s = max(int(s), 0)
    if s < 60:    return f"{s}s"
    if s < 3600:  return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def stream_iter(it: Iterable[str]) -> str:
    """Print streamed string chunks as they arrive; return the full text."""
    out = []
    for chunk in it:
        sys.stdout.write(chunk)
        sys.stdout.flush()
        out.append(chunk)
    print()
    return "".join(out)
