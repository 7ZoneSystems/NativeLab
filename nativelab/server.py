#!/usr/bin/env python3
"""
NativeLab Local Server - Standalone local LLM server.

Turn any GGUF model into an OpenAI-compatible API server.
No account, no cloud, runs entirely on your machine.

Usage:
    python server.py                          # Interactive setup
    python server.py --model path/to/model.gguf  # Direct start
    python server.py --model model.gguf --port 8787 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

# ── Add parent to path for nativelab imports ──────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))


def find_gguf_models(search_dirs: list[Path]) -> list[Path]:
    """Find all GGUF files in given directories."""
    models = []
    for d in search_dirs:
        if d.exists():
            models.extend(d.glob("**/*.gguf"))
    return sorted(set(models), key=lambda p: p.name.lower())


def interactive_model_picker(models: list[Path]) -> Path | None:
    """Let user pick a model from a list."""
    if not models:
        print("No GGUF models found.")
        print("Download a model from https://huggingface.co and place it in ./models/")
        return None

    print(f"\nFound {len(models)} model(s):\n")
    for i, m in enumerate(models, 1):
        size_mb = m.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {m.name}  ({size_mb:.0f} MB)")

    print()
    while True:
        try:
            choice = input("Select model number (or 'q' to quit): ").strip()
            if choice.lower() == "q":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            print(f"Enter a number between 1 and {len(models)}")
        except ValueError:
            print("Enter a number or 'q'")
        except (EOFError, KeyboardInterrupt):
            return None


def detect_hardware() -> dict:
    """Detect basic hardware info."""
    import multiprocessing
    cores = multiprocessing.cpu_count()

    ram_mb = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram_mb = int(line.split()[1]) // 1024
                    break
    except Exception:
        pass

    return {"cores": cores, "ram_mb": ram_mb}


def recommend_settings(ram_mb: int, model_size_mb: int) -> dict:
    """Recommend server settings based on hardware."""
    if ram_mb < 4096:
        return {"ctx": 1024, "threads": min(2, os.cpu_count() or 2), "n_predict": 256}
    elif ram_mb < 8192:
        return {"ctx": 2048, "threads": min(4, os.cpu_count() or 4), "n_predict": 512}
    else:
        return {"ctx": 4096, "threads": min(8, os.cpu_count() or 8), "n_predict": 1024}


def start_server(
    model_path: Path,
    host: str = "127.0.0.1",
    port: int = 8787,
    threads: int = 4,
    ctx_size: int = 2048,
    n_predict: int = 512,
) -> None:
    """Start the local LLM server."""
    try:
        from nativelab.api_server.config import ApiServerConfig, generate_api_key
        from nativelab.api_server.server import NativeLabApiServer
        from nativelab.core.backend import get_backend
        from nativelab.Model.ModelRegistry import get_model_registry
    except ImportError as e:
        print(f"Error: NativeLab not installed. Run: pip install nativelab")
        print(f"  ({e})")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  NativeLab Local Server")
    print(f"{'='*60}")
    print(f"  Model:    {model_path.name}")
    print(f"  Size:     {model_path.stat().st_size / (1024*1024):.0f} MB")
    print(f"  Host:     {host}")
    print(f"  Port:     {port}")
    print(f"  Threads:  {threads}")
    print(f"  Context:  {ctx_size}")
    print(f"  Predict:  {n_predict}")
    print(f"{'='*60}\n")

    # Register model
    registry = get_model_registry()
    registry.add(str(model_path))

    # Create server config
    config = ApiServerConfig(
        host=host,
        port=port,
        protocol="both",
        require_api_key=True,
        local_api_key=generate_api_key(),
        lan_api_key=generate_api_key(),
    )
    config.save()

    print(f"Local API Key: ...{config.local_api_key[-4:]}")
    print(f"LAN API Key:   ...{config.lan_api_key[-4:]}")
    print(f"\nEndpoints:")
    print(f"  http://{host}:{port}/v1/chat/completions  (OpenAI)")
    print(f"  http://{host}:{port}/v1/messages           (Anthropic)")
    print(f"  http://{host}:{port}/health                (Health check)")
    print(f"  http://{host}:{port}/v1/models             (List models)")
    print(f"\nPress Ctrl+C to stop.\n")

    # Load model
    backend = get_backend()
    result = backend.load_model(
        str(model_path),
        threads=threads,
        ctx=ctx_size,
        n_predict=n_predict,
    )
    if not result.ok:
        print(f"Error loading model: {result.error}")
        sys.exit(1)

    print(f"Model loaded. Server starting...\n")

    # Start server
    from nativelab.labs.endpoints import LabEndpoints
    endpoints = LabEndpoints(
        llama_engine=backend._llama_engine,
        api_engine=backend._api_engine,
    )

    server = NativeLabApiServer(
        config=config,
        endpoints=endpoints,
    )
    result_msg = server.start()
    print(f"{result_msg}\n")

    # Wait for Ctrl+C
    def _shutdown(sig, frame):
        print("\nShutting down...")
        server.stop()
        backend.unload_model()
        print("Server stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown(None, None)


def main():
    parser = argparse.ArgumentParser(
        description="NativeLab Local Server - turn any GGUF model into an API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py                                    # Interactive
  python server.py --model models/llama-7b.gguf       # Direct
  python server.py --model model.gguf --port 9000     # Custom port
  python server.py --model model.gguf --host 0.0.0.0  # LAN access
        """,
    )
    parser.add_argument("--model", "-m", type=str, help="Path to GGUF model file")
    parser.add_argument("--port", "-p", type=int, default=8787, help="Server port (default: 8787)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1, use 0.0.0.0 for LAN)")
    parser.add_argument("--threads", "-t", type=int, default=0, help="CPU threads (0 = auto)")
    parser.add_argument("--ctx", type=int, default=0, help="Context size (0 = auto)")
    parser.add_argument("--n-predict", type=int, default=0, help="Max tokens (0 = auto)")

    args = parser.parse_args()

    # Find model
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Error: Model not found: {model_path}")
            sys.exit(1)
    else:
        search_dirs = [
            Path("./models"),
            Path("./PhonoLab/models"),
            Path.home() / "models",
        ]
        models = find_gguf_models(search_dirs)
        model_path = interactive_model_picker(models)
        if model_path is None:
            sys.exit(0)

    # Detect hardware and recommend settings
    hw = detect_hardware()
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    rec = recommend_settings(hw["ram_mb"], model_size_mb)

    threads = args.threads or rec["threads"]
    ctx_size = args.ctx or rec["ctx"]
    n_predict = args.n_predict or rec["n_predict"]

    print(f"Hardware: {hw['cores']} cores, {hw['ram_mb']} MB RAM")
    print(f"Model: {model_path.name} ({model_size_mb:.0f} MB)")
    print(f"Recommended: threads={threads}, ctx={ctx_size}, n_predict={n_predict}")

    start_server(
        model_path=model_path,
        host=args.host,
        port=args.port,
        threads=threads,
        ctx_size=ctx_size,
        n_predict=n_predict,
    )


if __name__ == "__main__":
    main()
