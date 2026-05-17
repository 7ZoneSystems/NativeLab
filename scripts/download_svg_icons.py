#!/usr/bin/env python3
"""
Download a complete monochrome SVG icon library into nativelab/assets/.

Default source:
  lucide-static from npm

Examples:
  python scripts/download_svg_icons.py
  python scripts/download_svg_icons.py --version 0.468.0
  python scripts/download_svg_icons.py --output nativelab/assets/icons --stroke "#f5f5f5"
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

from nativelab.GlobalConfig.timeouts import LONG_TIMEOUT_SECONDS


PACKAGE_NAME = "lucide-static"
REGISTRY_URL = f"https://registry.npmjs.org/{PACKAGE_NAME}"
DEFAULT_OUTPUT = Path("nativelab/assets/icons")
SVG_NAMESPACE = "http://www.w3.org/2000/svg"


def _http_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "NativeLab-icon-downloader"})
    with urllib.request.urlopen(request, timeout=LONG_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "NativeLab-icon-downloader"})
    with urllib.request.urlopen(request, timeout=LONG_TIMEOUT_SECONDS) as response:
        return response.read()


def _resolve_package(version: str | None) -> tuple[str, str]:
    metadata = _http_json(REGISTRY_URL)
    resolved = version or metadata.get("dist-tags", {}).get("latest")
    if not resolved:
        raise RuntimeError("Could not resolve latest lucide-static version from npm metadata.")

    versions = metadata.get("versions", {})
    package = versions.get(resolved)
    if not package:
        known = ", ".join(sorted(versions.keys())[-5:])
        raise RuntimeError(f"lucide-static version {resolved!r} was not found. Recent versions: {known}")

    tarball = package.get("dist", {}).get("tarball")
    if not tarball:
        raise RuntimeError(f"lucide-static {resolved} metadata did not include a tarball URL.")
    return resolved, tarball


def _icon_name_from_member(member_name: str) -> str | None:
    parts = Path(member_name).parts
    if len(parts) < 3 or parts[-2] != "icons" or not parts[-1].endswith(".svg"):
        return None
    return parts[-1]


def _normalize_svg(svg_text: str, stroke: str, size: int) -> str:
    ET.register_namespace("", SVG_NAMESPACE)
    root = ET.fromstring(svg_text)
    root.set("width", str(size))
    root.set("height", str(size))
    root.set("fill", "none")
    root.set("stroke", stroke)
    root.set("stroke-width", root.get("stroke-width", "2"))
    root.set("stroke-linecap", root.get("stroke-linecap", "round"))
    root.set("stroke-linejoin", root.get("stroke-linejoin", "round"))

    for element in root.iter():
        if element.tag.endswith("}svg"):
            continue
        if "stroke" in element.attrib:
            element.set("stroke", stroke)
        if "fill" in element.attrib and element.attrib["fill"] not in {"none", "transparent"}:
            element.set("fill", "none")

    return ET.tostring(root, encoding="unicode")


def _extract_icons(tarball_bytes: bytes, output: Path, stroke: str, size: int) -> int:
    count = 0
    with tempfile.TemporaryDirectory(prefix="nativelab-icons-") as tmp:
        tmp_output = Path(tmp) / "icons"
        tmp_output.mkdir(parents=True, exist_ok=True)

        with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as archive:
            for member in archive.getmembers():
                icon_name = _icon_name_from_member(member.name)
                if not icon_name or not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                raw_svg = extracted.read().decode("utf-8")
                normalized = _normalize_svg(raw_svg, stroke=stroke, size=size)
                (tmp_output / icon_name).write_text(normalized + "\n", encoding="utf-8")
                count += 1

        if count == 0:
            raise RuntimeError("No SVG icons were found in the downloaded lucide-static tarball.")

        if output.exists():
            shutil.rmtree(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_output), str(output))
    return count


def _write_manifest(output: Path, version: str, tarball_url: str, count: int, stroke: str, size: int) -> None:
    manifest = {
        "library": PACKAGE_NAME,
        "version": version,
        "source": tarball_url,
        "icon_count": count,
        "stroke": stroke,
        "size": size,
        "downloaded_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _validate_color(value: str) -> str:
    if value == "currentColor":
        return value
    if re.fullmatch(r"#[0-9a-fA-F]{3}([0-9a-fA-F]{3})?", value):
        return value
    raise argparse.ArgumentTypeError("stroke must be currentColor or a hex color like #111111")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Lucide SVG icons into nativelab/assets/icons.")
    parser.add_argument("--version", help="lucide-static npm version. Defaults to npm latest.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Icon output directory.")
    parser.add_argument(
        "--stroke",
        type=_validate_color,
        default="currentColor",
        help="SVG stroke color to write. Use currentColor for theme-controlled monochrome icons.",
    )
    parser.add_argument("--size", type=int, default=24, help="SVG width/height in pixels.")
    args = parser.parse_args()

    if args.size < 8 or args.size > 128:
        parser.error("--size must be between 8 and 128")

    try:
        version, tarball_url = _resolve_package(args.version)
        print(f"Downloading {PACKAGE_NAME} {version}...")
        tarball = _http_bytes(tarball_url)
        count = _extract_icons(tarball, args.output, stroke=args.stroke, size=args.size)
        _write_manifest(args.output, version, tarball_url, count, args.stroke, args.size)
    except urllib.error.URLError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Icon download failed: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {count} SVG icons to {args.output}")
    print(f"Manifest: {args.output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
