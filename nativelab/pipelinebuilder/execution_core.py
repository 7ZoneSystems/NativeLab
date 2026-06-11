from __future__ import annotations

import json
from typing import Any, Iterable, List, Tuple

from nativelab.native.pipeline_core import apply_transform, merge_texts


def transform_text(context: str, metadata: dict) -> str:
    return apply_transform(str(context), metadata or {})


def merge_contexts(contexts: Iterable[str], metadata: dict) -> str:
    return merge_texts([str(item) for item in contexts], metadata or {})


def collect_merge_inputs(queue: List[Tuple[int, str]], bid: int, context: str):
    extras = [context]
    remaining = []
    for qbid, qctx in queue:
        if qbid == bid:
            extras.append(qctx)
        else:
            remaining.append((qbid, qctx))
    return extras, remaining


def output_sender_label(blocks_by_id: dict, connections: list, output_bid: int) -> str:
    for conn in connections:
        if conn.to_block_id == output_bid:
            sender = blocks_by_id.get(conn.from_block_id)
            if sender:
                return str(sender.label)
            break
    return ""


def output_payload(text: str, sender: str = "") -> str:
    return json.dumps({"text": text, "sender": sender})


def reference_context(context: str, metadata: dict, fallback_label: str) -> tuple[str, str]:
    text = (metadata or {}).get("ref_text", "")
    if not text:
        return context, ""
    name = (metadata or {}).get("ref_name", fallback_label)
    injected = (
        f"[REFERENCE: {name}]\n"
        f"{text[:4000]}"
        + ("\u2026\n[truncated]" if len(text) > 4000 else "")
        + f"\n[/REFERENCE]\n\n{context}"
    )
    return injected, str(name)


def knowledge_context(context: str, metadata: dict) -> str:
    text = (metadata or {}).get("knowledge_text", "")
    if not text:
        return context
    return (
        f"Knowledge Base:\n"
        f"{text[:3000]}"
        + ("\u2026\n[truncated]" if len(text) > 3000 else "")
        + f"\n\n---\n\n{context}"
    )
