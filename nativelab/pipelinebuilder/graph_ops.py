from __future__ import annotations

from typing import Any, Dict, Iterable, List, MutableMapping, Optional

from nativelab.native.pipeline_core import normalize_ids, route_edges, would_form_loop as native_would_form_loop


def normalize_block_ids(blocks: list, connections: list, current_counter: int) -> int:
    """Normalize loaded/inserted block ids and remap connection endpoints."""
    block_ids = [getattr(block, "bid", 0) for block in blocks]
    endpoints = [
        (getattr(conn, "from_block_id", 0), getattr(conn, "to_block_id", 0))
        for conn in connections
    ]
    result = normalize_ids(block_ids, endpoints, int(current_counter or 0))
    ids = list(result.get("ids", []))
    remapped = list(result.get("connections", []))

    for block, bid in zip(blocks, ids):
        block.bid = int(bid)
    for conn, pair in zip(connections, remapped):
        conn.from_block_id = int(pair[0])
        conn.to_block_id = int(pair[1])

    counter = int(result.get("counter", current_counter or 0))
    if ids:
        counter = max(counter, max(int(bid) for bid in ids))
    return counter


def build_adjacency(connections: Iterable[Any]) -> Dict[int, List[Any]]:
    adj: Dict[int, List[Any]] = {}
    for conn in connections:
        adj.setdefault(int(conn.from_block_id), []).append(conn)
    return adj


def would_form_loop(connections: Iterable[Any], from_bid: int, to_bid: int) -> bool:
    pairs = [
        (int(getattr(conn, "from_block_id", 0)), int(getattr(conn, "to_block_id", 0)))
        for conn in connections
    ]
    return native_would_form_loop(pairs, int(from_bid), int(to_bid))


def route_connections(
    connections: List[Any],
    from_bid: int,
    visit_counts: MutableMapping[str, int],
    mode: str = "all",
    branch_key: str = "",
    port_labels: Optional[MutableMapping[str, Any]] = None,
) -> List[Any]:
    records = [
        (
            idx,
            int(getattr(conn, "from_block_id", 0)),
            str(getattr(conn, "from_port", "")),
            int(getattr(conn, "to_block_id", 0)),
            bool(getattr(conn, "is_loop", False)),
            int(getattr(conn, "loop_times", 1)),
        )
        for idx, conn in enumerate(connections)
    ]
    selected = route_edges(records, int(from_bid), visit_counts, mode, branch_key, port_labels)
    out: List[Any] = []
    for idx, _target, _port in selected:
        if 0 <= idx < len(connections):
            out.append(connections[idx])
    return out
