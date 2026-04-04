from nativelab.imports.import_global import json, List, Dict
from nativelab.GlobalConfig.config_global import PIPELINES_DIR
from .blck_typ import PipelineConnection
from .pipblck import PipelineBlock
def _pipeline_to_dict(blocks: list, connections: list) -> dict:
    """Serialise a pipeline to a JSON-safe dict."""
    return {
        "version": 2,
        "blocks": [
            {
                "bid":        b.bid,
                "btype":      b.btype,
                "x":          b.x,
                "y":          b.y,
                "w":          b.w,
                "h":          b.h,
                "model_path": b.model_path,
                "role":       b.role,
                "label":      b.label,
                "metadata":   b.metadata,
            }
            for b in blocks
        ],
        "connections": [
            {
                "from_block_id": c.from_block_id,
                "from_port":     c.from_port,
                "to_block_id":   c.to_block_id,
                "to_port":       c.to_port,
                "is_loop":       c.is_loop,
                "loop_times":    c.loop_times,
            }
            for c in connections
        ],
    }

def _pipeline_from_dict(data: dict):
    """Deserialise blocks + connections from a saved dict. Returns (blocks, connections)."""
    blocks = []
    id_map: Dict[int, int] = {}   # old bid → new bid (counter is global)
    for bd in data.get("blocks", []):
        b            = PipelineBlock(bd["btype"], bd["x"], bd["y"],
                                     bd.get("model_path", ""),
                                     bd.get("role", "general"),
                                     bd.get("label", ""))
        b.metadata   = bd.get("metadata", {})
        id_map[bd["bid"]] = b.bid
        blocks.append(b)

    connections = []
    for cd in data.get("connections", []):
        connections.append(PipelineConnection(
            from_block_id=id_map.get(cd["from_block_id"], cd["from_block_id"]),
            from_port=cd["from_port"],
            to_block_id=id_map.get(cd["to_block_id"],   cd["to_block_id"]),
            to_port=cd["to_port"],
            is_loop=cd.get("is_loop", False),
            loop_times=cd.get("loop_times", 1),
        ))
    return blocks, connections

def list_saved_pipelines() -> List[str]:
    """Return sorted list of saved pipeline names (without .json)."""
    return sorted(p.stem for p in PIPELINES_DIR.glob("*.json"))

def save_pipeline(name: str, blocks: list, connections: list):
    path = PIPELINES_DIR / f"{name}.json"
    path.write_text(
        json.dumps(_pipeline_to_dict(blocks, connections), indent=2),
        encoding="utf-8")

def load_pipeline(name: str):
    path = PIPELINES_DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return _pipeline_from_dict(data)
