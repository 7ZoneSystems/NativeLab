from nativelab.imports.import_global import Path, Optional, QRect
from .blck_typ import PipelineBlockType

class PipelineBlock:
    _id_counter = 0

    def __init__(self, btype: str, x: int, y: int,
                 model_path: str = "", role: str = "general", label: str = ""):
        PipelineBlock._id_counter += 1
        self.bid        = PipelineBlock._id_counter
        self.btype      = btype
        self.x          = x
        self.y          = y
        self.w          = 148
        self.h          = 76
        self.model_path = model_path
        self.role       = role
        self.label      = label or self._default_label()
        self.selected   = False
        self.metadata:  dict = {}   # stores ref_text / knowledge_text / pdf_path

    def _default_label(self) -> str:
        if self.btype == PipelineBlockType.INPUT:        return "Input"
        if self.btype == PipelineBlockType.OUTPUT:       return "Output"
        if self.btype == PipelineBlockType.INTERMEDIATE: return "Intermediate"
        if self.btype == PipelineBlockType.REFERENCE:    return "Reference"
        if self.btype == PipelineBlockType.KNOWLEDGE:    return "Knowledge"
        if self.btype == PipelineBlockType.PDF_SUMMARY:  return "PDF"
        if self.model_path:
            return Path(self.model_path).stem[:18]
        return "Model"

    def rect(self):
        return QRect(self.x, self.y, self.w, self.h)

    def port_pos(self, port: str) -> tuple:
        cx = self.x + self.w // 2
        cy = self.y + self.h // 2
        return {
            "N": (cx, self.y),
            "S": (cx, self.y + self.h),
            "W": (self.x, cy),
            "E": (self.x + self.w, cy),
        }.get(port, (cx, cy))

    def port_at(self, px: int, py: int, radius: int = 11) -> Optional[str]:
        for port in ("N", "S", "E", "W"):
            bx, by = self.port_pos(port)
            if abs(px - bx) <= radius and abs(py - by) <= radius:
                return port
        return None

    def contains(self, px: int, py: int) -> bool:
        return self.rect().contains(px, py)
