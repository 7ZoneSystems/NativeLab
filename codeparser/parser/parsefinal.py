from imports.import_global import List, Tuple, re, dataclass, field
from .typeparser import ParsedItem
@dataclass
class ParsedScript:
    """
    Structured representation of a source-code file,
    ready to be injected as rich LLM context.
    """
    language:   str
    lang_key:   str          # e.g. "python", "js", "sql"
    filename:   str
    raw:        str          # full raw text (kept for fallback)
    imports:    List[ParsedItem] = field(default_factory=list)
    functions:  List[ParsedItem] = field(default_factory=list)
    classes:    List[ParsedItem] = field(default_factory=list)
    constants:  List[ParsedItem] = field(default_factory=list)
    types:      List[ParsedItem] = field(default_factory=list)
    misc:       List[ParsedItem] = field(default_factory=list)
    errors:     List[str]        = field(default_factory=list)

    # ── meta ──────────────────────────────────────────────────────────────────
    @property
    def total_items(self) -> int:
        return (len(self.imports) + len(self.functions) +
                len(self.classes) + len(self.constants) +
                len(self.types) + len(self.misc))

    @property
    def all_items(self) -> List[ParsedItem]:
        return (self.imports + self.functions + self.classes +
                self.constants + self.types + self.misc)

    def summary_header(self) -> str:
        """One-line structural overview."""
        parts = []
        if self.imports:    parts.append(f"{len(self.imports)} imports")
        if self.classes:    parts.append(f"{len(self.classes)} classes")
        if self.functions:  parts.append(f"{len(self.functions)} functions/methods")
        if self.constants:  parts.append(f"{len(self.constants)} constants")
        if self.types:      parts.append(f"{len(self.types)} type defs")
        return (f"[{self.language} · {self.filename}]  "
                + (", ".join(parts) if parts else "no parsed items"))

    # ── context block builder ─────────────────────────────────────────────────
    def build_context(self, query: str = "", top_k_items: int = 8,
                      max_chars: int = 3500) -> str:
        """
        Build a structured context block for LLM injection.
        When a query is provided, ranks items by keyword relevance.
        """
        header = (
            f"╔══ SCRIPT REFERENCE ══════════════════════════════════════\n"
            f"║  File    : {self.filename}\n"
            f"║  Language: {self.language}\n"
            f"║  Structure: {self.summary_header()}\n"
            f"╚══════════════════════════════════════════════════════════\n\n"
        )

        sections: List[str] = []
        total_chars = len(header)

        # Imports always included (compact)
        if self.imports:
            imp_block = "── Imports ──\n"
            imp_block += "\n".join(i.signature for i in self.imports[:30])
            sections.append(imp_block)
            total_chars += len(imp_block)

        # Constants / type aliases (compact)
        if self.constants or self.types:
            ct_block = "\n── Constants & Types ──\n"
            for item in (self.constants + self.types)[:20]:
                ct_block += item.signature + "\n"
            sections.append(ct_block)
            total_chars += len(ct_block)

        # Rank remaining items
        words = set(re.findall(r'\w+', query.lower())) if query else set()
        candidates: List[Tuple[float, ParsedItem]] = []
        for item in self.functions + self.classes + self.misc:
            score = item.keyword_score(words) if words else 0.0
            candidates.append((score, item))
        candidates.sort(key=lambda x: -x[0])

        for score, item in candidates[:top_k_items]:
            if total_chars >= max_chars:
                break
            budget = min(600, max_chars - total_chars)
            snippet = item.to_context_snippet(
                include_body=(score > 0 or len(candidates) <= 4),
                max_body=budget)
            section = f"\n── {item.kind.capitalize()}: {item.name} ──\n{snippet}\n"
            sections.append(section)
            total_chars += len(section)

        return header + "\n".join(sections)

    def query(self, query: str, top_k: int = 6) -> str:
        """Compatibility with SmartReference.query() interface."""
        return self.build_context(query=query, top_k_items=top_k)
