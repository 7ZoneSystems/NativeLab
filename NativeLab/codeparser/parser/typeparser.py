from imports.import_global import List, dataclass, field

@dataclass
class ParsedItem:
    """A single parsed code element (function, class, import, etc.)."""
    kind:       str          # "import" | "function" | "class" | "constant" |
                             # "type" | "query" | "table" | "route" | "schema"
    name:       str
    signature:  str = ""     # full declaration line / signature
    body:       str = ""     # raw body text
    docstring:  str = ""     # extracted docstring / doc comment
    decorators: List[str] = field(default_factory=list)
    children:   List["ParsedItem"] = field(default_factory=list)  # methods in class
    line_start: int = 0
    line_end:   int = 0
    language:   str = ""

    def keyword_score(self, words: set) -> float:
        target = (self.name + " " + self.signature + " " +
                  self.docstring).lower()
        return sum(1.0 for w in words if w in target)

    def to_context_snippet(self, include_body: bool = True,
                           max_body: int = 600) -> str:
        parts = []
        if self.decorators:
            parts.append("\n".join(self.decorators))
        parts.append(self.signature if self.signature else self.name)
        if self.docstring:
            parts.append(f"    \"\"\"{self.docstring.strip()[:300]}\"\"\"")
        if include_body and self.body:
            body = self.body[:max_body]
            if len(self.body) > max_body:
                body += "\n    # … (truncated)"
            parts.append(body)
        # children (methods)
        for child in self.children[:6]:
            parts.append(f"    {child.signature}")
            if child.docstring:
                parts.append(f'        """{child.docstring.strip()[:120]}"""')
        return "\n".join(parts)
