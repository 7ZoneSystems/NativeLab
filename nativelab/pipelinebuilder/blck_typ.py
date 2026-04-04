from nativelab.imports.import_global import dataclass
class PipelineBlockType:
    INPUT        = "input"
    OUTPUT       = "output"
    MODEL        = "model"
    INTERMEDIATE = "intermediate"
    REFERENCE    = "reference"    # injects a reference text snippet before context
    KNOWLEDGE    = "knowledge"    # prepends a knowledge-base chunk before context
    PDF_SUMMARY  = "pdf_summary"  # extracts / summarises a PDF and prepends it
    # ── Logic blocks ─────────────────────────────────────────────────────────
    IF_ELSE      = "if_else"      # condition → True port or False port
    SWITCH       = "switch"       # match value → one of N labelled ports
    FILTER       = "filter"       # pass-through or drop based on condition
    TRANSFORM    = "transform"    # deterministic text transform (prefix/suffix/regex)
    MERGE        = "merge"        # combine multiple incoming contexts into one
    SPLIT        = "split"        # broadcast same text to all outgoing connections
    CUSTOM_CODE  = "custom_code"  # user-written Python executed at runtime
    # ── LLM Logic blocks — conditions evaluated by an attached LLM ───────────
    LLM_IF       = "llm_if"       # LLM answers YES/NO → TRUE/FALSE routing
    LLM_SWITCH   = "llm_switch"   # LLM classifies into one of N user-defined labels
    LLM_FILTER   = "llm_filter"   # LLM decides pass/drop in plain English
    LLM_TRANSFORM= "llm_transform"# LLM rewrites/transforms text per instruction
    LLM_SCORE    = "llm_score"    # LLM scores 1–10 → routes to low/mid/high port

# Runtime PyPDF2 guard — PDF blocks show a friendly error if not installed
try:
    import PyPDF2 as _pypdf2_check
    HAS_PDF = True
except ImportError:
    HAS_PDF = False


@dataclass
class PipelineConnection:
    from_block_id: int
    from_port:     str   # "N" | "S" | "E" | "W"
    to_block_id:   int
    to_port:       str
    is_loop:       bool = False
    loop_times:    int  = 1
