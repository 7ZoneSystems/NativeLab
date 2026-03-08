from imports.import_global import List, Dict, Tuple, Path, threading, pickle, re, dataclass, json
from GlobalConfig.hardwareUtil import ram_free_mb, simple_hash
from GlobalConfig.config_global import REF_CACHE_DIR, REF_INDEX_DIR, RAM_WATCHDOG_MB, CHUNK_INDEX_SIZE, MAX_RAM_CHUNKS
from .parser.parsefinal import ParsedScript
from .scriptparser import ScriptParser
from Model.templates import SCRIPT_LANGUAGES
@dataclass
class RefChunk:
    idx:     int
    text:    str
    score:   float = 0.0

class SmartReference:
    """
    A single uploaded reference file with intelligent lazy chunk indexing.
    Supports RAM-limited streaming: cold chunks live on disk, hot chunks in RAM.
    """

    def __init__(self, ref_id: str, name: str, ftype: str, raw_text: str):
        self.ref_id   = ref_id
        self.name     = name
        self.ftype    = ftype           # "pdf" | "python" | "text"
        self._raw     = raw_text
        self._chunks: List[RefChunk]    = []
        self._hot:    Dict[int, str]    = {}   # idx → text (RAM cache)
        self._disk_path = REF_CACHE_DIR / f"{ref_id}.pkl"
        self._indexed   = False
        self._lock      = threading.Lock()
        self.build_index()

    def build_index(self):
        """Split raw text into overlapping chunks and save cold index to disk."""
        text   = self._raw
        step   = CHUNK_INDEX_SIZE()
        chunks = []
        i = 0
        while i < len(text):
            chunk = text[i: i + step + 80]
            chunks.append(RefChunk(idx=len(chunks), text=chunk))
            i += step
        self._chunks = chunks

        if ram_free_mb() > RAM_WATCHDOG_MB():
            for c in chunks[:MAX_RAM_CHUNKS()]:
                self._hot[c.idx] = c.text
        else:
            self.spill_to_disk()

        self._indexed = True

    def spill_to_disk(self):
        try:
            with open(self._disk_path, "wb") as f:
                pickle.dump([c.text for c in self._chunks], f)
            self._hot.clear()
        except Exception:
            pass

    def load_chunk_from_disk(self, idx: int) -> str:
        try:
            if self._disk_path.exists():
                with open(self._disk_path, "rb") as f:
                    all_texts = pickle.load(f)
                if idx < len(all_texts):
                    return all_texts[idx]
        except Exception:
            pass
        return ""

    def get_chunk_text(self, idx: int) -> str:
        if idx in self._hot:
            return self._hot[idx]
        text = self.load_chunk_from_disk(idx)
        if ram_free_mb() > RAM_WATCHDOG_MB() + 200:
            self._hot[idx] = text
        return text

    def query(self, query_text: str, top_k: int = 6) -> str:
        """Return the top_k most relevant chunks for the query (keyword scoring)."""
        if not self._chunks:
            return ""
        words = set(re.findall(r'\w+', query_text.lower()))
        if not words:
            return self.get_chunk_text(0)[:1200]

        scored: List[Tuple[float, int]] = []
        for c in self._chunks:
            text  = self.get_chunk_text(c.idx).lower()
            score = sum(1 for w in words if w in text)
            # Boost for exact phrase
            if query_text.lower()[:30] in text:
                score += 5
            scored.append((score, c.idx))

        scored.sort(key=lambda x: -x[0])
        results = []
        for score, idx in scored[:top_k]:
            if score == 0 and results:
                break
            results.append(self.get_chunk_text(idx))

        return "\n\n---\n\n".join(results)[:3000]

    def full_text_preview(self) -> str:
        return self._raw[:400] + ("…" if len(self._raw) > 400 else "")

    def cleanup(self):
        self._hot.clear()
        if self._disk_path.exists():
            try: self._disk_path.unlink()
            except Exception: pass

# ═══════════════════════════════════════════════════════════════════════════
# ScriptSmartReference — RAM/disk-aware with parsed structure
# ═══════════════════════════════════════════════════════════════════════════

class ScriptSmartReference:
    """
    Extends the reference concept for source code files.
    Stores:
      • ParsedScript  — structured AST/regex parse result (always in RAM)
      • Raw text chunks on disk (same model as SmartReference)
    """

    def __init__(self, ref_id: str, name: str, lang_key: str,
                 raw: str, cache_dir: Path):
        self.ref_id   = ref_id
        self.name     = name
        self.ftype    = "script"
        self.lang_key = lang_key
        self._raw     = raw
        self._cache_dir = cache_dir

        # Parse immediately (fast — no LLM involved)
        self.parsed: ParsedScript = ScriptParser.parse(name, raw)

        # Disk cache path
        self._disk_path = cache_dir / f"{ref_id}_script.pkl"
        self.save_to_disk()

    def save_to_disk(self):
        try:
            with open(self._disk_path, "wb") as f:
                pickle.dump(self.parsed, f, protocol=4)
        except Exception:
            pass

    def query(self, query: str, top_k: int = 6) -> str:
        return self.parsed.build_context(query=query, top_k_items=top_k)

    def full_text_preview(self) -> str:
        hdr = self.parsed.summary_header()
        fns = ", ".join(f.name for f in self.parsed.functions[:8])
        cls = ", ".join(c.name for c in self.parsed.classes[:5])
        return (
            f"{hdr}\n"
            + (f"Functions: {fns}\n" if fns else "")
            + (f"Classes:   {cls}\n" if cls else "")
            + f"\n{self._raw[:300]}…"
        )

    @property
    def _hot(self):  # compat with ReferencePanel RAM badge code
        return {}

    @property
    def _chunks(self):  # compat
        return [1]  # one "chunk" for display purposes

    def cleanup(self):
        if self._disk_path.exists():
            try:
                self._disk_path.unlink()
            except Exception:
                pass

class SessionReferenceStore:
    """Manages all SmartReference objects for one chat session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._refs: Dict[str, SmartReference] = {}
        self._index_path = REF_INDEX_DIR / f"{session_id}_refs.json"
        self._load_meta()

    def _load_meta(self):
        if self._index_path.exists():
            try:
                meta = json.loads(self._index_path.read_text())
                for r in meta.get("refs", []):
                    raw_path = REF_CACHE_DIR / f"{r['ref_id']}_raw.txt"
                    if not raw_path.exists():
                        continue
                    raw = raw_path.read_text(encoding="utf-8", errors="replace")
                    if r.get("ftype") == "script":
                        lang_key = r.get("lang_key", "text")
                        self._refs[r["ref_id"]] = ScriptSmartReference(
                            r["ref_id"], r["name"], lang_key, raw, REF_CACHE_DIR)
                    else:
                        self._refs[r["ref_id"]] = SmartReference(
                            r["ref_id"], r["name"], r["ftype"], raw)
            except Exception:
                pass

    def save_meta(self):
        entries = []
        for r in self._refs.values():
            entry = {"ref_id": r.ref_id, "name": r.name, "ftype": r.ftype}
            if isinstance(r, ScriptSmartReference):
                entry["lang_key"] = r.lang_key
            entries.append(entry)
        meta = {"refs": entries}
        self._index_path.write_text(json.dumps(meta), encoding="utf-8")

    def add_reference(self, name: str, ftype: str, raw_text: str) -> SmartReference:
        ref_id = simple_hash(name + raw_text[:64])
        raw_path = REF_CACHE_DIR / f"{ref_id}_raw.txt"
        raw_path.write_text(raw_text, encoding="utf-8", errors="replace")
        ref = SmartReference(ref_id, name, ftype, raw_text)
        self._refs[ref_id] = ref
        self.save_meta()
        return ref

    def remove(self, ref_id: str):
        r = self._refs.pop(ref_id, None)
        if r:
            r.cleanup()
        raw_path = REF_CACHE_DIR / f"{ref_id}_raw.txt"
        if raw_path.exists():
            try: raw_path.unlink()
            except Exception: pass
        self.save_meta()

    def build_context_block(self, query: str) -> str:
        if not self._refs:
            return ""
        parts = []
        for ref in self._refs.values():
            snippet = ref.query(query, top_k=5)
            if snippet.strip():
                ftype_label = {"pdf": "📄 PDF", "python": "🐍 Python", "text": "📝 Text"}.get(ref.ftype, "📎")
                parts.append(
                    f"[REFERENCE: {ftype_label} · {ref.name}]\n{snippet}\n[/REFERENCE]")
        return "\n\n".join(parts)

    @property
    def refs(self) -> Dict[str, "SmartReference"]:
        return self._refs

    def flush_ram(self):
        """Emergency RAM flush — spill all chunks to disk."""
        for ref in self._refs.values():
            ref.spill_to_disk()

    def reactive_reload(self, query: str):
        """Selectively reload relevant chunks before final summarization."""
        if ram_free_mb() < RAM_WATCHDOG_MB():
            return
        for ref in self._refs.values():
            words = set(re.findall(r'\w+', query.lower()))
            for c in ref._chunks[:MAX_RAM_CHUNKS()]:
                if c.idx not in ref._hot:
                    text = ref.load_chunk_from_disk(c.idx)
                    if any(w in text.lower() for w in words):
                        ref._hot[c.idx] = text
                        if ram_free_mb() < RAM_WATCHDOG_MB():
                            return
  
    def add_script_reference(self, store, name: str, raw: str) -> ScriptSmartReference:
        """Add a parsed script reference to the store."""
        ref_id = simple_hash(name + raw[:64])
        # Save raw to disk for resume/reload
        raw_path = REF_CACHE_DIR / f"{ref_id}_raw.txt"
        raw_path.write_text(raw, encoding="utf-8", errors="replace")

        ext = Path(name).suffix.lower()
        _, lang_key = SCRIPT_LANGUAGES.get(ext, ("Text", "text"))
        cache_dir   = REF_CACHE_DIR

        ref = ScriptSmartReference(ref_id, name, lang_key, raw, cache_dir)
        store._refs[ref_id] = ref
        store.save_meta()
        return ref


    def build_context_block_extended(self, query: str) -> str:
        """
        Extended build_context_block that handles ScriptSmartReference.
        """
        store = self
        if not store._refs:
            return ""
        parts = []
        for ref in store._refs.values():
            if isinstance(ref, ScriptSmartReference):
                snippet = ref.query(query, top_k=6)
                label   = f"💻 Script ({ref.lang_key.upper()}) · {ref.name}"
            else:
                snippet = ref.query(query, top_k=5)
                ftype_label = {
                    "pdf": "📄 PDF", "python": "🐍 Python", "text": "📝 Text"
                }.get(getattr(ref, "ftype", ""), "📎")
                label = f"{ftype_label} · {ref.name}"

            if snippet.strip():
                parts.append(
                    f"[REFERENCE: {label}]\n{snippet}\n[/REFERENCE]")
        return "\n\n".join(parts)
