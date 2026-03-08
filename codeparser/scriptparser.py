class ScriptParser:
    """
    Multi-language script parser.
    Uses Python's AST for .py files and regex-based parsers for others.
    """

    @classmethod
    def parse(cls, filename: str, raw: str) -> "ParsedScript":
        ext = Path(filename).suffix.lower()
        lang, key = SCRIPT_LANGUAGES.get(ext, ("Text", "text"))
        ps = ParsedScript(language=lang, lang_key=key,
                          filename=filename, raw=raw)
        try:
            parser = getattr(cls, f"_parse_{key}", cls._parse_generic)
            parser(ps, raw)
        except Exception as e:
            ps.errors.append(f"Parser error: {e}")
            cls._parse_generic(ps, raw)
        return ps

    # ── Python parser (AST) ───────────────────────────────────────────────────
    @classmethod
    def _parse_python(cls, ps: ParsedScript, raw: str):
        try:
            tree = ast.parse(raw)
        except SyntaxError as e:
            ps.errors.append(f"SyntaxError: {e}")
            cls._parse_generic(ps, raw)
            return

        lines = raw.splitlines()

        def _src(node) -> str:
            try:
                return ast.get_source_segment(raw, node) or ""
            except Exception:
                return ""

        def _docstr(node) -> str:
            try:
                v = ast.get_docstring(node)
                return v or ""
            except Exception:
                return ""

        def _sig(node, lines) -> str:
            """Reconstruct signature line(s)."""
            try:
                l = node.lineno - 1
                sig_lines = []
                while l < len(lines):
                    sig_lines.append(lines[l])
                    if ":" in lines[l]:
                        break
                    l += 1
                return "\n".join(sig_lines)
            except Exception:
                return getattr(node, "name", "")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    n = alias.asname or alias.name
                    ps.imports.append(ParsedItem(
                        kind="import", name=n,
                        signature=f"import {alias.name}"
                        + (f" as {alias.asname}" if alias.asname else ""),
                        language="python"))

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names  = ", ".join(
                    (a.asname or a.name) for a in node.names)
                full   = f"from {module} import {names}"
                ps.imports.append(ParsedItem(
                    kind="import", name=module,
                    signature=full, language="python"))

            elif isinstance(node, ast.FunctionDef) and \
                 not isinstance(node, ast.AsyncFunctionDef):
                if not any(
                    isinstance(p, ast.FunctionDef)
                    for p in ast.walk(tree)
                    if isinstance(getattr(p, "body", None), list)
                    and any(child is node for child in p.body)
                ):
                    decs = [ast.unparse(d) for d in node.decorator_list]
                    ps.functions.append(ParsedItem(
                        kind="function", name=node.name,
                        signature=_sig(node, lines),
                        body=_src(node),
                        docstring=_docstr(node),
                        decorators=[f"@{d}" for d in decs],
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        language="python"))

            elif isinstance(node, ast.AsyncFunctionDef):
                decs = [ast.unparse(d) for d in node.decorator_list]
                ps.functions.append(ParsedItem(
                    kind="function", name=node.name,
                    signature="async " + _sig(node, lines),
                    body=_src(node), docstring=_docstr(node),
                    decorators=[f"@{d}" for d in decs],
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    language="python"))

            elif isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b) for b in node.bases]
                methods = []
                for child in ast.walk(node):
                    if isinstance(child, (ast.FunctionDef,
                                          ast.AsyncFunctionDef)) and \
                       child is not node:
                        decs = [ast.unparse(d) for d in child.decorator_list]
                        methods.append(ParsedItem(
                            kind="method", name=child.name,
                            signature=_sig(child, lines),
                            body=_src(child),
                            docstring=_docstr(child),
                            decorators=[f"@{d}" for d in decs],
                            language="python"))

                base_str = f"({', '.join(bases)})" if bases else ""
                ps.classes.append(ParsedItem(
                    kind="class", name=node.name,
                    signature=f"class {node.name}{base_str}:",
                    docstring=_docstr(node),
                    children=methods,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    language="python"))

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if name.isupper() or name.startswith("_"):
                            try:
                                val = ast.unparse(node.value)[:80]
                            except Exception:
                                val = "..."
                            ps.constants.append(ParsedItem(
                                kind="constant", name=name,
                                signature=f"{name} = {val}",
                                language="python"))

        # Remove top-level functions that are actually methods
        class_method_names: set = set()
        for cls_item in ps.classes:
            for m in cls_item.children:
                class_method_names.add(m.name)
        ps.functions = [f for f in ps.functions
                        if f.name not in class_method_names
                        or f.line_start not in
                        {c.line_start for c in ps.classes}]

    # ── JavaScript / TypeScript parser (regex) ────────────────────────────────
    @classmethod
    def _parse_js(cls, ps: ParsedScript, raw: str):
        cls._parse_js_ts(ps, raw, "javascript")

    @classmethod
    def _parse_ts(cls, ps: ParsedScript, raw: str):
        cls._parse_js_ts(ps, raw, "typescript")

    @classmethod
    def _parse_js_ts(cls, ps: ParsedScript, raw: str, lang: str):
        lines = raw.splitlines()

        # Imports
        for m in re.finditer(
                r'^(?:import|export\s+(?:default\s+)?)?(?:const|let|var)?\s*'
                r'\{?[^}]*\}?\s*(?:from\s+)?["\'][^"\']+["\']',
                raw, re.MULTILINE):
            ps.imports.append(ParsedItem(
                kind="import", name=m.group(0)[:60],
                signature=m.group(0).split("\n")[0][:120],
                language=lang))

        for m in re.finditer(
                r'^import\s+.*?(?:from\s+)?["\'][^"\']+["\'];?$',
                raw, re.MULTILINE):
            ps.imports.append(ParsedItem(
                kind="import", name=m.group(0)[:60],
                signature=m.group(0)[:120], language=lang))

        # Functions
        for m in re.finditer(
                r'^(?:export\s+)?(?:default\s+)?(?:async\s+)?'
                r'function\s+(\w+)\s*\([^)]*\)',
                raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.functions.append(ParsedItem(
                kind="function", name=m.group(1),
                signature=m.group(0), line_start=ln, language=lang))

        # Arrow functions assigned to const
        for m in re.finditer(
                r'^(?:export\s+)?const\s+(\w+)\s*=\s*'
                r'(?:async\s+)?\([^)]*\)\s*=>',
                raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.functions.append(ParsedItem(
                kind="function", name=m.group(1),
                signature=m.group(0), line_start=ln, language=lang))

        # Classes
        for m in re.finditer(
                r'^(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?',
                raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.classes.append(ParsedItem(
                kind="class", name=m.group(1),
                signature=m.group(0),
                line_start=ln, language=lang))

        # TypeScript types / interfaces
        if lang == "typescript":
            for m in re.finditer(
                    r'^(?:export\s+)?(?:type|interface)\s+(\w+)',
                    raw, re.MULTILINE):
                ln = raw[:m.start()].count("\n")
                ps.types.append(ParsedItem(
                    kind="type", name=m.group(1),
                    signature=m.group(0), line_start=ln, language=lang))

        # Constants
        for m in re.finditer(
                r'^(?:export\s+)?const\s+([A-Z_][A-Z0-9_]{2,})\s*=\s*(.{1,80})',
                raw, re.MULTILINE):
            ps.constants.append(ParsedItem(
                kind="constant", name=m.group(1),
                signature=f"const {m.group(1)} = {m.group(2)[:60]}",
                language=lang))

    # ── SQL parser ────────────────────────────────────────────────────────────
    @classmethod
    def _parse_sql(cls, ps: ParsedScript, raw: str):
        # Tables
        for m in re.finditer(
                r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)',
                raw, re.IGNORECASE):
            ln = raw[:m.start()].count("\n")
            ps.misc.append(ParsedItem(
                kind="table", name=m.group(1),
                signature=m.group(0), line_start=ln, language="sql"))

        # Views
        for m in re.finditer(
                r'CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+(\w+)',
                raw, re.IGNORECASE):
            ln = raw[:m.start()].count("\n")
            ps.misc.append(ParsedItem(
                kind="view", name=m.group(1),
                signature=m.group(0), line_start=ln, language="sql"))

        # Procedures / Functions
        for m in re.finditer(
                r'CREATE\s+(?:OR\s+REPLACE\s+)?'
                r'(?:FUNCTION|PROCEDURE)\s+(\w+)',
                raw, re.IGNORECASE):
            ln = raw[:m.start()].count("\n")
            ps.functions.append(ParsedItem(
                kind="function", name=m.group(1),
                signature=m.group(0), line_start=ln, language="sql"))

        # Indexes
        for m in re.finditer(
                r'CREATE\s+(?:UNIQUE\s+)?INDEX\s+(\w+)\s+ON\s+(\w+)',
                raw, re.IGNORECASE):
            ps.misc.append(ParsedItem(
                kind="index", name=m.group(1),
                signature=m.group(0), language="sql"))

        # Named queries (CTEs)
        for m in re.finditer(r'\bWITH\s+(\w+)\s+AS\s*\(',
                             raw, re.IGNORECASE):
            ps.misc.append(ParsedItem(
                kind="query", name=m.group(1),
                signature=m.group(0), language="sql"))

    # ── Rust parser ───────────────────────────────────────────────────────────
    @classmethod
    def _parse_rust(cls, ps: ParsedScript, raw: str):
        for m in re.finditer(r'^use\s+([^;]+);', raw, re.MULTILINE):
            ps.imports.append(ParsedItem(
                kind="import", name=m.group(1).split("::")[-1],
                signature=m.group(0)[:100], language="rust"))
        for m in re.finditer(
                r'^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)',
                raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.functions.append(ParsedItem(
                kind="function", name=m.group(1),
                signature=m.group(0), line_start=ln, language="rust"))
        for m in re.finditer(
                r'^(?:pub\s+)?struct\s+(\w+)', raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.classes.append(ParsedItem(
                kind="struct", name=m.group(1),
                signature=m.group(0), line_start=ln, language="rust"))
        for m in re.finditer(
                r'^(?:pub\s+)?(?:trait|impl(?:\s+\w+\s+for)?)\s+(\w+)',
                raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.types.append(ParsedItem(
                kind="trait/impl", name=m.group(1),
                signature=m.group(0), line_start=ln, language="rust"))
        for m in re.finditer(
                r'^(?:pub\s+)?const\s+([A-Z_]+)\s*:\s*[^=]+=\s*(.{1,60})',
                raw, re.MULTILINE):
            ps.constants.append(ParsedItem(
                kind="constant", name=m.group(1),
                signature=m.group(0)[:100], language="rust"))

    # ── Go parser ────────────────────────────────────────────────────────────
    @classmethod
    def _parse_go(cls, ps: ParsedScript, raw: str):
        for m in re.finditer(
                r'^import\s*\(([^)]+)\)', raw, re.MULTILINE | re.DOTALL):
            for pkg in m.group(1).splitlines():
                pkg = pkg.strip().strip('"')
                if pkg:
                    ps.imports.append(ParsedItem(
                        kind="import", name=pkg,
                        signature=f'import "{pkg}"', language="go"))
        for m in re.finditer(
                r'^func\s+(?:\(\s*\w+\s+\*?\w+\s*\)\s+)?(\w+)\s*\([^)]*\)',
                raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.functions.append(ParsedItem(
                kind="function", name=m.group(1),
                signature=m.group(0), line_start=ln, language="go"))
        for m in re.finditer(r'^type\s+(\w+)\s+struct', raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.classes.append(ParsedItem(
                kind="struct", name=m.group(1),
                signature=m.group(0), line_start=ln, language="go"))
        for m in re.finditer(r'^type\s+(\w+)\s+interface', raw, re.MULTILINE):
            ps.types.append(ParsedItem(
                kind="interface", name=m.group(1),
                signature=m.group(0), language="go"))

    # ── C / C++ parser ────────────────────────────────────────────────────────
    @classmethod
    def _parse_c(cls, ps: ParsedScript, raw: str):
        cls._parse_cpp(ps, raw)

    @classmethod
    def _parse_cpp(cls, ps: ParsedScript, raw: str):
        for m in re.finditer(r'^#include\s*[<"]([^>"]+)[>"]',
                             raw, re.MULTILINE):
            ps.imports.append(ParsedItem(
                kind="import", name=m.group(1),
                signature=m.group(0), language="cpp"))
        for m in re.finditer(
                r'^(?:class|struct)\s+(\w+)',
                raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.classes.append(ParsedItem(
                kind=m.group(0).split()[0], name=m.group(1),
                signature=m.group(0), line_start=ln, language="cpp"))
        for m in re.finditer(
                r'^(?:[\w:*&<>\s]+)\s+(\w+)\s*\([^)]*\)\s*(?:const)?\s*[{;]',
                raw, re.MULTILINE):
            name = m.group(1)
            if name not in ("if", "while", "for", "switch", "return"):
                ln = raw[:m.start()].count("\n")
                ps.functions.append(ParsedItem(
                    kind="function", name=name,
                    signature=m.group(0).rstrip("{;").strip(),
                    line_start=ln, language="cpp"))
        for m in re.finditer(
                r'^#define\s+([A-Z_][A-Z0-9_]+)\s+(.{1,60})',
                raw, re.MULTILINE):
            ps.constants.append(ParsedItem(
                kind="constant", name=m.group(1),
                signature=m.group(0)[:100], language="cpp"))

    # ── Java / Kotlin parser ──────────────────────────────────────────────────
    @classmethod
    def _parse_java(cls, ps: ParsedScript, raw: str):
        for m in re.finditer(r'^import\s+([\w.]+);', raw, re.MULTILINE):
            ps.imports.append(ParsedItem(
                kind="import", name=m.group(1).split(".")[-1],
                signature=m.group(0), language="java"))
        for m in re.finditer(
                r'^(?:public|private|protected|abstract|final|static)*\s*'
                r'class\s+(\w+)',
                raw, re.MULTILINE):
            ln = raw[:m.start()].count("\n")
            ps.classes.append(ParsedItem(
                kind="class", name=m.group(1),
                signature=m.group(0), line_start=ln, language="java"))
        for m in re.finditer(
                r'(?:public|private|protected|static|final|void|synchronized)*\s+'
                r'(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\([^)]*\)',
                raw, re.MULTILINE):
            if m.group(1) not in ("if","while","for","switch","new","return"):
                ln = raw[:m.start()].count("\n")
                ps.functions.append(ParsedItem(
                    kind="method", name=m.group(1),
                    signature=m.group(0), line_start=ln, language="java"))

    @classmethod
    def _parse_kotlin(cls, ps: ParsedScript, raw: str):
        for m in re.finditer(r'^import\s+([\w.]+)', raw, re.MULTILINE):
            ps.imports.append(ParsedItem(
                kind="import", name=m.group(1).split(".")[-1],
                signature=m.group(0), language="kotlin"))
        for m in re.finditer(
                r'^(?:data\s+)?class\s+(\w+)', raw, re.MULTILINE):
            ps.classes.append(ParsedItem(
                kind="class", name=m.group(1),
                signature=m.group(0), language="kotlin"))
        for m in re.finditer(
                r'^(?:fun|suspend fun|private fun|public fun|internal fun)\s+(\w+)',
                raw, re.MULTILINE):
            ps.functions.append(ParsedItem(
                kind="function", name=m.group(1),
                signature=m.group(0), language="kotlin"))

    # ── Bash parser ───────────────────────────────────────────────────────────
    @classmethod
    def _parse_bash(cls, ps: ParsedScript, raw: str):
        for m in re.finditer(r'^source\s+(\S+)', raw, re.MULTILINE):
            ps.imports.append(ParsedItem(
                kind="import", name=m.group(1),
                signature=m.group(0), language="bash"))
        for m in re.finditer(
                r'^(?:function\s+)?(\w+)\s*\(\s*\)\s*\{',
                raw, re.MULTILINE):
            ps.functions.append(ParsedItem(
                kind="function", name=m.group(1),
                signature=m.group(0), language="bash"))
        for m in re.finditer(
                r'^([A-Z_][A-Z0-9_]*)=(.{1,80})', raw, re.MULTILINE):
            ps.constants.append(ParsedItem(
                kind="constant", name=m.group(1),
                signature=m.group(0)[:100], language="bash"))

    # ── Ruby parser ───────────────────────────────────────────────────────────
    @classmethod
    def _parse_ruby(cls, ps: ParsedScript, raw: str):
        for m in re.finditer(r'^require(?:_relative)?\s+["\']([^"\']+)["\']',
                             raw, re.MULTILINE):
            ps.imports.append(ParsedItem(
                kind="import", name=m.group(1),
                signature=m.group(0), language="ruby"))
        for m in re.finditer(r'^class\s+(\w+)', raw, re.MULTILINE):
            ps.classes.append(ParsedItem(
                kind="class", name=m.group(1),
                signature=m.group(0), language="ruby"))
        for m in re.finditer(r'^\s*def\s+(\w+)', raw, re.MULTILINE):
            ps.functions.append(ParsedItem(
                kind="method", name=m.group(1),
                signature=m.group(0).strip(), language="ruby"))

    # ── JSON / YAML / TOML (structural only) ─────────────────────────────────
    @classmethod
    def _parse_json(cls, ps: ParsedScript, raw: str):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                for k, v in list(data.items())[:40]:
                    ps.misc.append(ParsedItem(
                        kind="key", name=k,
                        signature=f'"{k}": {json.dumps(v)[:80]}',
                        language="json"))
        except Exception as e:
            ps.errors.append(str(e))
            cls._parse_generic(ps, raw)

    @classmethod
    def _parse_yaml(cls, ps: ParsedScript, raw: str):
        # Basic key extraction without pyyaml dependency
        for m in re.finditer(r'^(\w[\w_-]*):\s*(.{0,80})',
                             raw, re.MULTILINE):
            ps.misc.append(ParsedItem(
                kind="key", name=m.group(1),
                signature=f"{m.group(1)}: {m.group(2)[:60]}",
                language="yaml"))

    @classmethod
    def _parse_toml(cls, ps: ParsedScript, raw: str):
        for m in re.finditer(r'^\[([^\]]+)\]', raw, re.MULTILINE):
            ps.misc.append(ParsedItem(
                kind="section", name=m.group(1),
                signature=f"[{m.group(1)}]", language="toml"))
        for m in re.finditer(r'^(\w[\w_-]*)\s*=\s*(.{0,80})',
                             raw, re.MULTILINE):
            ps.constants.append(ParsedItem(
                kind="constant", name=m.group(1),
                signature=f"{m.group(1)} = {m.group(2)[:60]}",
                language="toml"))

    # ── Generic fallback ──────────────────────────────────────────────────────
    @classmethod
    def _parse_generic(cls, ps: ParsedScript, raw: str):
        """Best-effort parser for unknown/unsupported languages."""
        lines = raw.splitlines()
        for i, line in enumerate(lines[:200]):
            stripped = line.strip()
            # Anything that looks like a function/method definition
            if re.match(r'\w.*\(.*\)\s*[:{]?\s*$', stripped) and \
               not stripped.startswith(("#", "//", "*", "/*")):
                ps.misc.append(ParsedItem(
                    kind="item", name=stripped[:50],
                    signature=stripped[:120],
                    line_start=i + 1, language=ps.lang_key))
