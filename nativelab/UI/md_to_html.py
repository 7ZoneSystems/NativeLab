import json
import shutil
import subprocess
import re
from functools import lru_cache
from html import escape as html_escape, unescape as html_unescape
from pathlib import Path
from typing import Dict, Optional


KATEX_DIR = Path(__file__).resolve().parents[1] / "assets" / "katex"
KATEX_JS = KATEX_DIR / "katex.min.js"
KATEX_CSS = KATEX_DIR / "katex.min.css"
KATEX_NODE_TIMEOUT_SECONDS = 2.0


def _latex_to_readable(expr: str) -> str:
    expr = expr.strip()
    greek = {
        "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
        "epsilon": "ε", "theta": "θ", "lambda": "λ", "mu": "μ",
        "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ",
        "phi": "φ", "omega": "ω", "Delta": "Δ", "Theta": "Θ",
        "Lambda": "Λ", "Pi": "Π", "Sigma": "Σ", "Omega": "Ω",
    }
    ops = {
        r"\times": "×", r"\cdot": "·", r"\div": "÷", r"\pm": "±",
        r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈",
        r"\infty": "∞", r"\to": "→", r"\rightarrow": "→",
        r"\left": "", r"\right": "",
    }
    for k, v in ops.items():
        expr = expr.replace(k, v)
    for name, char in greek.items():
        expr = expr.replace("\\" + name, char)
    expr = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', expr)
    expr = re.sub(r'\\sqrt\{([^{}]+)\}', r'√(\1)', expr)
    expr = re.sub(r'\\text\{([^{}]+)\}', r'\1', expr)
    expr = expr.replace("{", "").replace("}", "")
    return expr


@lru_cache(maxsize=1)
def _katex_css_html() -> str:
    if not KATEX_CSS.exists():
        return ""
    try:
        css = KATEX_CSS.read_text(encoding="utf-8")
        font_root = (KATEX_DIR / "fonts").resolve().as_uri() + "/"
        css = css.replace("url(fonts/", f"url({font_root}")
        return f"<style>{css}</style>"
    except Exception:
        return ""


@lru_cache(maxsize=512)
def _render_katex(expr: str, display: bool) -> str:
    """Render TeX with the vendored KaTeX bundle when Node is available."""
    if not KATEX_JS.exists() or not shutil.which("node"):
        return ""
    script = (
        "const katex=require(process.argv[1]);"
        "let input='';"
        "process.stdin.on('data',d=>input+=d);"
        "process.stdin.on('end',()=>{"
        " const req=JSON.parse(input);"
        " try {"
        "  const html=katex.renderToString(req.expr,{"
        "    displayMode:!!req.display,throwOnError:false,strict:false,"
        "    trust:true,output:'html'"
        "  });"
        "  process.stdout.write(JSON.stringify({ok:true,html}));"
        " } catch(e) {"
        "  process.stdout.write(JSON.stringify({ok:false,error:String(e&&e.message||e)}));"
        " }"
        "});"
    )
    try:
        proc = subprocess.run(
            ["node", "-e", script, str(KATEX_JS.resolve())],
            input=json.dumps({"expr": expr, "display": display}),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=KATEX_NODE_TIMEOUT_SECONDS,
            check=False,
        )
        data = json.loads(proc.stdout or "{}")
        return str(data.get("html") or "") if data.get("ok") else ""
    except Exception:
        return ""


def _fallback_math_html(expr: str, display: bool, colors: Dict[str, str]) -> str:
    text = html_escape(_latex_to_readable(expr), quote=False)
    bg = colors["bg2"]
    bd = colors["bdr"]
    fg = colors["acc2"]
    if display:
        return (
            f'<table width="100%" cellpadding="0" cellspacing="0" '
            f'style="background:{bg};border:1px solid {bd};'
            f'border-radius:6px;margin:8px 0;">'
            f'<tr><td align="center" style="padding:10px 14px;'
            f'font-family:Cambria Math,Times New Roman,serif;'
            f'font-size:15px;color:{fg};">{text}</td></tr></table>'
        )
    return (
        f'<span style="font-family:Cambria Math,Times New Roman,serif;'
        f'background:{bg};color:{fg};border:1px solid {bd};'
        f'border-radius:4px;padding:0 4px;">{text}</span>'
    )


def _math_html(expr: str, display: bool, colors: Dict[str, str],
               katex_used: list) -> str:
    rendered = _render_katex(expr, display)
    if rendered:
        katex_used[0] = True
        if display:
            return (
                f'<table width="100%" cellpadding="0" cellspacing="0" '
                f'style="background:{colors["bg2"]};border:1px solid {colors["bdr"]};'
                f'border-radius:6px;margin:8px 0;">'
                f'<tr><td align="center" style="padding:10px 14px;'
                f'color:{colors["txt"]};">{rendered}</td></tr></table>'
            )
        return rendered
    return _fallback_math_html(expr, display, colors)


def _is_table_separator(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) < 2:
        return False
    return all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells)


def _split_table_row(line: str) -> list:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table_alignments(separator: str, n_cols: int) -> list:
    aligns = []
    for cell in _split_table_row(separator):
        left = cell.startswith(":")
        right = cell.endswith(":")
        aligns.append("center" if left and right else "right" if right else "left")
    while len(aligns) < n_cols:
        aligns.append("left")
    return aligns[:n_cols]


def _inline_cell_format(cell: str, colors: Dict[str, str]) -> str:
    ic_bg = colors["bg2"]
    ic_acc = colors["acc"]
    cell = re.sub(
        r'`([^`\n]+)`',
        lambda m: (f'<code style="background:{ic_bg};border-radius:3px;'
                   f'padding:1px 5px;font-family:Consolas,monospace;'
                   f'font-size:12px;color:{ic_acc};">{m.group(1)}</code>'),
        cell,
    )
    cell = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', cell)
    cell = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', cell)
    cell = re.sub(r'\*(.+?)\*', r'<i>\1</i>', cell)
    return cell


def _render_markdown_table(lines: list, colors: Dict[str, str]) -> str:
    header = _split_table_row(lines[0])
    aligns = _table_alignments(lines[1], len(header))
    body_rows = [_split_table_row(line) for line in lines[2:]]
    bdr = colors["bdr"]
    head_bg = colors["bg2"]
    body_bg = colors["surface"]
    txt = colors["txt"]
    txt2 = colors["txt2"]
    html = [
        f'<table width="100%" cellpadding="0" cellspacing="0" '
        f'style="background:{body_bg};border:1px solid {bdr};'
        f'border-radius:6px;margin:8px 0;">'
    ]
    html.append("<tr>")
    for idx, cell in enumerate(header):
        html.append(
            f'<td align="{aligns[idx]}" style="background:{head_bg};'
            f'border-bottom:1px solid {bdr};padding:6px 8px;'
            f'color:{txt};font-weight:700;font-size:12px;">'
            f'{_inline_cell_format(cell, colors)}</td>'
        )
    html.append("</tr>")
    for row in body_rows:
        row = row + [""] * (len(header) - len(row))
        html.append("<tr>")
        for idx, cell in enumerate(row[:len(header)]):
            html.append(
                f'<td align="{aligns[idx]}" style="border-top:1px solid {bdr};'
                f'padding:6px 8px;color:{txt2};font-size:12px;">'
                f'{_inline_cell_format(cell, colors)}</td>'
            )
        html.append("</tr>")
    html.append("</table>")
    return "".join(html)


def md_to_html(text: str,
                code_store: Optional[Dict[str, str]] = None,
                colors: Optional[Dict[str, str]] = None) -> str:
    """
    Markdown → HTML for Qt's limited renderer.

    Parameters
    ----------
    text       : raw markdown / plain text from the model
    code_store : dict that will be filled with {block_id: raw_code}.
                 Pass the RichTextEdit's ._code_blocks dict so copy works.
    colors     : dict of color values for theming. If None, uses default theme.

    Qt renderer constraints
    ───────────────────────
    <table>, <tr>, <td align="…">
    Inline styles: color, background-color, font-family, font-size,
        font-weight, padding, margin, border (simple 1px solid …),
        white-space:pre, width
    float, position, display:flex/grid
    JavaScript / onclick
    CSS pseudo-elements, :hover
    """
    if colors is None:
        from .UI_const import C
        colors = C

    if code_store is None:
        code_store = {}

    placeholders: Dict[str, str] = {}
    katex_used = [False]

    def _stash(html: str) -> str:
        key = f"@@NLHTML{len(placeholders)}@@"
        placeholders[key] = html
        return key

    # Escape HTML entities BEFORE any substitutions
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    _counter = [0]

    # ── Fenced code blocks ──────────────────────────────────────────────────
    def _fenced(m: re.Match) -> str:
        lang  = m.group(1).strip().lower()
        body  = m.group(2)          # already HTML-escaped by outer replace
        bid   = f"cb{abs(hash(body[:40]))}_{_counter[0]}"
        _counter[0] += 1

        # Raw code for clipboard (un-escape entities)
        raw_code = (body
                    .replace("&amp;", "&")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">"))
        code_store[bid] = raw_code

        n_lines = body.count("\n") + 1
        lang_up = lang.upper() if lang else "CODE"

        LANG_COL = {
            "python": "#f59e0b", "py": "#f59e0b",
            "javascript": "#f7df1e", "js": "#f7df1e",
            "typescript": "#3178c6", "ts": "#3178c6",
            "sql":  "#00b4d8", "rust": "#f74c00",
            "bash": "#4ade80", "sh": "#4ade80",
            "go":   "#00acd7", "c":   "#a9b8c3",
            "cpp":  "#f34b7d", "java": "#b07219",
            "json": "#cbcb41", "yaml": "#cc3e44",
        }.get(lang, "#a78bfa")

        # Syntax highlighting (Python only - safe regex on escaped text)
        display = body
        if lang in ("python", "py", ""):
            KW   = "#c792ea"; STR = "#c3e88d"; NUM = "#f78c6c"; CMT = "#546e7a"
            # Comments first (they take priority)
            display = re.sub(
                r'(#[^\n]*)',
                f'<span style="color:{CMT};">\\1</span>',
                display)
            # Strings (double)
            display = re.sub(
                r'(&quot;(?:[^&]|&(?!quot;))*?&quot;|"(?:[^"\\]|\\.)*?")',
                f'<span style="color:{STR};">\\1</span>',
                display)
            # Numbers
            display = re.sub(
                r'\b(\d+\.?\d*)\b',
                f'<span style="color:{NUM};">\\1</span>',
                display)
            # Keywords
            for kw in ("def","class","import","from","return","yield",
                       "if","elif","else","for","while","try","except",
                       "finally","with","as","pass","break","continue",
                       "lambda","and","or","not","in","is","None",
                       "True","False","async","await","raise","del",
                       "global","nonlocal","assert"):
                display = re.sub(
                    rf'\b({re.escape(kw)})\b',
                    f'<span style="color:{KW};font-weight:600;">\\1</span>',
                    display)

        # Table-based layout (Qt supports tables reliably)
        _cb  = colors["bg2"]
        _tb  = colors["bg1"]
        _bdr = colors["bdr"]
        _ctxt = colors["txt"]
        _dim = colors["txt3"]
        _lnk = colors["acc"]

        html = (
            f'<table width="100%" cellpadding="0" cellspacing="0" '
            f'style="background:{_cb};'
            f'border:1px solid {_bdr};'
            f'border-radius:6px;margin:8px 0;">'

            f'<tr>'
            f'<td style="padding:5px 12px;'
            f'border-bottom:1px solid {_bdr};'
            f'background:{_tb};">'
            f'<span style="color:{LANG_COL};font-size:9px;font-weight:700;'
            f'font-family:Consolas,monospace;background:transparent;'
            f'border:1px solid {_bdr};border-radius:4px;'
            f'padding:1px 6px;">{lang_up}</span>'
            f'<span style="color:{_dim};font-size:10px;"> &nbsp;{n_lines} lines</span>'
            f'</td>'
            f'<td align="right" style="padding:5px 12px;'
            f'border-bottom:1px solid {_bdr};'
            f'background:{_tb};">'
            f'<a href="copy://{bid}" '
            f'style="color:{_lnk};font-size:10px;'
            f'text-decoration:none;font-family:Segoe UI,sans-serif;">'
            f'⧉ Copy</a>'
            f'</td>'
            f'</tr>'

            f'<tr>'
            f'<td colspan="2" style="padding:10px 14px;background:{_cb};">'
            f'<pre style="margin:0;'
            f'font-family:Consolas,&quot;Courier New&quot;,monospace;'
            f'font-size:12px;color:{_ctxt};background:{_cb};'
            f'white-space:pre-wrap;line-height:1.6;">'
            f'{display}</pre>'
            f'</td>'
            f'</tr>'
            f'</table>'
        )
        return _stash(html)

    text = re.sub(r'```(\w*)\n?(.*?)```', _fenced, text, flags=re.DOTALL)

    # ── Thinking blocks (<think>, [think], and common typo [thinl]) ─────────
    _think_bg = colors["bg2"]
    _think_bd = colors["bdr"]
    _think_fg = colors["txt2"]

    def _think_block(m: re.Match) -> str:
        body = m.group(1).strip().replace("\n", "<br>")
        return (
            f'<table width="100%" cellpadding="0" cellspacing="0" '
            f'style="background:{_think_bg};border:1px solid {_think_bd};'
            f'border-radius:6px;margin:8px 0;">'
            f'<tr><td style="padding:6px 10px;color:{_think_fg};'
            f'font-size:10px;font-weight:700;">Thinking</td></tr>'
            f'<tr><td style="padding:8px 12px;color:{_think_fg};'
            f'font-size:12px;">{body}</td></tr></table>'
        )

    text = re.sub(r'&lt;think&gt;([\s\S]*?)&lt;/think&gt;', _think_block, text, flags=re.IGNORECASE)
    text = re.sub(r'\[(?:think|thinl)\]([\s\S]*?)\[/(?:think|thinl)\]', _think_block, text, flags=re.IGNORECASE)

    # ── Inline code ──────────────────────────────────────────────────────────
    _ic_bg  = colors["bg2"]
    _ic_acc = colors["acc"]

    def _inline_code(m: re.Match) -> str:
        return _stash(
            f'<code style="background:{_ic_bg};border-radius:3px;'
            f'padding:1px 5px;font-family:Consolas,monospace;'
            f'font-size:12px;color:{_ic_acc};">{m.group(1)}</code>'
        )

    text = re.sub(r'`([^`\n]+)`', _inline_code, text)

    # ── LaTeX / TeX / KaTeX-style math delimiters ────────────────────────────
    def _display_math(m: re.Match) -> str:
        expr = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
        return _stash(_math_html(html_unescape(expr.strip()), True, colors, katex_used))

    def _inline_math(m: re.Match) -> str:
        return _stash(_math_html(html_unescape(m.group(1).strip()), False, colors, katex_used))

    envs = (
        "equation\\*?|align\\*?|aligned|gather\\*?|multline\\*?|"
        "matrix|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|cases|split"
    )
    text = re.sub(rf'\\begin\{{({envs})\}}([\s\S]+?)\\end\{{\1\}}', _display_math, text)
    text = re.sub(r'\$\$([\s\S]+?)\$\$', _display_math, text)
    text = re.sub(r'\\\[([\s\S]+?)\\\]', _display_math, text)
    text = re.sub(r'\\\((.+?)\\\)', _inline_math, text)
    text = re.sub(r'(?<!\\)(?<!\$)\$([^\n$]+?)(?<!\\)\$(?!\$)', _inline_math, text)

    # ── GitHub-style markdown tables ─────────────────────────────────────────
    def _extract_tables(src: str) -> str:
        lines = src.split("\n")
        out = []
        i = 0
        while i < len(lines):
            if (
                i + 1 < len(lines)
                and "|" in lines[i]
                and "|" in lines[i + 1]
                and _is_table_separator(lines[i + 1])
            ):
                table_lines = [lines[i], lines[i + 1]]
                i += 2
                while i < len(lines) and "|" in lines[i] and lines[i].strip():
                    table_lines.append(lines[i])
                    i += 1
                out.append(_stash(_render_markdown_table(table_lines, colors)))
                continue
            out.append(lines[i])
            i += 1
        return "\n".join(out)

    text = _extract_tables(text)

    # ── Headers ──────────────────────────────────────────────────────────────
    _h_col  = colors["acc"]
    text = re.sub(r'^### (.+)$',
        rf'<p style="color:{_h_col};font-size:13px;margin:10px 0 2px;'
        r'font-weight:600;">\1</p>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$',
        rf'<p style="color:{_h_col};font-size:14px;margin:12px 0 3px;'
        r'font-weight:700;">\1</p>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$',
        rf'<p style="color:{_h_col};font-size:15px;margin:12px 0 4px;'
        r'font-weight:700;">\1</p>', text, flags=re.MULTILINE)

    # ── Bold / italic ─────────────────────────────────────────────────────────
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
    text = re.sub(r'\*\*(.+?)\*\*',     r'<b>\1</b>',         text)
    text = re.sub(r'\*(.+?)\*',         r'<i>\1</i>',          text)

    # ── Horizontal rules ─────────────────────────────────────────────────────
    _bdr_c = colors["bdr"]
    _bull  = colors["acc"]
    text = re.sub(
        r'^---+$',
        rf'<table width="100%" cellpadding="0" cellspacing="0" style="margin:8px 0;">'
        rf'<tr><td style="border-top:1px solid {_bdr_c};"></td></tr></table>',
        text, flags=re.MULTILINE)

    # ── Bullet lists ──────────────────────────────────────────────────────────
    text = re.sub(
        r'^[ \t]*[-*•] (.+)$',
        rf'<p style="margin:2px 0;padding-left:16px;">'
        rf'<span style="color:{_bull};">•</span>&nbsp;\1</p>',
        text, flags=re.MULTILINE)

    # ── Numbered lists ────────────────────────────────────────────────────────
    text = re.sub(
        r'^[ \t]*(\d+)\. (.+)$',
        rf'<p style="margin:2px 0;padding-left:16px;">'
        rf'<span style="color:{_bull};">\1.</span>&nbsp;\2</p>',
        text, flags=re.MULTILINE)

    for _ in range(len(placeholders) + 1):
        replaced = False
        for key, html in placeholders.items():
            if key in text:
                text = text.replace(key, html)
                replaced = True
        if not replaced:
            break

    # ── Newlines → <br> (only outside block tags) ────────────────────────────
    parts = re.split(r'(<table[\s\S]*?</table>|<p[\s\S]*?</p>)', text)
    out = []
    for part in parts:
        if part.startswith('<'):
            out.append(part)
        else:
            out.append(part.replace('\n', '<br>'))
    rendered = ''.join(out)
    if katex_used[0]:
        rendered = _katex_css_html() + rendered
    return rendered
