from nativelab.imports.import_global import Optional, Dict, re


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
        return html

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

    # ── LaTeX / KaTeX-style math delimiters ─────────────────────────────────
    _math_bg = colors["bg2"]
    _math_bd = colors["bdr"]
    _math_fg = colors["acc2"]

    def _display_math(m: re.Match) -> str:
        body = _latex_to_readable(m.group(1))
        return (
            f'<table width="100%" cellpadding="0" cellspacing="0" '
            f'style="background:{_math_bg};border:1px solid {_math_bd};'
            f'border-radius:6px;margin:8px 0;">'
            f'<tr><td align="center" style="padding:10px 14px;'
            f'font-family:Cambria Math,Times New Roman,serif;'
            f'font-size:15px;color:{_math_fg};">{body}</td></tr></table>'
        )

    def _inline_math(m: re.Match) -> str:
        body = _latex_to_readable(m.group(1))
        return (
            f'<span style="font-family:Cambria Math,Times New Roman,serif;'
            f'background:{_math_bg};color:{_math_fg};border:1px solid {_math_bd};'
            f'border-radius:4px;padding:0 4px;">{body}</span>'
        )

    text = re.sub(r'\$\$([\s\S]+?)\$\$', _display_math, text)
    text = re.sub(r'\\\[([\s\S]+?)\\\]', _display_math, text)
    text = re.sub(r'\\\((.+?)\\\)', _inline_math, text)
    text = re.sub(r'(?<!\$)\$([^\n$]+?)\$(?!\$)', _inline_math, text)

    # ── Inline code ──────────────────────────────────────────────────────────
    _ic_bg  = colors["bg2"]
    _ic_acc = colors["acc"]
    _h_col  = colors["acc"]
    text = re.sub(
        r'`([^`\n]+)`',
        lambda m: (f'<code style="background:{_ic_bg};border-radius:3px;'
                   f'padding:1px 5px;font-family:Consolas,monospace;'
                   f'font-size:12px;color:{_ic_acc};">{m.group(1)}</code>'),
        text)

    # ── Headers ──────────────────────────────────────────────────────────────
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

    # ── Newlines → <br> (only outside block tags) ────────────────────────────
    parts = re.split(r'(<table[\s\S]*?</table>|<p[\s\S]*?</p>)', text)
    out = []
    for part in parts:
        if part.startswith('<'):
            out.append(part)
        else:
            out.append(part.replace('\n', '<br>'))
    return ''.join(out)
