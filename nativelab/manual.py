from nativelab.UI.UI_const import C
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Manual — rendered inside _show_manual dialog
# ─────────────────────────────────────────────────────────────────────────────
def make_manual_html() -> str:
    BG   = C.get("bg0","#1e1e2e"); TXT  = C.get("txt","#cdd6f4")
    TXT2 = C.get("txt2","#a6adc8"); ACC  = C.get("acc","#cba6f7")
    ACC2 = C.get("acc2","#a6e3a1"); WARN = C.get("warn","#f9e2af")
    BG2  = C.get("bg2","#313244"); ERR  = C.get("err","#f38ba8")
    OK   = C.get("ok","#a6e3a1");  PL   = C.get("pipeline","#89b4fa")

    def h1(t): return (f'<h1 style="color:{TXT};font-size:17px;font-weight:800;'
                       f'margin:0 0 4px;letter-spacing:0.5px;">{t}</h1>')
    def h2(t): return (f'<h2 style="color:{ACC};font-size:13px;font-weight:700;'
                       f'margin:22px 0 6px;border-bottom:1px solid {BG2};'
                       f'padding-bottom:5px;">{t}</h2>')
    def h3(t): return (f'<h3 style="color:{ACC2};font-size:12px;font-weight:700;'
                       f'margin:12px 0 3px;">{t}</h3>')
    def p(t):  return (f'<p style="color:{TXT2};font-size:11px;margin:3px 0 9px;'
                       f'line-height:1.7;">{t}</p>')
    def note(t): return (f'<p style="color:{WARN};font-size:10.5px;margin:4px 0 10px;'
                         f'background:{BG2};border-left:3px solid {WARN};'
                         f'padding:6px 10px;border-radius:0 5px 5px 0;">'
                         f'⚠️&nbsp; {t}</p>')
    def tip(t):  return (f'<p style="color:{OK};font-size:10.5px;margin:4px 0 10px;'
                         f'background:{BG2};border-left:3px solid {OK};'
                         f'padding:6px 10px;border-radius:0 5px 5px 0;">'
                         f'💡&nbsp; {t}</p>')
    def code(t): return (f'<code style="font-family:Consolas,monospace;font-size:10.5px;'
                         f'color:{ACC2};background:{BG2};padding:1px 5px;'
                         f'border-radius:3px;">{t}</code>')
    def badge(label, col): return (
        f'<span style="background:transparent;color:{col};border:1px solid {col};'
        f'border-radius:9px;padding:2px 8px;font-size:10px;font-weight:700;'
        f'white-space:nowrap;">{label}</span> ')
    def kbd(k): return (f'<span style="background:{BG2};color:{TXT};border:1px solid {ACC};'
                        f'border-radius:3px;padding:1px 6px;font-size:10px;'
                        f'font-family:Consolas,monospace;">{k}</span>')

    def port_table(*rows):
        hdr = (f'<tr><th style="color:{TXT};font-size:10px;font-weight:700;'
               f'padding:0 12px 5px 0;text-align:left;border-bottom:1px solid {BG2};">Port</th>'
               f'<th style="color:{TXT};font-size:10px;font-weight:700;'
               f'padding:0 12px 5px 0;text-align:left;border-bottom:1px solid {BG2};">Direction</th>'
               f'<th style="color:{TXT};font-size:10px;font-weight:700;'
               f'padding:0 0 5px;text-align:left;border-bottom:1px solid {BG2};">Notes</th></tr>')
        body = "".join(
            f'<tr><td style="color:{ACC2};font-size:11px;padding:4px 12px 4px 0;'
            f'font-family:Consolas,monospace;font-weight:700;">{r[0]}</td>'
            f'<td style="color:{OK if "in" in r[1].lower() else ERR};font-size:11px;'
            f'padding:4px 12px 4px 0;">{r[1]}</td>'
            f'<td style="color:{TXT2};font-size:11px;padding:4px 0;">{r[2]}</td></tr>'
            for r in rows)
        return (f'<table cellspacing="0" cellpadding="0" '
                f'style="width:100%;margin:8px 0 14px;">{hdr}{body}</table>')

    def example_box(title, body_html):
        return (f'<div style="background:{BG2};border-radius:6px;'
                f'padding:10px 14px;margin:6px 0 14px;">'
                f'<p style="color:{ACC};font-size:10px;font-weight:700;'
                f'margin:0 0 6px;letter-spacing:0.8px;">{title}</p>'
                f'{body_html}</div>')

    # ── Pre-defined code-display strings that contain backslash literals ─────
    # Backslashes inside f-string {} expressions are a SyntaxError in Python
    # < 3.12, so any string that must display a literal \n is defined here.
    _sort_lines_code = r"result = '\n'.join(sorted(text.split('\n')))"
    # ─────────────────────────────────────────────────────────────────────────

    return f"""<html><body style="background:{BG};color:{TXT};
font-family:Inter,sans-serif;padding:20px 26px 30px;margin:0;line-height:1.5;">

{h1("🔗  NativeLab Pipeline Builder — Full Manual")}
{p(f'Version 2 &nbsp;·&nbsp; All pipeline data stored in {code("./localllm/pipelines/")} as JSON')}

<hr style="border:none;border-top:1px solid {BG2};margin:14px 0 6px;">

{h2("📌  Quick Start — 5 Steps to First Run")}
{p(f'1. <b>Add an ▶ Input block</b> — sidebar left, section <i>Flow Blocks</i>.<br>'
   f'2. <b>Add a ⚡ Model block</b> — drag a model from the sidebar list onto the canvas (ghost appears while hovering) or double-click it.<br>'
   f'3. <b>Add an ■ Output block</b> — same sidebar section.<br>'
   f'4. <b>Draw connections</b> — click a port dot on one block and drag to a port dot on another.<br>'
   f'5. Type text in <i>Input text</i> on the right panel and press <b>▶ Run Pipeline</b>.')}
{tip("Start simple: Input → Model → Output. Add logic blocks only after you confirm the basic run works.")}

{h2("🎨  Canvas Controls")}
{p(f'{kbd("Drag block")} Move any block freely — it snaps to the grid automatically.<br>'
   f'{kbd("Click port dot")} Start drawing a connection arrow.<br>'
   f'{kbd("Drag to port dot")} Complete a connection — creates a curved Bezier arrow.<br>'
   f'{kbd("Right-click block")} Context menu: Delete, Rename, Change Role, Configure.<br>'
   f'{kbd("Right-click canvas")} Clear all blocks and connections.<br>'
   f'{kbd("Pill bar")} The horizontal strip above the canvas shows all model/logic blocks as clickable pills — click one to select it on canvas.')}
{tip("The canvas is larger than the visible area. Use the scrollbars to pan around. Make big pipelines by spreading blocks far apart.")}

{h2("🔵  Port Dots  ( N · S · E · W )")}
{p("Every block has four port dots sitting on its four edges — North (top), South (bottom), East (right), West (left). "
   "Click any dot and drag to any dot on another block to create an arrow. "
   "The arrow curves automatically and adapts its control handles to the port direction.")}
{port_table(
    ("E  →", "Output (default)", "Text leaves the block here — connect to the next block's input"),
    ("W  ←", "Input (default)",  "Text arrives at the block here — connect from the previous block's output"),
    ("N  ↑", "Alt input/output", "Use for branch merges or alternate exits on logic blocks"),
    ("S  ↓", "Alt input/output", "Use for mid-score routing (LLM SCORE) or alternate splits"),
)}
{note("For normal flow blocks (Input, Model, Output, Intermediate) only one arrow per port is allowed. "
     "For all logic blocks and LLM logic blocks multiple arrows can fan out from the same port — this is how branching works.")}
{tip("You can draw arrows in any direction. Input → E is just a convention. The execution engine follows the arrows regardless of port direction.")}

{h2("🟦  Flow Blocks")}

{h3(badge("▶ INPUT", OK) + " Input Block")}
{p("The mandatory starting point. The text you type in the <i>Input text</i> box on the right panel is injected here as the initial context. "
   "You only ever need one. Every pipeline must have exactly one Input block.")}
{port_table(
    ("E  →", "Output", "Sends the raw input text to the first connected block"),
    ("S  ↓", "Output", "Alternative output — use when fanning out to multiple first blocks"),
)}
{example_box("EXAMPLE PIPELINE: Summarise + translate",
    p(f'▶ Input → ⚡ Summariser model → ◈ Intermediate ("Now translate the above to French") → ⚡ Translator model → ■ Output'))}

{h3(badge("◈ INTERMEDIATE", WARN) + " Intermediate Block")}
{p("A pure prompt-injection node — no model is called. It takes the arriving context and wraps your custom instruction around it before passing it on. "
   "Useful for steering the next model without breaking the data flow.")}
{p(f'<b>Right-click → Configure block…</b> to set:<br>'
   f'&nbsp;&nbsp;• <b>Prompt position</b>: above (prompt → model output) or below (model output → prompt)<br>'
   f'&nbsp;&nbsp;• <b>Prompt text</b>: any instruction, e.g. <i>"Now rewrite the above more concisely."</i>')}
{port_table(
    ("W  ←", "Input",  "Receives context from the previous block"),
    ("E  →", "Output", "Sends the wrapped context to the next block"),
)}
{tip("During execution each Intermediate block gets its own live tab in the right panel output area — you can watch the injected context build up in real time.")}
{note("Leaving the prompt blank makes the Intermediate block a transparent pass-through — context flows through unchanged. Useful as a visual separator.")}

{h3(badge("■ OUTPUT", ERR) + " Output Block")}
{p("The terminal node. Whatever context arrives here is shown in the <b>■ Output</b> tab on the right panel. "
   "The pipeline stops when it reaches an Output block. You can have multiple Output blocks — each one terminates its own branch independently.")}
{port_table(
    ("W  ←", "Input",  "Receives the final context — this becomes the displayed output"),
    ("N  ←", "Input",  "Alternative input port for pipelines where the final arrow comes from above"),
)}

{h2("⚡  Model Block")}
{p("Represents a loaded GGUF model. When the pipeline reaches a Model block, NativeLab starts the model in server mode (or reuses it if already running) and sends the current context as a prompt. The response becomes the new context.")}
{p(f'<b>How to add:</b><br>'
   f'&nbsp;&nbsp;• <b>Drag</b> a model from the sidebar list onto the canvas — a ghost block shows the drop position.<br>'
   f'&nbsp;&nbsp;• <b>Double-click</b> a model in the sidebar list to add it at a default position.')}
{p(f'<b>Right-click options:</b><br>'
   f'&nbsp;&nbsp;• <b>Change Role</b> — sets the system prompt used for this block:<br>'
   f'&nbsp;&nbsp;&nbsp;&nbsp;{code("general")} You are a helpful assistant.<br>'
   f'&nbsp;&nbsp;&nbsp;&nbsp;{code("reasoning")} Think step by step.<br>'
   f'&nbsp;&nbsp;&nbsp;&nbsp;{code("summarization")} Be clear and concise.<br>'
   f'&nbsp;&nbsp;&nbsp;&nbsp;{code("coding")} Write clean, well-commented code.<br>'
   f'&nbsp;&nbsp;• <b>Rename block</b> — give it a human-readable name.<br>'
   f'&nbsp;&nbsp;• <b>Delete</b> — removes the block and all its connections.')}
{port_table(
    ("W  ←", "Input",  "Receives the context to use as the prompt"),
    ("E  →", "Output", "Sends the model response to the next block"),
    ("N  ←", "Input",  "Alternate input — use when connecting from above"),
    ("S  →", "Output", "Alternate output — use when chaining downward"),
)}
{note("Direct Model → Model connections are blocked. You must place an ◈ Intermediate block between two models. "
     "This forces you to explicitly define what the second model should do with the first model's output.")}
{tip("Use different roles on different model blocks in the same pipeline. A 'reasoning' model can analyse, then an 'summarization' model can compress the result.")}

{h2("📎  Context Blocks")}
{p("Context blocks inject fixed text into the pipeline without calling a model. They read configuration you set at design time, not at runtime.")}

{h3(badge("📎 REFERENCE", ACC) + " Reference Block")}
{p("Pastes a fixed reference text <b>before</b> the incoming context. Good for injecting background documents, company info, templates, or examples a model should work with.")}
{p(f'<b>Right-click → Configure block…</b> then choose:<br>'
   f'&nbsp;&nbsp;• <b>Type / paste text</b> — opens a multiline input dialog.<br>'
   f'&nbsp;&nbsp;• <b>Load from file</b> — reads any .txt, .md, .py, .json, .yaml file. '
   f'Content is truncated to 4,000 characters with a [truncated] marker if longer.')}
{tip("Name your reference block descriptively (e.g. 'Company FAQ'). The injected text is wrapped in [REFERENCE: name] tags so the model can distinguish it from the user content.")}

{h3(badge("💡 KNOWLEDGE", ACC2) + " Knowledge Block")}
{p("Identical to Reference but labelled as a <i>knowledge base chunk</i> — the injected section is prefixed with <i>Knowledge Base:</i> instead of REFERENCE. "
   "Useful when you want to semantically distinguish instructional reference docs from factual knowledge.")}
{p(f'<b>Right-click → Configure block…</b> — opens a multiline text input. Truncated to 3,000 characters.')}

{h3(badge("📄 PDF SUMMARY", PL) + " PDF Summary Block")}
{p("Loads a PDF file, extracts text from all pages, and injects it. If the PDF exceeds 4,500 characters it is automatically chunk-summarised by the primary engine model before injection.")}
{p(f'<b>Right-click → Configure block…</b> to:<br>'
   f'&nbsp;&nbsp;1. Select the PDF file.<br>'
   f'&nbsp;&nbsp;2. Set the <b>role</b> of the PDF:<br>'
   f'&nbsp;&nbsp;&nbsp;&nbsp;• <b>reference</b> — prior context is MAIN, PDF is supporting reference.<br>'
   f'&nbsp;&nbsp;&nbsp;&nbsp;• <b>main</b> — PDF is the MAIN content, prior context is supporting reference.')}
{note("PyPDF2 must be installed: pip install PyPDF2. If it is missing the block will show a warning when you try to configure it.")}

{h2("⑂  Logic Blocks  ( Python conditions )")}
{p(f'Logic blocks evaluate Python expressions or perform text operations at runtime. '
   f'The variable {code("text")} always holds the incoming context string. '
   f'These run instantly with no model call — use them for fast deterministic branching.')}

{h3(badge("⑂ IF / ELSE", "#f59e0b") + " IF / ELSE")}
{p("Evaluates a Python boolean expression against the incoming text and routes to one of two output arms.")}
{port_table(
    ("E  →", "Output — TRUE branch",  "Followed when the condition evaluates to True / YES"),
    ("W  →", "Output — FALSE branch", "Followed when the condition evaluates to False / NO"),
    ("W  ←", "Input",                 "Receives incoming context"),
)}
{p(f'<b>Configure:</b> Right-click → Configure block… and type a Python expression.<br>'
   f'<b>Draw two arrows</b> from this block, then label each one {code("TRUE")} and {code("FALSE")} when prompted.')}
{example_box("CONDITION EXAMPLES",
    p(f'{code("len(text) > 500")} — route long responses to a summariser<br>'
      f'{code("'error' in text.lower()")} — catch error messages<br>'
      f'{code("text.strip().startswith('```'")} — detect code output<br>'
      f'{code("len(text.split()) < 20")} — detect very short answers<br>'
      f'{code("'yes' in text.lower()[:50]")} — check if model said yes'))}
{note("The expression has access to: len, str, int, float, bool, list, dict, any, all, min, max, abs, isinstance. No file or network access.")}

{h3(badge("⑃ SWITCH", "#f97316") + " SWITCH")}
{p("Evaluates a Python expression that returns a string key, then follows the outgoing arrow whose label matches that key.")}
{port_table(
    ("E/S/W  →", "Output arms", "One per case — label each arrow with the matching key string"),
    ("W  ←",     "Input",       "Receives incoming context"),
)}
{p(f'<b>Configure:</b> Right-click → Configure block… and type an expression that returns a string.<br>'
   f'When drawing each outgoing arrow you will be asked to type the <b>branch label</b> — this must exactly match what the expression can return (case-insensitive).<br>'
   f'Add a {code("default")} labelled arrow to catch unmatched keys.')}
{example_box("EXPRESSION EXAMPLES",
    p(f'{code("'long' if len(text) > 400 else 'short'")} — length-based routing<br>'
      f'{code("'code' if text.strip().startswith('```') else 'prose'")} — format detection<br>'
      f'{code("text.split(':')[0].strip().lower()")} — route on first word / prefix<br>'
      f'{code("'positive' if text.count('!') > 2 else 'neutral'")} — punctuation heuristic'))}

{h3(badge("⊘ FILTER", "#84cc16") + " FILTER")}
{p("A gate. If the condition is TRUE the text continues through the pipeline unchanged. If FALSE the pipeline terminates immediately with a [FILTER DROPPED] message.")}
{port_table(
    ("E  →", "Output — PASS", "Followed only when condition is True"),
    ("W  ←", "Input",         "Receives incoming context"),
)}
{example_box("USE CASES",
    p('• Drop empty or whitespace-only responses before they reach the next model.<br>'
      '• Stop the pipeline if a safety keyword is detected.<br>'
      '• Gate on minimum length to avoid feeding junk to expensive models.'))}
{note("When FILTER drops text it emits a pipeline_done signal with the original text and a reason — this appears in the Output tab so you can debug why it was dropped.")}

{h3(badge("⟲ TRANSFORM", "#06b6d4") + " TRANSFORM")}
{p("Deterministic text operation — no model involved, instant execution. Modifies the context in a fixed, predictable way.")}
{p(f'<b>Available transforms:</b><br>'
   f'&nbsp;&nbsp;• {code("prefix")} — prepend fixed text before the context<br>'
   f'&nbsp;&nbsp;• {code("suffix")} — append fixed text after the context<br>'
   f'&nbsp;&nbsp;• {code("replace")} — find a substring and replace it<br>'
   f'&nbsp;&nbsp;• {code("upper")} / {code("lower")} — change case<br>'
   f'&nbsp;&nbsp;• {code("strip")} — remove leading/trailing whitespace and blank lines<br>'
   f'&nbsp;&nbsp;• {code("truncate")} — cut text to a maximum number of characters')}
{tip("Chain multiple TRANSFORM blocks together to build a text preprocessing pipeline before it hits a model — e.g. strip → truncate → prefix.")}

{h3(badge("⊕ MERGE", "#8b5cf6") + " MERGE")}
{p("Waits for all incoming arrows to deliver their context, then joins them all into a single string that flows to the next block. "
   "Essential after a SPLIT or any fan-out pattern.")}
{p(f'<b>Merge modes:</b><br>'
   f'&nbsp;&nbsp;• {code("concat")} — join with a separator string (default: two newlines + ---)<br>'
   f'&nbsp;&nbsp;• {code("prepend")} — newest first, oldest last<br>'
   f'&nbsp;&nbsp;• {code("append")} — oldest first, newest last<br>'
   f'&nbsp;&nbsp;• {code("json")} — wrap all inputs as a JSON array string')}
{note("MERGE collects all contexts queued for its block ID in the current execution pass. If only one arrow arrives the block still works — it just returns that single input.")}

{h3(badge("⑁ SPLIT", "#ec4899") + " SPLIT")}
{p("Broadcasts the exact same text to every single outgoing arrow simultaneously. Useful for running the same context through multiple models in parallel (they execute sequentially internally).")}
{port_table(
    ("E/S/W  →", "Output × N", "All outgoing arrows fire with identical text"),
    ("W  ←",     "Input",      "Receives one context, fans it to all outputs"),
)}
{tip("SPLIT + MERGE is the classic parallel-processing pattern: split into N model branches, each model processes the same input, MERGE collects all responses.")}
{example_box("PARALLEL REVIEW PATTERN",
    p('▶ Input → ⑁ SPLIT → ⚡ Reviewer A → ⊕ MERGE → ◈ Intermediate ("Combine the two reviews") → ⚡ Model → ■ Output<br>'
      '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↘ ⚡ Reviewer B ↗'))}

{h3(badge("⌥ CUSTOM CODE", "#10b981") + " Custom Code")}
{p("Write arbitrary Python that runs inline during pipeline execution. Full code editor with live syntax checking and a test runner.")}
{p(f'<b>Available variables:</b><br>'
   f'&nbsp;&nbsp;• {code("text")} — incoming context string (read-only)<br>'
   f'&nbsp;&nbsp;• {code("result")} — set this to your output string (defaults to {code("text")} if not set)<br>'
   f'&nbsp;&nbsp;• {code("metadata")} — block metadata dict, persists across pipeline runs<br>'
   f'&nbsp;&nbsp;• {code("log(msg)")} — writes a message to the Pipeline Log tab')}
{p(f'<b>Safe builtins available:</b> {code("len str int float bool list dict tuple range enumerate zip map filter sorted min max sum abs round isinstance hasattr getattr repr type print")}')}
{note("No file I/O, no network access, no os/subprocess. The exec() is sandboxed. If your code raises an exception the pipeline stops with the error message shown in the Log tab.")}
{example_box("CODE EXAMPLES",
    p(f'{code("result = text.upper()")} — uppercase<br>'
      f'{code(_sort_lines_code)} — sort lines<br>'
      f'{code("result = str(len(text.split())) + ' words: ' + text")} — prepend word count<br>'
      f'{code("import_count = metadata.get('runs', 0) + 1; metadata['runs'] = import_count; log(f'Run #{import_count}')")} — stateful counter'))}

{h2("🧠  LLM Logic Blocks  ( plain English conditions )")}
{p("LLM logic blocks work identically to their Python counterparts except the condition or instruction is written in <b>plain English</b> and the attached model evaluates it. "
   "The model is called with a tight system prompt that demands a specific answer format (YES/NO, a category name, PASS/STOP, etc.).")}
{p(f'<b>Each LLM logic block requires:</b><br>'
   f'&nbsp;&nbsp;1. A <b>GGUF model</b> selected in the config dialog (can be any registered model).<br>'
   f'&nbsp;&nbsp;2. A <b>plain English instruction</b> written by you.')}
{p(f'<b>Advanced settings</b> (expand in the config dialog):<br>'
   f'&nbsp;&nbsp;• <b>Max response tokens</b> — keep small (16–64) for routing, larger (128–512) for transforms.<br>'
   f'&nbsp;&nbsp;• <b>Temperature</b> — use 0–15 for deterministic routing, 30–60 for creative transforms.<br>'
   f'&nbsp;&nbsp;• <b>Show model reasoning in log</b> — logs the raw model response before it is parsed.<br>'
   f'&nbsp;&nbsp;• <b>Passthrough on error</b> — if the model call fails, pass text through unchanged instead of stopping.')}
{tip("Use a small fast model (e.g. Qwen2.5-0.5B or TinyLlama) for LLM routing blocks. They only need to say YES/NO — you do not need a 7B model for that.")}

{h3(badge("🧠 LLM IF / ELSE", "#a855f7") + " LLM IF / ELSE")}
{p("The model reads your condition and the incoming text, then answers with a single word: YES or NO.")}
{port_table(
    ("E  →", "Output — TRUE / YES",  "Followed when model answers YES"),
    ("W  →", "Output — FALSE / NO",  "Followed when model answers NO"),
)}
{example_box("CONDITION EXAMPLES",
    p('• Does this text contain a complaint or expression of frustration?<br>'
      '• Is the answer longer than a brief paragraph?<br>'
      '• Does the user seem confused or are they asking a follow-up question?<br>'
      '• Is this response in English?<br>'
      '• Does this text contain any personally identifiable information?'))}
{note("The parser accepts: YES Y TRUE 1 PASS POSITIVE as truthy. Everything else is FALSE. Enable 'Show model reasoning' to debug unexpected routing.")}

{h3(badge("🧠 LLM SWITCH", "#7c3aed") + " LLM SWITCH")}
{p("The model classifies the text into one of the categories you define. The category names are automatically read from the labels you put on the outgoing arrows.")}
{port_table(
    ("E/S/W  →", "Output arms", "One per category — label each arrow with the exact category name"),
)}
{example_box("INSTRUCTION EXAMPLES",
    p('• Classify this text as one of: positive, negative, neutral<br>'
      '• What language is this written in? english, french, spanish, german, other<br>'
      '• Is this a question, a complaint, a compliment, or a general statement?<br>'
      '• Route by topic: technical, billing, account, other<br>'
      '• What kind of content is this: code, prose, data, mixed?'))}
{tip("Always include a 'default' or 'other' labelled arrow to catch cases where the model returns something unexpected. The fallback prevents silent drops.")}

{h3(badge("🧠 LLM FILTER", "#6366f1") + " LLM FILTER")}
{p("The model decides whether the text should continue (PASS) or be dropped (STOP). When dropped the pipeline ends with a structured message explaining which filter stopped it and the model's reason.")}
{example_box("CONDITION EXAMPLES",
    p('• Only pass if this is a genuine technical support question<br>'
      '• Pass only if the response contains a concrete action item<br>'
      '• Allow through only if the content is safe and not harmful<br>'
      '• Pass only if the answer is at least two sentences long<br>'
      '• Block any response that mentions competitor products by name'))}
{note("The FILTER stop message is displayed in the Output tab with the filter name, condition, model decision, and the original text — so you can inspect exactly what was dropped and why.")}

{h3(badge("🧠 LLM TRANSFORM", "#0ea5e9") + " LLM TRANSFORM")}
{p("The model rewrites, reformats, or transforms the incoming text according to your instruction. The result replaces the context for all downstream blocks. "
   "Increase Max response tokens to 256–512 for this block type.")}
{example_box("INSTRUCTION EXAMPLES",
    p('• Summarise this in exactly three bullet points, each one sentence<br>'
      '• Rewrite in a formal professional business tone<br>'
      '• Extract only the action items as a numbered list<br>'
      '• Translate to Spanish keeping all technical terms in English<br>'
      '• Convert this prose into a structured JSON object with keys: title, summary, keywords<br>'
      '• Remove all filler words and redundant phrases, keep only the core meaning'))}
{tip("The transform block automatically strips common preamble phrases the model might add (Here is..., Result:, Output:) before passing the result downstream.")}

{h3(badge("🧠 LLM SCORE", "#d946ef") + " LLM SCORE")}
{p("The model rates the incoming text on your criterion from 1 to 10. The score is parsed from the response and used to route to one of three band arms.")}
{port_table(
    ("E  →", "Output — LOW (1–3)",   "Route to escalation, retry, or human review"),
    ("S  →", "Output — MID (4–7)",   "Route to standard processing"),
    ("W  →", "Output — HIGH (8–10)", "Route to fast-track or direct output"),
)}
{p(f'Label an outgoing arrow {code("score")} to receive the raw numeric score as text instead of the original context — useful for feeding the score into a TRANSFORM or OUTPUT.')}
{example_box("CRITERION EXAMPLES",
    p('• Rate the clarity and readability of this explanation (1=very unclear, 10=crystal clear)<br>'
      '• Score the sentiment positivity (1=very negative, 10=very positive)<br>'
      '• Rate the technical complexity (1=trivial, 10=expert-level)<br>'
      '• How complete and thorough is this answer? (1=missing key info, 10=comprehensive)<br>'
      '• Score the urgency of this message (1=low priority, 10=critical/immediate action needed)'))}

{h2("🔄  Loops")}
{p("Draw an arrow <b>backwards</b> — from a downstream block to an upstream block. NativeLab detects the cycle and asks how many times the loop should iterate (min 2, max 999). "
   "Loop arrows are shown as <b>dashed dash-dot lines</b> with a ×N badge at the midpoint.")}
{p(f'<b>How execution works:</b><br>'
   f'Each time the pipeline reaches the source block of a loop edge it checks how many times that specific edge has already been followed. '
   f'Once the limit is reached the edge is skipped and execution continues to the next non-loop outgoing connection.')}
{example_box("REFINEMENT LOOP PATTERN",
    p('▶ Input → ⚡ Model → ◈ Intermediate ("Critique the above and list improvements") → ⚡ Model<br>'
      '(draw a backwards arrow from the second Model back to the first Intermediate, set ×3)<br>'
      '→ ■ Output'))}
{tip("Connect a non-loop arrow from the loop body to an Output block to capture the final result after all iterations complete.")}

{h2("💾  Save & Load Pipelines")}
{p(f'Click <b>💾 Save Pipeline…</b> in the sidebar. Type a name — existing names are overwritten without warning.<br>'
   f'Click <b>📂 Load Pipeline…</b> to restore a saved pipeline. If the canvas has blocks you will be asked to confirm replacement.<br>'
   f'Pipelines are stored as JSON in {code("./localllm/pipelines/name.json")}.<br>'
   f'To delete: Load dialog → select <i>🗑 Delete a pipeline…</i> option.')}
{p(f'<b>What is saved per block:</b> type, position, size, model path, role, label, all metadata (prompt text, code, conditions, PDF path, etc.).<br>'
   f'<b>What is saved per connection:</b> from/to block IDs, ports, is_loop flag, loop_times, branch label.')}
{note("Model files are saved as absolute paths. If you move a .gguf file the pipeline will load but the model block will show 'no valid file' and validation will fail until you re-attach the model.")}

{h2("🐛  Debugging & Troubleshooting")}
{p(f'<b>📋 Log tab</b> — shows every step: block started, chars processed, decisions made, errors. Always check this first.<br>'
   f'<b>◈ Intermediate tabs</b> — each intermediate block gets a live tab showing exactly what text arrived at it.<br>'
   f'<b>■ Output tab</b> — shows the final rendered output with markdown, code highlighting, and bold.')}
{p(f'<b>Common errors:</b>')}
{p(f'• <i>No INPUT block</i> — add ▶ Input and connect it to something.<br>'
   f'• <i>No OUTPUT block</i> — add ■ Output and connect the last block to it.<br>'
   f'• <i>No connections drawn</i> — you must draw at least one arrow between blocks.<br>'
   f'• <i>Model block has no valid file</i> — double-click a model in the sidebar or check the file still exists.<br>'
   f'• <i>Reference / Knowledge has no text</i> — right-click → Configure before running.<br>'
   f'• <i>IF/ELSE has no condition</i> — right-click → Configure and type a Python expression.<br>'
   f'• <i>LLM logic block: no model</i> — open config dialog and select a GGUF model.<br>'
   f'• <i>LLM logic block: model not found</i> — the model file was moved; reconfigure the block.<br>'
   f'• <i>Engine Not Ready</i> — wait for the primary model to finish loading in the Server tab.<br>'
   f'• <i>Server HTTP 500</i> — the model is loaded but errored; check the Logs tab in the Server tab for details.<br>'
   f'• <i>FILTER DROPPED</i> in output — the filter condition was FALSE; check the condition logic or increase tolerance.')}
{tip("Press ⏹ Stop Execution at any time. The pipeline will finish its current HTTP request and then halt cleanly.")}

{h2("⚡  Performance Tips")}
{p(f'• Keep context short between models — truncate with a TRANSFORM block before expensive model calls.<br>'
   f'• Use small models (0.5B–1.5B) for all LLM logic routing blocks. Reserve big models for the actual generation steps.<br>'
   f'• PDF blocks auto-summarise large documents — but this adds extra model calls. Pre-summarise large PDFs offline if speed matters.<br>'
   f'• Loop iterations multiply model calls: 3 models × 5 loops = 15 server requests. Plan accordingly.<br>'
   f'• Set Max response tokens conservatively on routing blocks (16 is enough for YES/NO decisions).')}

</body></html>"""

PIPELINE_MANUAL_HTML = make_manual_html()