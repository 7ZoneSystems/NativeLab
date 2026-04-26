"""
Native Lab Pro v2 — Local LLM Desktop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v2 New Features:
  · Full GGUF quant format detection (Q2_K → F32, IQ* imatrix quants)
  · Auto model-family detection from filename → correct prompt template
    (DeepSeek, Mistral, LLaMA-2/3, Phi, Qwen/ChatML, Gemma, CodeLlama,
     Falcon, Vicuna, OpenChat, Neural-Chat, Starling, Yi, Command-R…)
  · Parallel model loading toggle with CPU/RAM warnings
  · Pipeline mode: Reasoning → Coding chain
    (reasoning model summarises intent → coding model generates code)
  · Python/code snippet copy buttons inside chat bubbles
  · All existing v1 features preserved
"""
#import 
from nativelab.imports.import_global import *
from nativelab.GlobalConfig.config_global import *
from nativelab.Model.model_global import *
from nativelab.UI.UI_global import *
from nativelab.Prefrences.prefrence_global import *
from nativelab.Server.server_global import *
from nativelab.core.streamer_global import *
from nativelab.components.components_global import *
from nativelab.core.engine_global import *
from nativelab.codeparser.codeparser_global import *
from nativelab.pipelinebuilder.pipe_global import *
class ModelLoaderThread(QThread):
    finished = pyqtSignal(bool, str)
    log      = pyqtSignal(str, str)

    def __init__(self, engine: "LlamaEngine", model_path: str, ctx: int):
        super().__init__()
        self.engine     = engine
        self.model_path = model_path
        self.ctx        = ctx

    def run(self):
        ok = self.engine.load(self.model_path, ctx=self.ctx)
        self.finished.emit(ok, self.engine.status_text)


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
      f'{code("\'error\' in text.lower()")} — catch error messages<br>'
      f'{code("text.strip().startswith(\'```\'")} — detect code output<br>'
      f'{code("len(text.split()) < 20")} — detect very short answers<br>'
      f'{code("\'yes\' in text.lower()[:50]")} — check if model said yes'))}
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
    p(f'{code("\'long\' if len(text) > 400 else \'short\'")} — length-based routing<br>'
      f'{code("\'code\' if text.strip().startswith(\'```\') else \'prose\'")} — format detection<br>'
      f'{code("text.split(\':\')[0].strip().lower()")} — route on first word / prefix<br>'
      f'{code("\'positive\' if text.count(\'!\') > 2 else \'neutral\'")} — punctuation heuristic'))}

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
      f'{code("result = \'\\n\'.join(sorted(text.split(\'\\n\')))")} — sort lines<br>'
      f'{code("result = str(len(text.split())) + \' words: \' + text")} — prepend word count<br>'
      f'{code("import_count = metadata.get(\'runs\', 0) + 1; metadata[\'runs\'] = import_count; log(f\'Run #{import_count}\')")} — stateful counter'))}

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

# ═════════════════════════════ MAIN WINDOW ══════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Native Lab Pro")
        self.setMinimumSize(1100, 700)
        self.resize(1300, 840)

        self.engine   = LlamaEngine()
        self.sessions: Dict[str, Session] = {}
        self.active:   Optional[Session]  = None

        self._worker:          Optional[QThread] = None
        self._stream_w:        Any = None
        self._summary_worker:  Optional[QThread] = None
        self._summary_bubble:  Any = None
        self._pipeline_worker: Any = None
        self.reasoning_engine:     Any = None
        self.summarization_engine: Any = None
        self.coding_engine:        Any = None
        self.secondary_engine:     Any = None
        self._thinking_block:  Any = None
        self._pipeline_reason_w: Any = None
        self._pipeline_code_w:   Any = None
        self._pipeline_insight_widgets: list = []
        self._chat_pipeline_worker: Optional[QThread] = None
        self._api_engine:   Any = None

        self._force_coding_mode:  bool = False
        self._pending_ref_ctx:    str  = ""
        # ── busy / session-tracking ───────────────────────────────────────────
        self._busy_session_id:    str  = ""   # sid of the session currently generating
        self._stream_session_id:  str  = ""   # sid that owns the active stream worker
        self._stream_buffer:      str  = ""   # shadow text buffer; survives widget deletion
        self._summary_session_id: str  = ""   # sid that owns the active summary worker
        self._multi_pdf_worker:  Optional[QThread] = None
        self._pause_banner: Optional[QWidget] = None
        self._summarizing_active: bool = False
        self.current_ctx = DEFAULT_CTX()
        self._ctx_reload_timer = QTimer(self)
        self._ctx_reload_timer.setSingleShot(True)
        self._ctx_reload_timer.timeout.connect(self._apply_new_context)

        self._load_sessions()
        self._build_ui()
        self._build_menu()
        self._build_status_bar()
        # Load saved custom palettes before applying stylesheet
        global C_LIGHT, C_DARK, C, QSS
        set_theme(
            CURRENT_THEME,
            APP_CONFIG.get("custom_light_palette"),
            APP_CONFIG.get("custom_dark_palette"),
        )
        self.setStyleSheet(build_qss(C))
        self.appearance_tab.load_palette(C_LIGHT if CURRENT_THEME == "light" else C_DARK)

        if self.sessions:
            last = max(self.sessions.values(), key=lambda s: s.id)
            self._switch_session(last.id)
        else:
            self._new_session()

        # ── Restore saved theme preference ────────────────────────────────────
        _saved_theme = APP_CONFIG.get("theme", CURRENT_THEME)
        if _saved_theme != CURRENT_THEME:
            self._toggle_theme()   # silently apply saved theme
        self._update_theme_action_label()

        QTimer.singleShot(300, self._start_model_load)

        # Auto-load parallel engines if prefs say so
        if PARALLEL_PREFS.enabled and PARALLEL_PREFS.auto_load_roles:
            QTimer.singleShot(1000, self._auto_load_parallel_engines)

    def _auto_load_parallel_engines(self):
        """Load all engines whose roles are in the auto_load list."""
        for role in PARALLEL_PREFS.auto_load_roles:
            models = get_model_registry().all_models()
            for m in models:
                if m.get("role") == role and Path(m["path"]).exists():
                    self._start_role_engine_load(role, m["path"])
                    break

    def _start_role_engine_load(self, role: str, path: str):
        attr        = f"{role}_engine"
        loader_attr = f"_loader_{role}"

        # Cancel any in-progress loader for this role to avoid double-loading
        old_loader = getattr(self, loader_attr, None)
        if old_loader and old_loader.isRunning():
            self._log("WARN", f"Cancelling previous {role} loader before new one starts")
            try: old_loader.finished.disconnect()
            except Exception: pass
            old_loader.quit()
            old_loader.wait(2000)

        # Shutdown any existing engine for this role before creating a fresh one
        old_eng = getattr(self, attr, None)
        if old_eng and old_eng.is_loaded:
            self._log("INFO", f"Shutting down existing {role} engine before reload")
            old_eng.shutdown()

        # Always create a brand-new engine instance to avoid state leakage
        new_eng = LlamaEngine()
        setattr(self, attr, new_eng)

        cfg    = get_model_registry().get_config(path)
        loader = ModelLoaderThread(new_eng, path, cfg.ctx)
        loader.log.connect(self._log)
        loader.finished.connect(
            lambda ok, st, r=role, n=Path(path).name:
            self._on_role_engine_loaded(ok, st, r, n, None))
        loader.start()
        setattr(self, loader_attr, loader)

        # Give immediate visual feedback so the user sees something changed
        self._refresh_engine_status()
        self._log("INFO", f"Loading {role} engine: {Path(path).name}")

    # ── context management ────────────────────────────────────────────────────

    def _apply_new_context(self):
        if not self.engine.is_loaded:
            return
        new_ctx = self.ctx_slider.value()
        if new_ctx == getattr(self.engine, "ctx_value", DEFAULT_CTX()):
            return
        if new_ctx > 8192:
            ram_estimate = (new_ctx / 1024) * 0.5
            result = QMessageBox.question(
                self, "Confirm Context Reload",
                f"Changing context to {new_ctx:,} tokens requires restarting the model\n"
                f"and may use an additional ~{ram_estimate:.0f} MB of RAM.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if result != QMessageBox.StandardButton.Yes:
                loaded_ctx = getattr(self.engine, "ctx_value", DEFAULT_CTX())
                self.ctx_slider.blockSignals(True)
                self.ctx_slider.setValue(loaded_ctx)
                self.ctx_slider.blockSignals(False)
                self.ctx_input.setText(str(loaded_ctx))
                self.current_ctx = loaded_ctx
                return
        self._log("INFO", f"Reloading model with context {new_ctx}")
        if self._worker:
            if hasattr(self._worker, "abort"):
                self._worker.abort()
            self._worker.wait(1000)
            self._worker = None
        self.engine.shutdown()
        self.current_ctx = new_ctx
        self.ctx_bar.setRange(0, new_ctx)
        self._start_model_load()

    def _on_ctx_changed(self, value: int):
        self.ctx_input.setText(str(value))
        self.current_ctx = value
        if hasattr(self, "model_list"):
            try:
                item = self.model_list.currentItem()
                if item:
                    path = item.data(Qt.ItemDataRole.UserRole)
                    if path and path == getattr(self.engine, "model_path", ""):
                        self.cfg_ctx.blockSignals(True)
                        self.cfg_ctx.setText(str(value))
                        self.cfg_ctx.blockSignals(False)
            except RuntimeError:
                pass
        color = C["ok"]
        warn_text = ""
        if value > 24576:
            color = C["err"]; warn_text = "⚠"
            self.ctx_warn.setToolTip("Very high context.\nExpect heavy RAM usage.")
        elif value > 16384:
            color = C["warn"]; warn_text = "⚠"
            self.ctx_warn.setToolTip("High context.\nPerformance may degrade.")
        else:
            self.ctx_warn.setToolTip("")
        self._ctx_reload_timer.start(2000)
        self.ctx_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height:6px; background:{C['bg2']}; border-radius:3px;
            }}
            QSlider::handle:horizontal {{
                background:{color}; width:14px; margin:-4px 0; border-radius:7px;
            }}
            QSlider::sub-page:horizontal {{
                background:{color}; border-radius:3px;
            }}
        """)
        self.ctx_warn.setText(warn_text)

    def _on_ctx_input_changed(self):
        try:
            value = max(512, min(32768, int(self.ctx_input.text())))
            self.ctx_slider.setValue(value)
        except ValueError:
            self.ctx_input.setText(str(self.ctx_slider.value()))

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        ml = QHBoxLayout()
        ml.setContentsMargins(0, 0, 0, 0); ml.setSpacing(0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(2)

        self.sidebar = SessionSidebar()
        self.sidebar.session_selected.connect(self._switch_session)
        self.sidebar.new_session.connect(self._new_session)
        self.sidebar.session_deleted.connect(self._delete_session)
        self.sidebar.session_renamed.connect(self._rename_session)
        self.sidebar.session_exported.connect(
            lambda sid: self._export_session(sid, "md"))
        self.splitter.addWidget(self.sidebar)

        self.tabs = QTabWidget()
        self._tab_overlay = FadeOverlay(self.tabs)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # ── Chat tab (ChatModule) ──
        self.chat_module = ChatModule(
            session_id=self.active.id if self.active else "default")
        self.chat_area = self.chat_module.chat_area
        self.input_bar = self.chat_module.input_bar
        self.chat_module.send_requested.connect(self._on_send_with_refs)
        self.chat_module.stop_requested.connect(self._on_stop)
        self.chat_module.pdf_requested.connect(self._load_pdf)
        self.chat_module.clear_requested.connect(self._clear_chat)
        self.chat_module.multi_pdf_requested.connect(self._start_multi_pdf)
        self.input_bar.code_btn.toggled.connect(
            lambda chk: setattr(self, "_force_coding_mode", chk))
        self.input_bar.pipeline_run_requested.connect(self._on_pipeline_from_chat)
        self.tabs.addTab(self.chat_module, "💬  Chat")

        # ── Models tab ──
        self.models_tab = self._build_models_tab()
        self.tabs.addTab(self.models_tab, "🗂  Models")

        # ── Config tab ──
        self.config_tab = ConfigTab()
        self.config_tab.config_changed.connect(self._on_config_changed)
        self.config_tab.btn_resume_job.clicked.connect(self._resume_paused_job)
        self.tabs.addTab(self.config_tab, "⚙️  Config")

        # ── Server tab ──
        self.server_tab = ServerTab()
        self.server_tab.config_changed.connect(
            lambda: self._log("INFO", "Server config updated."))
        self.tabs.addTab(self.server_tab, "🖥️  Server")

        # ── Pipeline Builder tab ──
        self.pipeline_tab = PipelineBuilderTab(self.engine)
        self.tabs.addTab(self.pipeline_tab, "🔗  Pipeline")

        # ── API Models tab ──
        self.api_tab = ApiModelsTab()
        self.api_tab.api_model_loaded.connect(self._on_api_model_loaded)
        self.tabs.addTab(self.api_tab, "🌐  API Models")

        # ── Model Download tab ──
        self.download_tab = ModelDownloadTab()
        self.tabs.addTab(self.download_tab, "⬇️  Download")

        # ── MCP tab ──
        self.mcp_tab = McpTab()
        self.tabs.addTab(self.mcp_tab, "🔌  MCP")

        # ── Logs tab ──
        self.log_console = LogConsole()
        self.tabs.addTab(self.log_console, "🐞  Logs")
        self.appearance_tab = AppearanceTab()
        self.appearance_tab.theme_changed.connect(self._on_appearance_changed)
        self.tabs.addTab(self.appearance_tab, "🎨  Appearance")

        self.splitter.addWidget(self.tabs)
        self.splitter.setSizes([220, 1080])
        self.splitter.setStretchFactor(1, 1)
        ml.addWidget(self.splitter)
        central.setLayout(ml)
        self.setCentralWidget(central)

    # ── models tab ───────────────────────────────────────────────────────────

    def _build_models_tab(self) -> QWidget:
        outer = QWidget()
        outer_l = QVBoxLayout()
        outer_l.setContentsMargins(0, 0, 0, 0); outer_l.setSpacing(0)
        outer.setLayout(outer_l)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setObjectName("chat_scroll")

        w = QWidget()
        w.setObjectName("chat_container")
        root = QVBoxLayout()
        root.setContentsMargins(20, 18, 20, 18); root.setSpacing(0)
        w.setLayout(root); scroll.setWidget(w); outer_l.addWidget(scroll)

        def _section_label(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color:{C['txt']};font-size:12px;font-weight:bold;"
                              f"letter-spacing:0.5px;padding:0;margin-bottom:2px;")
            return lbl

        def _card(layout) -> QFrame:
            card = QFrame()
            card.setObjectName("tab_card")
            card.setLayout(layout); return card

        # ── header ───────────────────────────────────────────────────────────
        hdr = QLabel("🗂  GGUF Model Manager")
        hdr.setStyleSheet(f"color:{C['txt']};font-size:16px;font-weight:bold;margin-bottom:4px;")
        root.addWidget(hdr)
        note = QLabel("Add models, assign roles, and configure the reasoning→coding pipeline.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{C['txt2']};font-size:11px;margin-bottom:14px;")
        root.addWidget(note)

        # ── MODEL LIBRARY ─────────────────────────────────────────────────────
        root.addWidget(_section_label("MODEL LIBRARY"))
        list_card_l = QVBoxLayout()
        list_card_l.setContentsMargins(0, 0, 0, 0); list_card_l.setSpacing(0)

        # legend
        legend_row = QHBoxLayout()
        legend_row.setContentsMargins(10, 8, 10, 6); legend_row.setSpacing(14)
        for role, icon in ROLE_ICONS.items():
            pill = QLabel(f"{icon} {role.capitalize()}")
            pill.setStyleSheet(f"color:{C['txt2']};font-size:10px;"
                               f"background:{C['bg2']};border-radius:4px;padding:2px 6px;")
            legend_row.addWidget(pill)
        legend_row.addStretch()
        list_card_l.addLayout(legend_row)

        self.model_list = QListWidget()
        self.model_list.setObjectName("model_list")
        self.model_list.setMinimumHeight(150)
        self.model_list.setMaximumHeight(240)
        self.model_list.currentItemChanged.connect(self._on_model_list_select)
        list_card_l.addWidget(self.model_list)

        btn_strip = QHBoxLayout()
        btn_strip.setContentsMargins(10, 8, 10, 8); btn_strip.setSpacing(8)
        self.btn_browse_model = QPushButton("📂  Browse GGUF…")
        self.btn_load_primary = QPushButton("⚡  Load Selected")
        self.btn_load_primary.setObjectName("btn_send")
        self.btn_remove_model = QPushButton("🗑  Remove")
        self.btn_remove_model.setObjectName("btn_stop")
        for b in (self.btn_browse_model, self.btn_load_primary, self.btn_remove_model):
            b.setFixedHeight(30); btn_strip.addWidget(b)
        btn_strip.addStretch()
        list_card_l.addLayout(btn_strip)
        root.addWidget(_card(list_card_l))
        root.addSpacing(14)
        self.btn_browse_model.clicked.connect(self._browse_add_model)
        self.btn_load_primary.clicked.connect(self._load_selected_as_primary)
        self.btn_remove_model.clicked.connect(self._remove_selected_model)

        # ── PER-MODEL PARAMETERS ──────────────────────────────────────────────
        root.addWidget(_section_label("PER-MODEL PARAMETERS"))
        hint = QLabel("Select a model above to edit its parameters.")
        hint.setStyleSheet(f"color:{C['txt2']};font-size:10px;margin-bottom:6px;")
        root.addWidget(hint)

        cfg_card_l = QVBoxLayout()
        cfg_card_l.setContentsMargins(16, 14, 16, 14); cfg_card_l.setSpacing(10)
        LW = 110

        def _field_row(label_text: str, *widgets, stretch=True) -> QHBoxLayout:
            row = QHBoxLayout(); row.setSpacing(8)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(LW)
            lbl.setStyleSheet(f"color:{C['txt2']};font-size:12px;")
            row.addWidget(lbl)
            for ww in widgets: row.addWidget(ww)
            if stretch: row.addStretch()
            return row

        # Detected family banner (read-only, auto-set)
        self.cfg_family_lbl = QLabel("—")
        self.cfg_family_lbl.setStyleSheet(
            f"color:{C['acc2']};font-size:11px;"
            f"background:{C['bg2']};border-radius:4px;padding:3px 8px;"
        )
        cfg_card_l.addLayout(_field_row("Detected Family:", self.cfg_family_lbl))

        # Quant type banner
        self.cfg_quant_lbl = QLabel("—")
        self.cfg_quant_lbl.setStyleSheet(
            f"color:{C['ok']};font-size:11px;"
            f"background:{C['bg2']};border-radius:4px;padding:3px 8px;"
        )
        cfg_card_l.addLayout(_field_row("Quant Type:", self.cfg_quant_lbl))

        dv0 = QFrame(); dv0.setFrameShape(QFrame.Shape.HLine)
        dv0.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        cfg_card_l.addWidget(dv0)

        self.cfg_role = QComboBox()
        self.cfg_role.setMinimumWidth(200); self.cfg_role.setFixedHeight(28)
        for r in MODEL_ROLES:
            self.cfg_role.addItem(f"{ROLE_ICONS[r]}  {r.capitalize()}", r)
        cfg_card_l.addLayout(_field_row("Role:", self.cfg_role))

        dv1 = QFrame(); dv1.setFrameShape(QFrame.Shape.HLine)
        dv1.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        cfg_card_l.addWidget(dv1)

        self.cfg_threads = QLineEdit(str(DEFAULT_THREADS()))
        self.cfg_threads.setFixedWidth(64); self.cfg_threads.setFixedHeight(28)
        self.cfg_threads.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Threads:", self.cfg_threads))

        self.cfg_ctx = QLineEdit(str(DEFAULT_CTX()))
        self.cfg_ctx.setFixedWidth(80); self.cfg_ctx.setFixedHeight(28)
        self.cfg_ctx.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Context (tokens):", self.cfg_ctx))

        self.cfg_temp = QLineEdit("0.7")
        self.cfg_temp.setFixedWidth(64); self.cfg_temp.setFixedHeight(28)
        self.cfg_temp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Temperature:", self.cfg_temp))

        self.cfg_topp = QLineEdit("0.9")
        self.cfg_topp.setFixedWidth(64); self.cfg_topp.setFixedHeight(28)
        self.cfg_topp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Top-P:", self.cfg_topp))

        self.cfg_rep = QLineEdit("1.1")
        self.cfg_rep.setFixedWidth(64); self.cfg_rep.setFixedHeight(28)
        self.cfg_rep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Repeat Penalty:", self.cfg_rep))

        self.cfg_npred = QLineEdit(str(DEFAULT_N_PRED))
        self.cfg_npred.setFixedWidth(80); self.cfg_npred.setFixedHeight(28)
        self.cfg_npred.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Max Tokens:", self.cfg_npred))

        self.cfg_param_warn = QLabel("")
        self.cfg_param_warn.setWordWrap(True)
        self.cfg_param_warn.setStyleSheet(
            f"color:{C['warn']};font-size:11px;padding:4px 8px;"
            f"background:#2a2000;border-radius:5px;border:1px solid #5a4500;"
        )
        self.cfg_param_warn.setVisible(False)
        cfg_card_l.addWidget(self.cfg_param_warn)

        dv2 = QFrame(); dv2.setFrameShape(QFrame.Shape.HLine)
        dv2.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        cfg_card_l.addWidget(dv2)

        save_row = QHBoxLayout(); save_row.setSpacing(8)
        self.btn_save_cfg = QPushButton("💾  Save Parameters")
        self.btn_save_cfg.setFixedHeight(30)
        self.btn_save_cfg.clicked.connect(self._save_model_config)
        save_row.addWidget(self.btn_save_cfg)
        save_row.addStretch()
        cfg_card_l.addLayout(save_row)
        for wf in (self.cfg_ctx, self.cfg_threads, self.cfg_temp):
            wf.textChanged.connect(self._check_param_warnings)
        root.addWidget(_card(cfg_card_l))
        root.addSpacing(14)

        # ── ACTIVE ENGINES ────────────────────────────────────────────────────
        root.addWidget(_section_label("ACTIVE ENGINES"))
        eng_card_l = QVBoxLayout()
        eng_card_l.setContentsMargins(0, 0, 0, 0); eng_card_l.setSpacing(0)
        self.engine_status_list = QListWidget()
        self.engine_status_list.setFixedHeight(130)
        self.engine_status_list.setObjectName("engine_list")
        eng_card_l.addWidget(self.engine_status_list)
        eng_btn_strip = QHBoxLayout()
        eng_btn_strip.setContentsMargins(10, 8, 10, 8); eng_btn_strip.setSpacing(8)
        self.btn_load_role_engine = QPushButton("⚡  Load Engine for Role")
        self.btn_load_role_engine.setFixedHeight(30)
        self.btn_load_role_engine.clicked.connect(self._load_engine_for_selected)
        self.btn_unload_all = QPushButton("⏏  Unload All")
        self.btn_unload_all.setFixedHeight(30)
        self.btn_unload_all.clicked.connect(self._unload_all_engines)
        eng_btn_strip.addWidget(self.btn_load_role_engine)
        eng_btn_strip.addWidget(self.btn_unload_all)
        eng_btn_strip.addStretch()
        eng_card_l.addLayout(eng_btn_strip)
        root.addWidget(_card(eng_card_l))
        root.addSpacing(14)

        # ── PARALLEL LOADING ──────────────────────────────────────────────────
        root.addWidget(_section_label("PARALLEL LOADING & PIPELINE"))
        par_card_l = QVBoxLayout()
        par_card_l.setContentsMargins(0, 0, 0, 0)
        self.parallel_panel = ParallelLoadingDialog()
        self.parallel_panel.settings_changed.connect(self._on_parallel_settings_changed)
        par_card_l.addWidget(self.parallel_panel)
        root.addWidget(_card(par_card_l))
        root.addSpacing(14)

        # hidden compat stubs
        self.reasoning_status      = QLabel()
        self.summary_engine_status = QLabel()
        self.coding_engine_status  = QLabel()
        for lbl in (self.reasoning_status, self.summary_engine_status, self.coding_engine_status):
            lbl.setVisible(False); root.addWidget(lbl)

        root.addStretch()
        self._refresh_model_list()
        return outer

    def _on_parallel_settings_changed(self):
        pipeline_on = bool(
            PARALLEL_PREFS.enabled and
            PARALLEL_PREFS.pipeline_mode and
            self.reasoning_engine is not None and self.reasoning_engine.is_loaded and
            self.coding_engine is not None and self.coding_engine.is_loaded
        )
        self.input_bar.set_pipeline_mode(pipeline_on)
        self._log("INFO",
            f"Parallel settings: enabled={PARALLEL_PREFS.enabled}, "
            f"pipeline={PARALLEL_PREFS.pipeline_mode}, "
            f"auto_roles={PARALLEL_PREFS.auto_load_roles}")

    # ── model config helpers ──────────────────────────────────────────────────

    def _check_param_warnings(self):
        warnings = []
        try:
            ctx = int(self.cfg_ctx.text())
            if ctx > 24576:
                warnings.append(f"⚠  Context {ctx:,} tokens is very high")
            elif ctx > 16384:
                warnings.append(f"⚠  Context {ctx:,} tokens is high")
        except ValueError:
            pass
        try:
            threads = int(self.cfg_threads.text())
            import multiprocessing
            ncpu = multiprocessing.cpu_count()
            if threads > ncpu:
                warnings.append(f"⚠  {threads} threads exceeds {ncpu} logical CPUs")
        except (ValueError, NotImplementedError):
            pass
        try:
            temp = float(self.cfg_temp.text())
            if temp > 1.5:
                warnings.append("⚠  Temperature > 1.5")
            elif temp < 0.05:
                warnings.append("⚠  Temperature near 0")
        except ValueError:
            pass
        if warnings:
            self.cfg_param_warn.setText("\n".join(warnings))
            self.cfg_param_warn.setVisible(True)
        else:
            self.cfg_param_warn.setVisible(False)

    def _on_model_list_select(self, item: Optional[QListWidgetItem], _=None):
        if not item: return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path: return

        # Update family / quant labels
        fam   = detect_model_family(path)
        quant = detect_quant_type(path)
        ql, qcolor = quant_info(quant)
        self.cfg_family_lbl.setText(f"{fam.name}  (template: {fam.template})")
        self.cfg_quant_lbl.setText(f"{quant}  ·  {ql}")
        self.cfg_quant_lbl.setStyleSheet(
            f"color:{qcolor};font-size:11px;"
            f"background:{C['bg2']};border-radius:4px;padding:3px 8px;")

        cfg = get_model_registry().get_config(path)
        idx = self.cfg_role.findData(cfg.role)
        self.cfg_role.setCurrentIndex(max(idx, 0))
        self.cfg_threads.setText(str(cfg.threads))
        if path == getattr(self.engine, "model_path", "") and hasattr(self, "ctx_slider"):
            self.cfg_ctx.setText(str(self.ctx_slider.value()))
        else:
            self.cfg_ctx.setText(str(cfg.ctx))
        self.cfg_temp.setText(str(cfg.temperature))
        self.cfg_topp.setText(str(cfg.top_p))
        self.cfg_rep.setText(str(cfg.repeat_penalty))
        self.cfg_npred.setText(str(cfg.n_predict))
        self._check_param_warnings()

    def _save_model_config(self):
        item = self.model_list.currentItem()
        if not item:
            self._log("WARN", "No model selected"); return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path: return
        try:
            ctx = int(self.cfg_ctx.text()); threads = int(self.cfg_threads.text())
            temp = float(self.cfg_temp.text()); topp = float(self.cfg_topp.text())
            rep = float(self.cfg_rep.text()); npred = int(self.cfg_npred.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Parameter", "One or more fields are invalid."); return

        dangers = []
        if ctx > 24576:  dangers.append(f"Context = {ctx:,} tokens (very high)")
        if threads > 32: dangers.append(f"Threads = {threads}")
        if temp > 2.0:   dangers.append(f"Temperature = {temp}")
        if dangers:
            msg = "High-compute parameters:\n\n" + "\n".join(f"  • {d}" for d in dangers) + "\n\nSave?"
            if QMessageBox.warning(self, "⚠ Confirm", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) != QMessageBox.StandardButton.Yes:
                return

        fam = detect_model_family(path)
        cfg = ModelConfig(
            path=path, role=self.cfg_role.currentData() or "general",
            threads=threads, ctx=ctx, temperature=temp, top_p=topp,
            repeat_penalty=rep, n_predict=npred, family=fam.family,
        )
        get_model_registry().set_config(path, cfg)
        self._refresh_model_list()
        self._log("INFO", f"Saved config for {Path(path).name}: family={fam.name}, "
                          f"role={cfg.role}, ctx={cfg.ctx}")

        if path == getattr(self.engine, "model_path", ""):
            if ctx != self.ctx_slider.value():
                self.ctx_slider.blockSignals(True)
                self.ctx_slider.setValue(ctx)
                self.ctx_input.setText(str(ctx))
                self.ctx_slider.blockSignals(False)
                self.current_ctx = ctx
                self._ctx_reload_timer.start(500)

    def _load_engine_for_selected(self):
        item = self.model_list.currentItem()
        if not item: return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "File Not Found", f"Cannot find:\n{path}"); return
        cfg = get_model_registry().get_config(path)
        role = cfg.role

        if PARALLEL_PREFS.enabled and role != "general":
            # Warn if loading a second/third model
            n_loaded = sum(1 for r in ("reasoning","summarization","coding","secondary")
                           if getattr(self, f"{r}_engine", None) and
                           getattr(self, f"{r}_engine").is_loaded)
            if n_loaded >= 1:
                size_mb = ModelConfig(path=path).size_mb
                ram_est = max(size_mb * 1.1 / 1000, 1)
                QMessageBox.information(
                    self, "⚠️ Parallel RAM Usage",
                    f"Loading an additional engine (~{ram_est:.1f} GB).\n"
                    f"Total parallel engines after this: {n_loaded + 2}\n\n"
                    f"Ensure you have sufficient free RAM."
                )

        if role == "general":
            idx = self.input_bar.model_combo.findData(path)
            if idx == -1:
                self.input_bar.model_combo.addItem(Path(path).name, path)
                idx = self.input_bar.model_combo.findData(path)
            self.input_bar.model_combo.setCurrentIndex(idx)
            self.engine.shutdown()
            QTimer.singleShot(200, self._start_model_load)
        elif role in ("reasoning", "summarization", "coding", "secondary"):
            # Disable button while loading to prevent rapid re-clicks causing races
            self.btn_load_role_engine.setEnabled(False)
            self.btn_load_role_engine.setText("⏳  Loading…")

            attr        = f"{role}_engine"
            loader_attr = f"_loader_{role}"

            # Cancel any in-progress loader for this role
            old_loader = getattr(self, loader_attr, None)
            if old_loader and old_loader.isRunning():
                self._log("WARN", f"Cancelling stale {role} loader")
                try: old_loader.finished.disconnect()
                except Exception: pass
                old_loader.quit()
                old_loader.wait(2000)

            # Cleanly shut down any already-running engine for this role
            old_eng = getattr(self, attr, None)
            if old_eng and old_eng.is_loaded:
                self._log("INFO", f"Shutting down existing {role} engine")
                old_eng.shutdown()

            new_eng = LlamaEngine()
            setattr(self, attr, new_eng)
            cfg = get_model_registry().get_config(path)

            def _on_loaded_reenable(ok, st, r=role, n=Path(path).name):
                self._on_role_engine_loaded(ok, st, r, n, None)
                self.btn_load_role_engine.setEnabled(True)
                self.btn_load_role_engine.setText("⚡  Load Engine for Role")

            loader = ModelLoaderThread(new_eng, path, cfg.ctx)
            loader.log.connect(self._log)
            loader.finished.connect(_on_loaded_reenable)
            loader.start()
            setattr(self, loader_attr, loader)
            self._log("INFO", f"Loading {role} engine: {Path(path).name}")

        self._refresh_engine_status()

    def _on_role_engine_loaded(self, ok: bool, status: str, role: str,
                                name: str, lbl):
        color = C["ok"] if ok else C["err"]
        icon  = ROLE_ICONS.get(role, "🔌")
        text  = f"{icon} {role.capitalize()}:  {'✅  ' if ok else '❌  '}{name}"
        if lbl:
            try:
                lbl.setText(text)
                lbl.setStyleSheet(f"color:{color};font-size:11px;")
            except RuntimeError:
                pass
        self._log("INFO" if ok else "ERROR", f"{role} engine: {status}")
        self._refresh_engine_status()
        # Update pipeline badge in input bar
        self._on_parallel_settings_changed()

    def _refresh_engine_status(self):
        self.engine_status_list.clear()
        engines = {"General (primary)": self.engine}
        for role in ("reasoning", "summarization", "coding", "secondary"):
            eng = getattr(self, f"{role}_engine", None)
            if eng:
                engines[role.capitalize()] = eng
        for role_name, eng in engines.items():
            icon       = "🟢" if eng.is_loaded else "⚪"
            model_name = Path(eng.model_path).name if eng.model_path else "not loaded"
            fam_tag    = ""
            if eng.model_path:
                fam = detect_model_family(eng.model_path)
                qt  = detect_quant_type(eng.model_path)
                fam_tag = f"  [{fam.name} · {qt}]"
            mode_tag = f"  [{eng.mode}]" if eng.is_loaded else ""
            item = QListWidgetItem(
                f"  {icon}  {role_name:<22}  {model_name}{mode_tag}{fam_tag}")
            item.setForeground(QColor(C["ok"] if eng.is_loaded else C["txt2"]))
            self.engine_status_list.addItem(item)

    def _unload_all_engines(self):
        if QMessageBox.question(
            self, "Unload All Engines",
            "Unload all model engines?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        for role in ("reasoning", "summarization", "coding", "secondary"):
            eng = getattr(self, f"{role}_engine", None)
            if eng:
                eng.shutdown()
                setattr(self, f"{role}_engine", None)
        self.engine.shutdown()
        self._refresh_engine_status()
        self._on_parallel_settings_changed()
        self._log("INFO", "All engines unloaded.")

    def _refresh_model_list(self):
        self.model_list.clear()
        active = getattr(self.engine, "model_path", "")
        for m in get_model_registry().all_models():
            tag       = "📌" if m["source"] == "custom" else "📦"
            role_icon = ROLE_ICONS.get(m.get("role", "general"), "💬")
            ql, qc    = quant_info(m.get("quant", ""))
            label = (f"{tag}  {role_icon} [{m.get('role','general'):<14}]  "
                     f"{m['name']}   ({m['size_mb']} MB)  "
                     f"[{m.get('family','?')}·{m.get('quant','?')}·{ql}]")
            if m["path"] == active: label += "  ✅"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, m["path"])
            if m["path"] == active:
                item.setForeground(QColor(C["ok"]))
            self.model_list.addItem(item)
        if hasattr(self, "engine_status_list"):
            self._refresh_engine_status()

    def _sync_input_bar_combo(self):
        cur = self.input_bar.model_combo.currentData()
        self.input_bar.model_combo.blockSignals(True)
        self.input_bar.model_combo.clear()
        for m in get_model_registry().all_models():
            self.input_bar.model_combo.addItem(m["name"], m["path"])
        idx = self.input_bar.model_combo.findData(cur)
        self.input_bar.model_combo.setCurrentIndex(max(idx, 0))
        self.input_bar.model_combo.blockSignals(False)
        self.input_bar._update_family_badge()
        if hasattr(self, "model_list"):
            self._refresh_model_list()

    def _browse_add_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", str(Path.home()),
            "GGUF Models (*.gguf);;All Files (*)")
        if not path: return
        get_model_registry().add(path)
        fam   = detect_model_family(path)
        quant = detect_quant_type(path)
        ql, _ = quant_info(quant)
        self._refresh_model_list()
        self._sync_input_bar_combo()
        self._log("INFO",
            f"Added model: {Path(path).name}  →  {fam.name}  ·  {quant}  ·  {ql}")

    def _load_selected_as_primary(self):
        item = self.model_list.currentItem()
        if not item: return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "File Not Found", f"Cannot find:\n{path}"); return
        idx = self.input_bar.model_combo.findData(path)
        if idx == -1:
            self.input_bar.model_combo.addItem(Path(path).name, path)
            idx = self.input_bar.model_combo.findData(path)
        self.input_bar.model_combo.setCurrentIndex(idx)
        self.engine.shutdown()
        QTimer.singleShot(200, self._start_model_load)
        self._log("INFO", f"Loading primary model: {Path(path).name}")

    def _remove_selected_model(self):
        item = self.model_list.currentItem()
        if not item: return
        get_model_registry().remove(item.data(Qt.ItemDataRole.UserRole))
        self._refresh_model_list()
        self._sync_input_bar_combo()

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")
        fm.addAction(QAction("New Session\tCtrl+N", self, triggered=self._new_session))
        fm.addSeparator()
        xm = fm.addMenu("Export Current Session")
        xm.addAction(QAction("JSON",     self, triggered=lambda: self._export_active("json")))
        xm.addAction(QAction("Markdown", self, triggered=lambda: self._export_active("md")))
        xm.addAction(QAction("TXT",      self, triggered=lambda: self._export_active("txt")))
        fm.addSeparator()
        fm.addAction(QAction("Quit\tCtrl+Q", self, triggered=self.close))
        vm = mb.addMenu("View")
        vm.addAction(QAction("Toggle Sidebar\tCtrl+B", self, triggered=self._toggle_sidebar))
        vm.addAction(QAction("Go to Logs\tCtrl+L",    self, triggered=self._goto_logs))
        vm.addAction(QAction("Go to Models\tCtrl+M",  self, triggered=self._goto_models_tab))
        vm.addSeparator()
        # ── Theme toggle ──────────────────────────────────────────────────────
        self._theme_action = QAction("☀  Switch to Dark Theme", self)
        self._theme_action.setCheckable(False)
        self._theme_action.triggered.connect(self._toggle_theme)
        vm.addAction(self._theme_action)
        self._update_theme_action_label()
        mm = mb.addMenu("Model")
        mm.addAction(QAction("Reload Model", self, triggered=self._reload_model))

    def _build_status_bar(self):
        sb = self.statusBar()
        self.lbl_engine = QLabel("⚪  Loading…")
        self.lbl_engine.setStyleSheet(f"color:{C['txt2']};padding:0 8px;")
        sb.addWidget(self.lbl_engine)
        sb.addWidget(self._vline())

        # Family badge in status bar
        self.lbl_family = QLabel("")
        self.lbl_family.setStyleSheet(f"color:{C['acc2']};padding:0 6px;font-size:10px;")
        sb.addWidget(self.lbl_family)
        sb.addWidget(self._vline())

        sb.addWidget(QLabel("  Context:"))
        self.ctx_slider = QSlider(Qt.Orientation.Horizontal)
        self.ctx_slider.setRange(512, 32768)
        self.ctx_slider.setFixedWidth(140)
        self.ctx_slider.blockSignals(True)
        self.ctx_slider.setValue(DEFAULT_CTX())
        self.ctx_slider.blockSignals(False)
        self.ctx_slider.valueChanged.connect(self._on_ctx_changed)
        sb.addWidget(self.ctx_slider)

        self.ctx_input = QLineEdit(str(DEFAULT_CTX())) 
        self.ctx_input.setFixedWidth(60)
        self.ctx_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ctx_input.editingFinished.connect(self._on_ctx_input_changed)
        sb.addWidget(self.ctx_input)

        self.ctx_warn = QLabel("")
        self.ctx_warn.setFixedWidth(24)
        self.ctx_warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ctx_warn.setStyleSheet(f"color:{C['warn']};font-weight:bold;")
        sb.addWidget(self.ctx_warn)

        self.ctx_bar = QProgressBar()
        self.ctx_bar.setRange(0, DEFAULT_CTX())
        self.ctx_bar.setValue(0)
        self.ctx_bar.setFixedWidth(100)
        self.ctx_bar.setFixedHeight(8)
        self.ctx_bar.setTextVisible(False)
        sb.addWidget(self.ctx_bar)

        self.ctx_lbl = QLabel(f"0 / {DEFAULT_CTX()}")
        self.ctx_lbl.setStyleSheet(f"color:{C['txt2']};padding:0 8px;")
        sb.addWidget(self.ctx_lbl)

        sb.addPermanentWidget(self._vline())
        self.tps_lbl = QLabel("— tok/s")
        self.tps_lbl.setStyleSheet(f"color:{C['txt2']};padding:0 8px;")
        sb.addPermanentWidget(self.tps_lbl)

        if HAS_PSUTIL:
            sb.addPermanentWidget(self._vline())
            self.ram_lbl = QLabel("RAM: —")
            self.ram_lbl.setStyleSheet(f"color:{C['txt2']};padding:0 8px;")
            sb.addPermanentWidget(self.ram_lbl)
            self._ram_timer = QTimer(self)
            self._ram_timer.timeout.connect(self._update_ram)
            self._ram_timer.start(2500)

    @staticmethod
    def _vline() -> QFrame:
        f = QFrame(); f.setFrameShape(QFrame.Shape.VLine)
        f.setStyleSheet(f"color:{C['bdr']};"); return f

    # ── session management ────────────────────────────────────────────────────

    def _load_sessions(self):
        self.sessions = {}
        for p in SESSIONS_DIR.glob("*.json"):
            try:
                s = Session.load(p)
                self.sessions[s.id] = s
            except Exception as e:
                self._log("WARN", f"Skipping corrupt session {p.name}: {e}")

    def _refresh_sidebar(self):
        self.sidebar.refresh(
            self.sessions,
            self.active.id if self.active else "",
            busy_id=self._busy_session_id,
        )

    def _new_session(self):
        s = Session.new()
        self.sessions[s.id] = s
        s.save()
        self._switch_session(s.id)

    def _switch_session(self, sid: str):
        # Null the widget refs NOW so callbacks that fire after the switch
        # won't try to write to widgets that are about to be destroyed by
        # clear_messages().  Workers keep running; _stream_buffer / session
        # ids ensure output is still saved to the correct session on completion.
        self._stream_w         = None
        self._summary_bubble   = None
        self._pipeline_reason_w = None
        self._pipeline_code_w   = None

        # Only reset the generating UI indicator if we're not the busy session
        if sid != self._busy_session_id:
            self.input_bar.set_generating(False)

        s = self.sessions.get(sid)
        if not s: return
        self.active = s
        self.chat_area.clear_messages()
        for m in s.messages:
            self.chat_area.add_message(m.role, m.content, m.timestamp)
        self._refresh_sidebar()
        self.sidebar.set_active(sid)
        self._update_ctx_bar()
        # Sync ChatModule reference panel to this session
        if hasattr(self, "chat_module"):
            self.chat_module.set_session(sid)

    def _delete_session(self, sid: str):
        name = self.sessions[sid].title
        if QMessageBox.question(
            self, "Delete Session", f'Delete "{name}"?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        p = SESSIONS_DIR / f"{sid}.json"
        if p.exists():
            p.unlink()
        del self.sessions[sid]
        if self.active and self.active.id == sid:
            self.active = None
            self.chat_area.clear_messages()
        if self.sessions:
            last = max(self.sessions.values(), key=lambda s: s.id)
            self._switch_session(last.id)
        else:
            self._new_session()

    def _rename_session(self, sid: str, title: str):
        if sid in self.sessions:
            self.sessions[sid].title = title
            self.sessions[sid].save()
            self._refresh_sidebar()

    def _clear_chat(self):
        if not self.active: return
        if QMessageBox.question(
            self, "Clear Chat", "Clear all messages?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        self.active.messages.clear()
        self.active.save()
        self.chat_area.clear_messages()
        self._update_ctx_bar()

    # ── model loading ─────────────────────────────────────────────────────────

    def _start_model_load(self):
        model = self.input_bar.selected_model
        if not model or not Path(model).exists():
            model = str(MODELS_DIR / DEFAULT_MODEL)
        self.lbl_engine.setText("🔄  Loading model…")
        fam   = detect_model_family(model)
        quant = detect_quant_type(model)
        ql, _ = quant_info(quant)
        self.lbl_family.setText(f"{fam.name}  ·  {quant}  ·  {ql}")
        self._log("INFO", f"Loading model: {Path(model).name}  [{fam.name} / {quant}]")
        self._loader = ModelLoaderThread(self.engine, model, self.ctx_slider.value())
        self._loader.log.connect(self._log)
        self._loader.finished.connect(self._on_model_loaded)
        self._loader.start()

    def _on_model_loaded(self, ok: bool, status: str):
        self.ctx_slider.setEnabled(True)
        self.lbl_engine.setText(status)
        color = C["ok"] if ok else C["err"]
        self.lbl_engine.setStyleSheet(f"color:{color};padding:0 8px;")
        self._log("INFO" if ok else "ERROR", f"Model load: {status}")
        if hasattr(self, "model_list"):
            self._refresh_model_list()
        # Keep pipeline tab's engine reference up-to-date
        if hasattr(self, "pipeline_tab"):
            self.pipeline_tab.update_engine(self.engine)

    def _reload_model(self):
        self.engine.shutdown()
        QTimer.singleShot(500, self._start_model_load)

    # ── coding detection ──────────────────────────────────────────────────────

    _CODING_KEYWORDS = (
        "def ", "class ", "import ", "function ", "```", "debug ",
        "fix bug", "write code", "python ", "javascript", "typescript",
        "rust ", "c++", "c#", "golang", " sql", "regex", "script ",
        "implement ", "algorithm", "refactor", "syntax error",
        "code to ", "write a ", "create a ", "build a ", "generate code",
    )

    def _is_coding_prompt(self, text: str) -> bool:
        tl = text.lower()
        return self._force_coding_mode or any(k in tl for k in self._CODING_KEYWORDS)

    def _can_use_pipeline(self, text: str) -> bool:
        """Return True if pipeline mode should be used for this prompt."""
        return (
            PARALLEL_PREFS.enabled and
            PARALLEL_PREFS.pipeline_mode and
            self._is_coding_prompt(text) and
            self.reasoning_engine is not None and self.reasoning_engine.is_loaded and
            self.coding_engine is not None and self.coding_engine.is_loaded
        )

    def _active_engine_for(self, text: str) -> "LlamaEngine":
        if self._is_coding_prompt(text) and \
                self.coding_engine and self.coding_engine.is_loaded:
            return self.coding_engine
        # API engine takes priority over local engine when loaded
        if self._api_engine and self._api_engine.is_loaded:
            return self._api_engine
        return self.engine

    # ── tab fade ─────────────────────────────────────────────────────────────

    def _on_tab_changed(self, idx: int):
        w = self.tabs.widget(idx)
        if not w:
            return
        # Cover exactly the newly visible tab page
        self._tab_overlay.setGeometry(w.geometry())
        self._tab_overlay.raise_()
        self._tab_overlay.show()
        anim = QPropertyAnimation(self._tab_overlay, b"alpha", self)
        anim.setDuration(180)
        anim.setStartValue(220)
        anim.setEndValue(0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(self._tab_overlay.hide)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    # ── API engine ────────────────────────────────────────────────────────────

    def _on_api_model_loaded(self, api_engine: ApiEngine):
        """Called when ApiModelsTab successfully verifies an API model."""
        self._api_engine = api_engine
        cfg = api_engine._config
        name = f"{cfg.provider}  ·  {cfg.model_id}" if cfg else api_engine.status_text
        self.lbl_engine.setText(f"🌐  {name}")
        self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        self.lbl_family.setText(f"API  ·  max {cfg.max_tokens if cfg else '?'} tokens")
        self._log("INFO", f"API model loaded: {api_engine.status_text}")
        # Update pipeline tab to use api engine for pipeline blocks
        if hasattr(self, "pipeline_tab"):
            self.pipeline_tab.update_engine(api_engine)

    # ── pipeline-from-chat ────────────────────────────────────────────────────

    def _on_pipeline_from_chat(self):
        """Run a saved pipeline from the chat window. Output rendered as structured chat bubbles."""
        saved = list_saved_pipelines()
        if not saved:
            QMessageBox.information(
                self, "No Saved Pipelines",
                "No saved pipelines found.\n\n"
                "Build and save a pipeline in the 🔗 Pipeline tab first.")
            return

        pipeline_name, ok = QInputDialog.getItem(
            self, "Select Pipeline",
            "Choose a pipeline to run on your current input:",
            saved, 0, False)
        if not ok:
            return

        text = self.input_bar.input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Input",
                                "Type a message in the chat input first."); return

        try:
            blocks, conns = load_pipeline(pipeline_name)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e)); return

        if not self.engine.is_loaded:
            QMessageBox.warning(self, "Engine Not Ready",
                                "Wait for the model to finish loading."); return

        ts = datetime.now().strftime("%H:%M")
        self.chat_area.add_message("user", text, ts)
        self.input_bar.input.clear()

        self.chat_area.add_message(
            "assistant",
            f"🔗 **Running pipeline: {pipeline_name}**\n\n"
            f"_{len(blocks)} blocks · processing…_",
            ts)

        self._chat_pipeline_worker = PipelineExecutionWorker(
            blocks, conns, text, self.engine)
        self._chat_pipeline_worker.step_started.connect(
            lambda bid, lbl: self._chat_pipeline_step(lbl))
        self._chat_pipeline_worker.intermediate_live.connect(
            self._chat_pipeline_intermediate)
        self._chat_pipeline_worker.pipeline_done.connect(
            self._chat_pipeline_done)
        self._chat_pipeline_worker.err.connect(
            lambda msg: self.chat_area.add_message(
                "assistant",
                f"❌ Pipeline error:\n\n{msg}",
                datetime.now().strftime("%H:%M")))
        self._chat_pipeline_worker.log_msg.connect(
            lambda m: self._log("INFO", m))
        self._chat_pipeline_worker.start()

    def _chat_pipeline_step(self, label: str):
        ts = datetime.now().strftime("%H:%M")
        self.chat_area.add_message("system_note",
                                   f"⚡ Processing block: **{label}**…", ts)

    def _chat_pipeline_intermediate(self, bid: int, label: str, text: str):
        ts = datetime.now().strftime("%H:%M")
        self.chat_area.add_message(
            "pipeline_intermediate",
            f"◈ **{label}** — intermediate output\n\n{text}", ts)

    def _chat_pipeline_done(self, payload: str):
        import json as _json
        try:
            data   = _json.loads(payload)
            final  = data.get("text", payload)
            sender = data.get("sender", "")
        except Exception:
            final  = payload
            sender = ""
        ts     = datetime.now().strftime("%H:%M")
        header = f"■ **Output** _(from: {sender})_\n\n" if sender else "■ **Output**\n\n"
        self.chat_area.add_message("assistant", header + final, ts)
        self._chat_pipeline_worker = None

    # ── chat send ─────────────────────────────────────────────────────────────

    def _on_send_with_refs(self, text: str, ref_ctx: str):
        """Called by ChatModule — injects reference context into prompt."""
        self._pending_ref_ctx = ref_ctx
        self._on_send(text)

    def _on_send(self, text: str):
        if not self.active:    self._new_session()
        if not self.engine.is_loaded:
            self._log("WARN", "Model not yet loaded — please wait."); return
        # Block sends while ANY session is generating
        if self._busy_session_id:
            current_sid = self.active.id if self.active else ""
            if self._busy_session_id != current_sid:
                busy_sess = self.sessions.get(self._busy_session_id)
                busy_title = busy_sess.title if busy_sess else self._busy_session_id
                QMessageBox.warning(
                    self, "Processing in Progress",
                    f'Chat "{busy_title}" is currently generating a response.\n\n'
                    f"Please wait for it to finish before sending a new message.")
            else:
                QMessageBox.warning(
                    self, "Processing in Progress",
                    "This chat is already generating a response.\n"
                    "Please wait for it to finish.")
            return
        # Block new messages while summarization is active
        if getattr(self, "_summarizing_active", False):
            QMessageBox.warning(
                self, "Summarization Active",
                "A summarization job is in progress.\n\n"
                "Please pause or abort it from the chat panel before sending new messages.\n"
                "You can resume paused jobs from the Config tab.")
            return
        ts = datetime.now().strftime("%H:%M")
        if not self.active.messages:
            self.active.title = text[:40].replace("\n", " ")
            self.active.save()

        self.active.add_message("user", text)
        self.active.save()
        self.chat_area.add_message("user", text, ts)
        self._update_ctx_bar()

        # ── Pipeline mode ─────────────────────────────────────────────────────
        if self._can_use_pipeline(text):
            self._start_pipeline(text, ts)
            return

        # ── Normal / single-engine mode ───────────────────────────────────────
        active_eng = self._active_engine_for(text)
        is_coding  = active_eng is self.coding_engine
        eng_label  = "💻 Coding engine" if is_coding else active_eng.status_text

        ctx_chars = getattr(active_eng, "ctx_value", DEFAULT_CTX()) * 4
        prompt    = self.active.build_prompt(
            model_path=active_eng.model_path,
            max_chars=ctx_chars
        )
        # Inject reference context if available
        ref_ctx = getattr(self, "_pending_ref_ctx", "")
        if ref_ctx:
            fam = detect_model_family(active_eng.model_path)
            ref_block = (
                f"{fam.bos}{fam.user_prefix}"
                f"The following reference material is provided for context:\n\n"
                f"{ref_ctx}\n\n"
                f"Use this reference when answering the user's question."
                f"{fam.user_suffix}{fam.assistant_prefix}"
            )
            prompt = ref_block + "\n" + prompt
            self._pending_ref_ctx = ""

        cfg_pred = DEFAULT_N_PRED
        if active_eng.model_path:
            cfg_pred = get_model_registry().get_config(active_eng.model_path).n_predict

        self._stream_w = self.chat_area.add_message(
            "assistant", "", ts, tag="💻 Coding" if is_coding else "")
        self._log("INFO", f"Prompt ≈ {len(prompt)} chars · engine: {eng_label}")

        # For API engines, pass the full structured message history
        if isinstance(active_eng, ApiEngine):
            api_msgs = [{"role": m.role, "content": m.content}
                        for m in self.active.messages[-60:]]
            active_eng.set_messages(api_msgs)

        self._worker = active_eng.create_worker(
            prompt, n_predict=cfg_pred, model_path=active_eng.model_path)
        self._worker.token.connect(self._on_token)
        self._worker.done.connect(self._on_done)
        self._worker.err.connect(self._on_err)
        self._worker.start()

        self._stream_buffer     = ""
        self._stream_session_id = self.active.id if self.active else ""
        self._busy_session_id   = self._stream_session_id
        self._refresh_sidebar()

        self.input_bar.set_generating(True)
        lbl_txt = "💻 Coding…" if is_coding else "⚡  Generating…"
        self.lbl_engine.setText(lbl_txt)
        self.lbl_engine.setStyleSheet(f"color:{C['warn']};padding:0 8px;")

    # ── Pipeline mode orchestration ───────────────────────────────────────────

    def _collect_insight_engines(self):
        """Return list of (label, engine) for all active non-coding loaded engines."""
        candidates = [
            ("🧠 Reasoning",     self.reasoning_engine),
            ("📝 Summarization", self.summarization_engine),
            ("🔮 Secondary",     self.secondary_engine),
        ]
        # Also include primary engine if it's not the coding engine
        if self.engine and self.engine.is_loaded and self.engine is not self.coding_engine:
            candidates.append(("⚡ Primary", self.engine))

        return [
            (label, eng)
            for label, eng in candidates
            if eng is not None and eng.is_loaded and eng is not self.coding_engine
        ]

    def _start_pipeline(self, text: str, ts: str):
        # Ensure all engines are in server mode before starting pipeline
        # to avoid CLI prompt echo glitch
        engines_to_check = []
        if self.coding_engine and self.coding_engine.is_loaded:
            engines_to_check.append(("coding", self.coding_engine))
        for role in ("reasoning", "summarization", "secondary"):
            eng = getattr(self, f"{role}_engine", None)
            if eng and eng.is_loaded:
                engines_to_check.append((role, eng))

        for role, eng in engines_to_check:
            if eng.mode != "server":
                self._log("WARN", f"{role} engine in CLI mode — attempting server upgrade…")
                ok = eng.ensure_server_or_reload(log_cb=self._log)
                if not ok:
                    self._log("ERROR",
                        f"{role} engine could not start server mode — aborting pipeline")
                    self.chat_area.add_message(
                        "assistant",
                        f"⚠️ Pipeline aborted: **{role}** engine could not start in server mode.\n"
                        f"Try reloading the model from the Models tab.",
                        ts)
                    self.input_bar.set_generating(False)
                    return

        insight_engines = self._collect_insight_engines()

        if not insight_engines:
            # Fallback: no insight engines, just run coding engine directly
            self._log("WARN", "No insight engines available — running coding engine directly")
            active_eng = self.coding_engine
            ctx_chars  = getattr(active_eng, "ctx_value", DEFAULT_CTX()) * 4
            prompt     = self.active.build_prompt(model_path=active_eng.model_path, max_chars=ctx_chars)
            cfg_pred   = get_model_registry().get_config(active_eng.model_path).n_predict
            self._stream_w = self.chat_area.add_message("assistant", "", ts, tag="💻 Coding")
            self._worker = active_eng.create_worker(prompt, n_predict=cfg_pred, model_path=active_eng.model_path)
            self._worker.token.connect(self._on_token)
            self._worker.done.connect(self._on_done)
            self._worker.err.connect(self._on_err)
            self._worker.start()
            self.input_bar.set_generating(True)
            return

        n_engines = len(insight_engines)
        self._log("INFO", f"🔗 Pipeline: {n_engines} insight engine(s) → coding")
        self.lbl_engine.setText("🧠 Structural Insights…")
        self.lbl_engine.setStyleSheet(f"color:{C['pipeline']};padding:0 8px;")
        self.input_bar.set_generating(True)

        # Create one bubble per insight engine + divider + coding bubble
        self._pipeline_insight_widgets = []
        for label, eng in insight_engines:
            w = self.chat_area.add_message("assistant", "", ts, tag=label)
            self._pipeline_insight_widgets.append(w)

        self.chat_area.add_pipeline_divider(
            f"{n_engines} model(s) analysed → Coding model generating"
        )
        self._pipeline_reason_w = self._pipeline_insight_widgets[0] if self._pipeline_insight_widgets else None
        self._pipeline_code_w   = self.chat_area.add_message("assistant", "", ts, tag="💻 Coding")

        insight_np = max(
            (get_model_registry().get_config(eng.model_path).n_predict for _, eng in insight_engines),
            default=512
        )
        code_np = get_model_registry().get_config(self.coding_engine.model_path).n_predict

        self._pipeline_worker = PipelineWorker(
            insight_engines, self.coding_engine, text,
            n_predict_insight=insight_np, n_predict_code=max(code_np, 1024)
        )
        self._pipeline_worker.insight_started.connect(self._on_pipeline_insight_started)
        self._pipeline_worker.insight_token.connect(self._on_pipeline_insight_token)
        self._pipeline_worker.insight_done.connect(self._on_pipeline_insight_done)
        self._pipeline_worker.coding_token.connect(self._on_pipeline_code_token)
        self._pipeline_worker.coding_done.connect(self._on_pipeline_done)
        self._pipeline_worker.stage_changed.connect(self._on_pipeline_stage)
        self._pipeline_worker.err.connect(self._on_pipeline_err)
        self._pipeline_worker.start()
        self._busy_session_id = self.active.id if self.active else ""
        self._refresh_sidebar()

    def _on_pipeline_insight_started(self, idx: int, label: str):
        self.lbl_engine.setText(f"{label} analysing…")
        self.lbl_engine.setStyleSheet(f"color:{C['pipeline']};padding:0 8px;")

    def _on_pipeline_insight_token(self, idx: int, token: str):
        widgets = getattr(self, "_pipeline_insight_widgets", [])
        if idx < len(widgets) and widgets[idx]:
            try:
                widgets[idx].append_text(token)
            except RuntimeError:
                pass
        self.chat_area._scroll_bottom()

    def _on_pipeline_insight_done(self, idx: int, full_text: str):
        widgets = getattr(self, "_pipeline_insight_widgets", [])
        if idx < len(widgets) and widgets[idx]:
            try:
                widgets[idx].finalize()
            except RuntimeError:
                pass

    # keep old name for compat
    def _on_pipeline_reason_token(self, text: str):
        self._on_pipeline_insight_token(0, text)

    def _on_pipeline_reason_done(self, full_text: str):
        self._on_pipeline_insight_done(0, full_text)

    def _on_pipeline_stage(self, stage: str):
        if stage == "coding":
            self.lbl_engine.setText("💻 Coding…")
            self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        elif stage == "insights":
            self.lbl_engine.setText("🧠 Structural Insights…")
            self.lbl_engine.setStyleSheet(f"color:{C['pipeline']};padding:0 8px;")

    def _on_pipeline_code_token(self, text: str):
        if self._pipeline_code_w:
            try:
                self._pipeline_code_w.append_text(text)
            except RuntimeError:
                pass
        self.chat_area._scroll_bottom()

    def _on_pipeline_done(self, tps: float):
        self.tps_lbl.setText(f"{tps:.1f} tok/s")
        self.lbl_engine.setText(self.engine.status_text)
        self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        self.input_bar.set_generating(False)

        # Collect all insight texts for session save
        insight_parts = []
        for w in getattr(self, "_pipeline_insight_widgets", []):
            if w:
                try:
                    w.finalize()
                    t = w.full_text.strip()
                    if t:
                        insight_parts.append(t)
                except RuntimeError:
                    pass

        code_text = ""
        if self._pipeline_code_w:
            try:
                self._pipeline_code_w.finalize()
                code_text = self._pipeline_code_w.full_text.strip()
            except RuntimeError:
                pass

        if self.active and (insight_parts or code_text):
            insights_joined = "\n\n---\n\n".join(
                f"**[Structural Insight {i+1}]**\n\n{t}"
                for i, t in enumerate(insight_parts)
            )
            self.active.add_message("assistant",
                f"{insights_joined}\n\n---\n\n**[Code Output]**\n\n{code_text}")
            self.active.save()

        self._pipeline_insight_widgets = []
        self._pipeline_reason_w = None
        self._pipeline_code_w   = None
        self._pipeline_worker   = None
        self._busy_session_id   = ""
        self._update_ctx_bar()
        self._refresh_sidebar()

    def _on_pipeline_err(self, msg: str):
        self._log("ERROR", f"Pipeline error: {msg}")
        self.lbl_engine.setText("❌  Pipeline Error")
        self.lbl_engine.setStyleSheet(f"color:{C['err']};padding:0 8px;")
        self.input_bar.set_generating(False)
        if self._pipeline_code_w:
            try:
                self._pipeline_code_w.append_text(f"\n\n⚠️ Pipeline error: {msg}")
            except RuntimeError:
                pass
        self._pipeline_insight_widgets = []
        self._pipeline_reason_w = None
        self._pipeline_code_w   = None
        self._pipeline_worker   = None
        self._busy_session_id   = ""
        self._refresh_sidebar()

    # ── normal streaming handlers ──────────────────────────────────────────────

    def _on_token(self, text: str):
        self._stream_buffer += text          # shadow-buffer; always kept
        if not self._stream_w: return
        try:
            self._stream_w.append_text(text)
        except RuntimeError:
            self._stream_w = None            # widget gone; stop future writes
            return
        self.chat_area._scroll_bottom()

    def _on_done(self, tps: float):
        self.tps_lbl.setText(f"{tps:.1f} tok/s")
        self.lbl_engine.setText(self.engine.status_text)
        self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        self.input_bar.set_generating(False)
        if self._stream_w:
            try: self._stream_w.finalize()
            except RuntimeError: pass
        self._save_streamed()
        self._busy_session_id = ""
        self._refresh_sidebar()

    def _on_err(self, msg: str):
        self._log("ERROR", msg)
        self.lbl_engine.setText("❌  Error")
        self.lbl_engine.setStyleSheet(f"color:{C['err']};padding:0 8px;")
        self.input_bar.set_generating(False)
        if self._stream_w:
            try: self._stream_w.append_text(f"\n\n⚠️ Error: {msg}")
            except RuntimeError: pass
        if self._stream_w:
            try: self._stream_w.finalize()
            except RuntimeError: pass
        self._save_streamed()
        self._busy_session_id = ""
        self._refresh_sidebar()

    def _on_stop(self):
        if self._pipeline_worker:
            if hasattr(self._pipeline_worker, "abort"):
                self._pipeline_worker.abort()
            self._pipeline_worker.wait(2000)
            self._pipeline_worker = None
            for w in (self._pipeline_reason_w, self._pipeline_code_w):
                if w:
                    try: w.finalize()
                    except RuntimeError: pass
            self._pipeline_reason_w = None
            self._pipeline_code_w   = None

        if self._worker:
            if hasattr(self._worker, "abort"): self._worker.abort()
            self._worker.wait(2000)
            self._worker = None

        if self._summary_worker:
            if hasattr(self._summary_worker, "request_pause"):
                self._summary_worker.request_pause()
                self._summary_worker.wait(4000)
            else:
                if hasattr(self._summary_worker, "abort"):
                    self._summary_worker.abort()
                self._summary_worker.wait(2000)
            self._summary_worker = None
            if hasattr(self, "_summary_bubble") and self._summary_bubble:
                self._summary_bubble.append_text(
                    "\n\n⏸ Paused & saved to disk. Resume from the Config tab.")
                self._summary_bubble = None
            if hasattr(self, "config_tab"):
                self.config_tab.refresh_paused_jobs()

        if self._multi_pdf_worker:
            if hasattr(self._multi_pdf_worker, "request_pause"):
                self._multi_pdf_worker.request_pause()
                self._multi_pdf_worker.wait(4000)
            else:
                if hasattr(self._multi_pdf_worker, "abort"):
                    self._multi_pdf_worker.abort()
                self._multi_pdf_worker.wait(2000)
            self._multi_pdf_worker = None
            if hasattr(self, "_summary_bubble") and self._summary_bubble:
                self._summary_bubble.append_text(
                    "\n\n⏸ Multi-PDF paused & saved. Resume from the Config tab.")
                self._summary_bubble = None
            if hasattr(self, "config_tab"):
                self.config_tab.refresh_paused_jobs()

        if self._stream_w:
            try: self._stream_w.finalize()
            except RuntimeError: pass

        self.input_bar.set_generating(False)
        self.lbl_engine.setText(self.engine.status_text)
        self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        self._log("INFO", "Generation stopped by user.")
        self._save_streamed(suffix=" ✋")
        self._busy_session_id  = ""
        self._stream_session_id = ""
        self._stream_buffer     = ""
        self._summary_session_id = ""
        self._refresh_sidebar()

    def _save_streamed(self, suffix: str = ""):
        # Try to flush the live widget first; fall back to the shadow buffer
        content = ""
        if self._stream_w:
            try:
                self._stream_w._flush_timer.stop()
                self._stream_w._flush_pending()
                content = self._stream_w.full_text.strip()
            except RuntimeError:
                self._stream_w = None
        if not content:
            content = self._stream_buffer.strip()

        # Save to the session that STARTED the stream, not necessarily self.active
        save_sid = self._stream_session_id or (self.active.id if self.active else "")
        if content and save_sid:
            sess = self.sessions.get(save_sid)
            if sess:
                sess.add_message("assistant", content + suffix)
                sess.save()

        self._stream_w         = None
        self._worker           = None
        self._stream_buffer    = ""
        self._stream_session_id = ""
        self._update_ctx_bar()

    # ── PDF loading ────────────────────────────────────────────────────────────

    def _load_pdf(self):
        if not HAS_PDF:
            QMessageBox.warning(self, "Missing Dependency",
                                "Install PyPDF2:  pip install PyPDF2"); return
        if not self.engine.is_loaded:
            QMessageBox.warning(self, "Model Not Ready",
                                "Wait for the model to finish loading."); return

        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        if not path: return

        reader   = PdfReader(path)
        n_pages  = len(reader.pages)
        filename = Path(path).name
        self._log("INFO", f"Reading PDF: {filename}  ({n_pages} pages)")

        text = ""
        for i, page in enumerate(reader.pages):
            text += page.extract_text() or ""
            if i % 10 == 0:
                self._log("INFO", f"  …extracted page {i+1}/{n_pages}")

        if not text.strip():
            QMessageBox.warning(self, "Empty PDF", "No text could be extracted."); return

        ctx_chars   = getattr(self.engine, "ctx_value", DEFAULT_CTX()) * 3
        DIRECT_LIMIT = min(ctx_chars, 6000)

        if len(text) <= DIRECT_LIMIT:
            self.input_bar.input.setPlainText(
                f"The following is extracted from '{filename}':\n\n{text}"
                "\n\nPlease summarise the key points.")
            self._log("INFO", "Short document — loaded directly into prompt.")
            return

        if not self.active: self._new_session()
        self.active.title = f"Summary: {filename}"
        self.active.save()
        self._refresh_sidebar()

        ts = datetime.now().strftime("%H:%M")
        mode = self.input_bar.summary_mode
        mode_label = {"summary": "📋 Summary", "logical": "🔬 Logical Analysis", "advice": "💡 Advisory"}.get(mode, "📋 Summary")

        self._summary_bubble = self.chat_area.add_message(
            "assistant",
            f"📄  Starting **{mode_label}** of **{filename}** "
            f"({n_pages} pages, {len(text):,} chars)…\n", ts)

        self.input_bar.set_generating(True)
        self.input_bar.input.setEnabled(False)
        self.input_bar.input.setPlaceholderText(
            "⏸ Summarization in progress — pause or abort above before typing…")
        self._summarizing_active = True
        self.lbl_engine.setText("📄  Summarising…")
        self.lbl_engine.setStyleSheet(f"color:{C['acc']};padding:0 8px;")

        # Add pause banner to chat
        self._pause_banner = self.chat_area.add_pause_banner(
            on_pause_cb=self._on_pause_summary,
            on_abort_cb=self._on_stop
        )

        if self._summary_worker:
            if hasattr(self._summary_worker, "abort"): self._summary_worker.abort()
            self._summary_worker.wait(1000)

        from math import ceil
        estimated_chunks = ceil(len(text) / int(APP_CONFIG["summary_chunk_chars"]))
        self._thinking_block = self.chat_area.add_thinking_block(estimated_chunks)

        self._summary_worker = ChunkedSummaryWorker(
            self.engine, text, filename,
            engine2=getattr(self, "summarization_engine", None) or
                    getattr(self, "reasoning_engine", None),
            session_id=self.active.id if self.active else "",
            summary_mode=mode)
        self._summary_worker.progress.connect(self._on_summary_progress)
        self._summary_worker.section_done.connect(self._on_section_done)
        self._summary_worker.final_done.connect(self._on_summary_final)
        self._summary_worker.err.connect(self._on_summary_err_or_pause)
        self._summary_worker.pause_suggest.connect(self._on_pause_suggest)
        self._summary_worker.start()
        self._summary_session_id = self.active.id if self.active else ""
        self._busy_session_id    = self._summary_session_id
        self._refresh_sidebar()

    def _on_pause_summary(self):
        """Pause the active summary worker gracefully."""
        if self._summary_worker and hasattr(self._summary_worker, "request_pause"):
            self._summary_worker.request_pause()
            self.lbl_engine.setText("⏸  Pausing…")
            self.lbl_engine.setStyleSheet(f"color:{C['warn']};padding:0 8px;")
            if self._pause_banner:
                try:
                    self._pause_banner.status_lbl.setText("Pausing after current chunk…")
                except RuntimeError:
                    pass
        elif self._multi_pdf_worker and hasattr(self._multi_pdf_worker, "request_pause"):
            self._multi_pdf_worker.request_pause()
            self.lbl_engine.setText("⏸  Pausing…")
            self.lbl_engine.setStyleSheet(f"color:{C['warn']};padding:0 8px;")

    def _on_summary_progress(self, msg: str):
        self._log("INFO", msg)
        self.lbl_engine.setText(f"📄  {msg}")
        if "final consolidation" in msg.lower():
            try:
                tb = getattr(self, "_thinking_block", None)
                if tb is not None: tb.add_phase("Final consolidation pass")
            except RuntimeError:
                pass

    def _on_section_done(self, num: int, total: int, chunk_text: str, summary: str):
        self._log("INFO", f"Section {num}/{total} done ({len(summary)} chars)")
        try:
            tb = getattr(self, "_thinking_block", None)
            if tb is not None: tb.add_section(num, total, chunk_text, summary)
        except RuntimeError:
            pass
        try:
            bubble = getattr(self, "_summary_bubble", None)
            if bubble is not None:
                bubble.append_text(f"\n✅ Section {num}/{total} summarised.\n")
        except RuntimeError:
            pass
        self.chat_area._scroll_bottom()

    def _on_summary_final(self, final: str):
        self._log("INFO", f"Final summary ready ({len(final)} chars)")

        # Remove pause banner
        self.chat_area.remove_pause_banner()
        self._pause_banner = None
        self._summarizing_active = False
        self.input_bar.input.setEnabled(True)
        self.input_bar.input.setPlaceholderText(
            "Type a message…  (Enter = send · Shift+Enter = newline)")

        try:
            bubble = getattr(self, "_summary_bubble", None)
            if bubble is not None:
                bubble._flush_timer.stop()
                bubble._flush_pending()
                bubble.append_text("\n\n---\n**Final Summary:**\n\n" + final)
                QTimer.singleShot(80, bubble._flush_pending)
        except RuntimeError:
            pass
        finally:
            self._summary_bubble = None
        try:
            tb = getattr(self, "_thinking_block", None)
            if tb is not None: tb.mark_done()
        except RuntimeError:
            pass
        finally:
            self._thinking_block = None

        final_text = final
        save_sid     = self._summary_session_id or (self.active.id if self.active else "")
        def _persist():
            try:
                sess = self.sessions.get(save_sid)
                if sess:
                    sess.add_message("assistant", f"**Document Summary**\n\n{final_text}")
                    sess.save()
            except Exception as e:
                self._log("WARN", f"Could not save final summary: {e}")
            self._summary_worker    = None
            self._summary_session_id = ""
            self._busy_session_id   = ""
            self.input_bar.set_generating(False)
            self.lbl_engine.setText(self.engine.status_text)
            self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
            self._update_ctx_bar()
            self._refresh_sidebar()
        QTimer.singleShot(120, _persist)

    def _on_multi_pdf_ram_warning(self, msg: str):
        self._log("WARN", msg)
        bubble = getattr(self, "_summary_bubble", None)
        if bubble:
            try:
                bubble.append_text(f"\n{msg}\n")
            except RuntimeError:
                pass

    def _start_multi_pdf(self, paths: List[str]):
        """Start multi-PDF summarization with adaptive RAM watchdog."""
        if not HAS_PDF:
            QMessageBox.warning(self, "Missing Dep", "Install PyPDF2"); return
        if not self.engine.is_loaded:
            QMessageBox.warning(self, "Not Ready", "Wait for model to load."); return

        pdf_texts: List[Tuple[str, str]] = []
        for path in paths:
            try:
                reader = PdfReader(path)
                text   = "\n".join(p.extract_text() or "" for p in reader.pages)
                if text.strip():
                    pdf_texts.append((Path(path).name, text))
                    self._log("INFO", f"Loaded: {Path(path).name} ({len(text):,} chars)")
            except Exception as e:
                self._log("WARN", f"Could not read {path}: {e}")

        if not pdf_texts:
            QMessageBox.warning(self, "No Content", "No readable PDFs found."); return

        if not self.active: self._new_session()
        self.active.title = f"Multi-PDF: {len(pdf_texts)} docs"
        self.active.save()
        self._refresh_sidebar()

        ts = datetime.now().strftime("%H:%M")
        self._summary_bubble = self.chat_area.add_message(
            "assistant",
            f"📚  Starting multi-PDF summarization:\n"
            + "\n".join(f"  • {fn} ({len(t):,} chars)" for fn, t in pdf_texts)
            + f"\n\n⏳ Processing…\n", ts)

        from math import ceil
        total_chunks = sum(
            ceil(len(t) / int(APP_CONFIG["summary_chunk_chars"])) for _, t in pdf_texts)
        self._thinking_block = self.chat_area.add_thinking_block(total_chunks)

        self.input_bar.set_generating(True)
        self.lbl_engine.setText("📚  Multi-PDF…")
        self.lbl_engine.setStyleSheet(f"color:{C['acc']};padding:0 8px;")

        sid = self.active.id if self.active else "default"
        self._multi_pdf_worker = MultiPdfSummaryWorker(
            self.engine, pdf_texts, sid,
            engine2=getattr(self, "summarization_engine", None) or
                    getattr(self, "reasoning_engine", None))
        self._multi_pdf_worker.file_started.connect(
            lambda fi, fn, nc: self._log("INFO", f"PDF {fi+1}: {fn} ({nc} chunks)"))
        self._multi_pdf_worker.file_progress.connect(
            lambda fi, msg: self.lbl_engine.setText(f"📄 PDF {fi+1}: {msg}"))
        self._multi_pdf_worker.ram_warning.connect(self._on_multi_pdf_ram_warning)
        self._multi_pdf_worker.section_done.connect(self._on_section_done)
        self._multi_pdf_worker.file_done.connect(
            lambda fi, s: (
                self._log("INFO", f"File {fi+1} summary done ({len(s)} chars)"),
                self._summary_bubble and self._summary_bubble.append_text(
                    f"\n✅ Document {fi+1} summarised.\n")))
        self._multi_pdf_worker.progress.connect(self._on_summary_progress)
        self._multi_pdf_worker.final_done.connect(self._on_multi_pdf_final)
        self._multi_pdf_worker.err.connect(self._on_summary_err_or_pause)
        self._multi_pdf_worker.pause_suggest.connect(self._on_pause_suggest)
        self._multi_pdf_worker.start()

    def _on_multi_pdf_final(self, final: str):
        self._log("INFO", f"Multi-PDF final summary ready ({len(final)} chars)")
        try:
            bubble = getattr(self, "_summary_bubble", None)
            if bubble:
                bubble.append_text("\n\n---\n**Final Multi-Document Summary:**\n\n" + final)
                QTimer.singleShot(80, bubble._flush_pending)
        except RuntimeError:
            pass
        self._summary_bubble = None
        try:
            tb = getattr(self, "_thinking_block", None)
            if tb: tb.mark_done()
        except RuntimeError:
            pass
        self._thinking_block = None

        if self.active:
            self.active.add_message("assistant",
                f"**Multi-Document Summary**\n\n{final}")
            self.active.save()

        self._multi_pdf_worker = None
        self.input_bar.set_generating(False)
        self.lbl_engine.setText(self.engine.status_text)
        self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        self._update_ctx_bar()

    def _on_config_changed(self):
        self._log("INFO", "App config updated and saved.")

    def _resume_paused_job(self):
        state = self.config_tab.get_selected_job_state()
        if not state:
            QMessageBox.warning(self, "No Job Selected", "Select a paused job first.")
            return
        if not self.engine.is_loaded:
            QMessageBox.warning(self, "Not Ready", "Wait for model to load."); return

        raw_text = state.get("raw_text", "")        
        filename   = state.get("filename", "unknown")
        job_id     = state.get("job_id", "")
        session_id = state.get("session_id", "")
        pdf_texts  = state.get("pdf_texts", None)   # present for multi-PDF jobs

        # ── Multi-PDF resume ──────────────────────────────────────────────────
        if pdf_texts is not None:
            if not self.engine.is_loaded:
                QMessageBox.warning(self, "Not Ready", "Wait for model to load.")
                return
            if not self.active: self._new_session()
            self.active.title = f"Resume Multi-PDF: {filename}"
            self.active.save(); self._refresh_sidebar()
            ts = datetime.now().strftime("%H:%M")
            next_fi = state.get("next_fi", 0)
            next_ci = state.get("next_ci", 0)
            total   = state.get("total", len(pdf_texts))
            self._summary_bubble = self.chat_area.add_message(
                "assistant",
                f"▶  Resuming multi-PDF job from file {next_fi+1}/{total}, "
                f"chunk {next_ci+1}…\n", ts)
            from math import ceil
            est = sum(
                ceil(len(t) / int(APP_CONFIG["summary_chunk_chars"]))
                for _, t in pdf_texts[next_fi:])
            self._thinking_block = self.chat_area.add_thinking_block(max(est, 1))
            self.input_bar.set_generating(True)
            self.lbl_engine.setText("▶  Resuming Multi-PDF…")
            self.lbl_engine.setStyleSheet(f"color:{C['acc']};padding:0 8px;")
            sid = self.active.id if self.active else "default"
            self._multi_pdf_worker = MultiPdfSummaryWorker(
                self.engine,
                [(fn, txt) for fn, txt in pdf_texts],
                sid,
                engine2=getattr(self, "summarization_engine", None) or
                        getattr(self, "reasoning_engine", None),
                resume_job_id=job_id)
            self._multi_pdf_worker.file_started.connect(
                lambda fi, fn, nc: self._log("INFO", f"PDF {fi+1}: {fn} ({nc} chunks)"))
            self._multi_pdf_worker.file_progress.connect(
                lambda fi, msg: self.lbl_engine.setText(f"📄 PDF {fi+1}: {msg}"))
            self._multi_pdf_worker.ram_warning.connect(
                lambda msg: self._log("WARN", msg))
            self._multi_pdf_worker.section_done.connect(self._on_section_done)
            self._multi_pdf_worker.file_done.connect(
                lambda fi, s: self._log("INFO", f"File {fi+1} done ({len(s)} chars)"))
            self._multi_pdf_worker.progress.connect(self._on_summary_progress)
            self._multi_pdf_worker.final_done.connect(self._on_multi_pdf_final)
            self._multi_pdf_worker.err.connect(self._on_summary_err_or_pause)
            self._multi_pdf_worker.pause_suggest.connect(self._on_pause_suggest)
            self._multi_pdf_worker.start()
            self.config_tab.refresh_paused_jobs()
            self._log("INFO", f"Resumed multi-PDF job: {job_id}")
            return

        # ── Single-PDF resume (original path) ─────────────────────────────────
        if not raw_text:
            QMessageBox.warning(self, "No Data", "Paused job has no raw text saved.")
            return

        if not self.active: self._new_session()
        self.active.title = f"Resume: {filename}"
        self.active.save()
        self._refresh_sidebar()

        ts = datetime.now().strftime("%H:%M")
        next_chunk = state.get("next_chunk", 0)
        total      = state.get("total", "?")
        self._summary_bubble = self.chat_area.add_message(
            "assistant",
            f"▶  Resuming summarization of **{filename}** "
            f"from chunk {next_chunk}/{total}…\n", ts)

        from math import ceil
        est_remaining = (total - next_chunk) if isinstance(total, int) else 5
        self._thinking_block = self.chat_area.add_thinking_block(
            max(est_remaining, 1))

        self.input_bar.set_generating(True)
        self.lbl_engine.setText(f"▶  Resuming '{filename}'…")
        self.lbl_engine.setStyleSheet(f"color:{C['acc']};padding:0 8px;")

        self._summary_worker = ChunkedSummaryWorker(
            self.engine, raw_text, filename,
            engine2=getattr(self, "summarization_engine", None) or
                    getattr(self, "reasoning_engine", None),
            resume_job_id=job_id,
            session_id=session_id or (self.active.id if self.active else ""),
            summary_mode=state.get("summary_mode", "summary"))
        self._summary_worker.progress.connect(self._on_summary_progress)
        self._summary_worker.section_done.connect(self._on_section_done)
        self._summary_worker.final_done.connect(self._on_summary_final)
        self._summary_worker.err.connect(self._on_summary_err_or_pause)
        self._summary_worker.pause_suggest.connect(self._on_pause_suggest)
        self._summary_worker.start()
        self.config_tab.refresh_paused_jobs()
        self._log("INFO", f"Resumed paused job: {job_id}")

    def _on_pause_suggest(self, job_id: str):
        """Show pause banner in chat when auto-pause threshold is hit."""
        ts = datetime.now().strftime("%H:%M")
        banner = self.chat_area.add_message(
            "assistant",
            f"💡 **Pause available** — The summarization has processed "
            f"{APP_CONFIG['pause_after_chunks']} chunks. You can pause & save "
            f"state now to resume later, or continue processing.\n\n"
            f"Click ⏸ **Pause** in the stop button area to pause, "
            f"or ignore this to keep going.", ts)

    def _on_summary_err_or_pause(self, msg: str):
        if msg.startswith("__PAUSED__:"):
            job_id = msg.split(":", 1)[1]
            self._log("INFO", f"Job paused: {job_id}")
            self.chat_area.remove_pause_banner()
            self._pause_banner = None
            self._summarizing_active = False
            self.input_bar.input.setEnabled(True)
            self.input_bar.input.setPlaceholderText(
                "Type a message…  (Enter = send · Shift+Enter = newline)")
            self.input_bar.set_generating(False)
            self.lbl_engine.setText("⏸  Paused — state saved")
            self.lbl_engine.setStyleSheet(f"color:{C['warn']};padding:0 8px;")
            self._summary_worker = None
            if hasattr(self, "config_tab"):
                self.config_tab.refresh_paused_jobs()
            return
        self._on_summary_err(msg)

    def _on_summary_err(self, msg: str):
        self._log("ERROR", f"Summary pipeline error: {msg}")
        self.chat_area.remove_pause_banner()
        self._pause_banner = None
        self._summarizing_active = False
        self.input_bar.input.setEnabled(True)
        self.input_bar.input.setPlaceholderText(
            "Type a message…  (Enter = send · Shift+Enter = newline)")
        if hasattr(self, "_summary_bubble") and self._summary_bubble:
            self._summary_bubble.append_text(f"\n\n⚠️ Error: {msg}")
        self._summary_worker = None
        self._summary_bubble = None
        self.input_bar.set_generating(False)
        self.lbl_engine.setText("❌  Summary failed")
        self.lbl_engine.setStyleSheet(f"color:{C['err']};padding:0 8px;")

    # ── export ────────────────────────────────────────────────────────────────

    def _export_active(self, fmt: str):
        if self.active: self._export_session(self.active.id, fmt)

    def _export_session(self, sid: str, fmt: str):
        s = self.sessions.get(sid)
        if not s: return
        ext_map = {"json": "JSON (*.json)", "md": "Markdown (*.md)", "txt": "Plain Text (*.txt)"}
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Session", f"{s.id}.{fmt}", ext_map.get(fmt, "*"))
        if not path: return
        content = {"json": s.to_json, "md": s.to_markdown, "txt": s.to_txt}.get(fmt, s.to_json)()
        Path(path).write_text(content, encoding="utf-8")
        self._log("INFO", f"Exported to {path}")

    # ── status bar helpers ────────────────────────────────────────────────────

    def _update_ctx_bar(self):
        if not self.active: return
        est     = self.active.approx_tokens
        max_ctx = getattr(self.engine, "ctx_value", self.ctx_slider.value())
        self.ctx_bar.setRange(0, max_ctx)
        self.ctx_bar.setValue(min(est, max_ctx))
        self.ctx_lbl.setText(f"{est:,} / {max_ctx:,}")
        pct   = est / max_ctx if max_ctx > 0 else 0
        color = C["ok"] if pct < 0.6 else C["warn"] if pct < 0.85 else C["err"]
        self.ctx_bar.setStyleSheet(
            f"QProgressBar{{background:{C['bg2']};border:1px solid {C['bdr']};"
            f"border-radius:3px;height:8px;}}"
            f"QProgressBar::chunk{{background:{color};border-radius:3px;}}"
        )

    def _update_ram(self):
        mem  = psutil.virtual_memory()
        used = mem.used  // (1024 ** 3)
        tot  = mem.total // (1024 ** 3)
        self.ram_lbl.setText(f"RAM: {used}/{tot} GB")

    def _log(self, level: str, msg: str):
        if hasattr(self, "log_console"):
            self.log_console.log(level, msg)
        else:
            print(f"[{level}] {msg}", file=sys.stderr)

    def _toggle_sidebar(self):
        self.sidebar.setVisible(not self.sidebar.isVisible())

    # ── Theme switching ───────────────────────────────────────────────────────

    def _update_theme_action_label(self):
        if not hasattr(self, "_theme_action"):
            return
        if CURRENT_THEME == "light":
            self._theme_action.setText("🌙  Switch to Dark Theme")
        else:
            self._theme_action.setText("☀  Switch to Light Theme")

    def _on_appearance_changed(self, new_palette: dict):
        global C_LIGHT, C_DARK, C, QSS
        if CURRENT_THEME == "light":
            C_LIGHT = dict(new_palette)
            C = dict(C_LIGHT)
        else:
            C_DARK = dict(new_palette)
            C = dict(C_DARK)
        QSS = build_qss(C)
        self.setStyleSheet(QSS)
        if self.active:
            self._switch_session(self.active.id)

    def _toggle_theme(self):
        global CURRENT_THEME, C, QSS
        CURRENT_THEME = "dark" if CURRENT_THEME == "light" else "light"
        C   = C_LIGHT if CURRENT_THEME == "light" else C_DARK
        QSS = build_qss(C)
        self.setStyleSheet(QSS)
        APP_CONFIG["theme"] = CURRENT_THEME
        save_app_config(APP_CONFIG)
        self._update_theme_action_label()
        self._log("INFO", f"Theme switched to: {CURRENT_THEME}")

        # Rebuild Models tab (has baked dark card gradients)
        mt_idx = self.tabs.indexOf(self.models_tab)
        self.models_tab.setParent(None)
        self.models_tab.deleteLater()
        self.models_tab = self._build_models_tab()
        self.tabs.insertTab(mt_idx, self.models_tab, "🗂  Models")

        # Rebuild Config tab (has baked dark card backgrounds)
        ct_idx = self.tabs.indexOf(self.config_tab)
        self.config_tab.setParent(None)
        self.config_tab.deleteLater()
        self.config_tab = ConfigTab()
        self.config_tab.config_changed.connect(self._on_config_changed)
        self.config_tab.btn_resume_job.clicked.connect(self._resume_paused_job)
        self.tabs.insertTab(ct_idx, self.config_tab, "⚙️  Config")

        # Rebuild Server tab
        srv_idx = self.tabs.indexOf(self.server_tab)
        self.server_tab.setParent(None)
        self.server_tab.deleteLater()
        self.server_tab = ServerTab()
        self.server_tab.config_changed.connect(
            lambda: self._log("INFO", "Server config updated."))
        self.tabs.insertTab(srv_idx, self.server_tab, "🖥️  Server")

        # Rebuild Download tab
        dl_idx = self.tabs.indexOf(self.download_tab)
        self.download_tab.setParent(None)
        self.download_tab.deleteLater()
        self.download_tab = ModelDownloadTab()
        self.tabs.insertTab(dl_idx, self.download_tab, "⬇️  Download")

        # Rebuild MCP tab
        mcp_idx = self.tabs.indexOf(self.mcp_tab)
        self.mcp_tab.setParent(None)
        self.mcp_tab.deleteLater()
        self.mcp_tab = McpTab()
        self.tabs.insertTab(mcp_idx, self.mcp_tab, "🔌  MCP")

        # Rebuild Logs tab
        log_idx = self.tabs.indexOf(self.log_console)
        self.log_console.setParent(None)
        self.log_console.deleteLater()
        self.log_console = LogConsole()
        self.tabs.insertTab(log_idx, self.log_console, "🐞  Logs")

        # Refresh Appearance tab palette to match active theme
        self.appearance_tab.load_palette(C_LIGHT if CURRENT_THEME == "light" else C_DARK)

        # Rebuild chat bubbles + status bar colours
        self._on_model_loaded(self.engine.is_loaded, self.engine.status_text)
        if self.active:
            self._switch_session(self.active.id)

    def _goto_logs(self):
        self.tabs.setCurrentWidget(self.log_console)

    def _goto_models_tab(self):
        self.tabs.setCurrentWidget(self.models_tab)

    def closeEvent(self, event):
        self._log("INFO", "Shutdown — stopping engine…")
        if self._worker:
            try:
                if hasattr(self._worker, "abort"): self._worker.abort()
                self._worker.wait(1000)
            except Exception:
                pass
        if self._pipeline_worker:
            try:
                self._pipeline_worker.abort()
                self._pipeline_worker.wait(1000)
            except Exception:
                pass
        if hasattr(self, "pipeline_tab") and self.pipeline_tab._exec_worker:
            try:
                self.pipeline_tab._exec_worker.abort()
                self.pipeline_tab._exec_worker.wait(1000)
            except Exception:
                pass

        # Cancel any pending role loaders gracefully before shutting down engines
        for role in ("reasoning", "summarization", "coding", "secondary"):
            ldr = getattr(self, f"_loader_{role}", None)
            if ldr and ldr.isRunning():
                try: ldr.finished.disconnect()
                except Exception: pass
                ldr.quit()
                ldr.wait(1500)

        self.engine.shutdown()
        for role in ("reasoning", "summarization", "coding", "secondary"):
            eng = getattr(self, f"{role}_engine", None)
            if eng: eng.shutdown()

        # Final safety net: kill ANY remaining llama-server stragglers
        # (handles orphans from previous crashed sessions too)
        kill_stray_llama_servers(keep_pids=set())
        super().closeEvent(event)


# ═════════════════════════════ ENTRY POINT ══════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Native Lab Pro")
    app.setWindowIcon(QIcon('icon.png'))
    _fnt = QFont("Inter")
    if not _fnt.exactMatch():
        _fnt = QFont("Segoe UI")
    _fnt.setPointSize(10)
    app.setFont(_fnt)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    win = MainWindow()
    win.setWindowTitle("✦  Native Lab Pro  v2")
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()