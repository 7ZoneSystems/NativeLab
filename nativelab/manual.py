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
                         f'&#9888;&#65039;&nbsp; {t}</p>')
    def tip(t):  return (f'<p style="color:{OK};font-size:10.5px;margin:4px 0 10px;'
                         f'background:{BG2};border-left:3px solid {OK};'
                         f'padding:6px 10px;border-radius:0 5px 5px 0;">'
                         f'&#128161;&nbsp; {t}</p>')
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

    # ── pre-computed strings to avoid backslashes inside f-string expressions ──
    nl      = '\n'
    bs      = '\\'
    tick3   = '```'
    tick3s  = "```'"

    # code() snippets used inside p() calls — built outside f-strings
    c_len500         = code("len(text) &gt; 500")
    c_err            = code("'error' in text.lower()")
    c_startscode     = code("text.strip().startswith('```')")
    c_shortwords     = code("len(text.split()) &lt; 20")
    c_yescheck       = code("'yes' in text.lower()[:50]")
    c_longshort      = code("'long' if len(text) &gt; 400 else 'short'")
    c_codeorprose    = code("'code' if text.strip().startswith('```') else 'prose'")
    c_firstword      = code("text.split(':')[0].strip().lower()")
    c_punctheur      = code("'positive' if text.count('!') &gt; 2 else 'neutral'")
    c_upper          = code("result = text.upper()")
    _sortlines_inner = "result = '\\n'.join(sorted(text.split('\\n')))"
    c_sortlines      = code(_sortlines_inner)
    c_wordcount      = code("result = str(len(text.split())) + ' words: ' + text")
    _stateful_inner  = "import_count = metadata.get('runs', 0) + 1; metadata['runs'] = import_count; log(f'Run #' + str(import_count))"
    c_stateful       = code(_stateful_inner)
    c_score_lbl      = code("score")
    c_default        = code("default")
    c_concat         = code("concat")
    c_prepend_m      = code("prepend")
    c_append_m       = code("append")
    c_json_m         = code("json")
    c_prefix_t       = code("prefix")
    c_suffix_t       = code("suffix")
    c_replace_t      = code("replace")
    c_upper_t        = code("upper")
    c_lower_t        = code("lower")
    c_strip_t        = code("strip")
    c_truncate_t     = code("truncate")
    c_true_lbl       = code("TRUE")
    c_false_lbl      = code("FALSE")
    c_text_var       = code("text")
    c_result_var     = code("result")
    c_metadata_var   = code("metadata")
    c_log_fn         = code("log(msg)")
    c_general        = code("general")
    c_reasoning      = code("reasoning")
    c_summarization  = code("summarization")
    c_coding         = code("coding")
    c_pipelines_path = code("./localllm/pipelines/")
    c_pip_json       = code("./localllm/pipelines/name.json")
    c_pypdf2         = code("pip install PyPDF2")
    c_safe_builtins  = code("len str int float bool list dict tuple range enumerate zip map filter sorted min max sum abs round isinstance hasattr getattr repr type print")
    c_text_ro        = code("text")

    body = (
        f'<html><body style="background:{BG};color:{TXT};'
        'font-family:Inter,sans-serif;padding:20px 26px 30px;margin:0;line-height:1.5;">\n'

        + h1("&#128279;  NativeLab Pipeline Builder &#8212; Full Manual")
        + p(f'Version 2 &nbsp;&middot;&nbsp; All pipeline data stored in {c_pipelines_path} as JSON')
        + f'<hr style="border:none;border-top:1px solid {BG2};margin:14px 0 6px;">'

        + h2("&#128204;  Quick Start &#8212; 5 Steps to First Run")
        + p('1. <b>Add an &#9654; Input block</b> &#8212; sidebar left, section <i>Flow Blocks</i>.<br>'
            '2. <b>Add a &#9889; Model block</b> &#8212; drag a model from the sidebar list onto the canvas (ghost appears while hovering) or double-click it.<br>'
            '3. <b>Add an &#9632; Output block</b> &#8212; same sidebar section.<br>'
            '4. <b>Draw connections</b> &#8212; click a port dot on one block and drag to a port dot on another.<br>'
            '5. Type text in <i>Input text</i> on the right panel and press <b>&#9654; Run Pipeline</b>.')
        + tip("Start simple: Input &#8594; Model &#8594; Output. Add logic blocks only after you confirm the basic run works.")

        + h2("&#127912;  Canvas Controls")
        + p(kbd("Drag block") + ' Move any block freely &#8212; it snaps to the grid automatically.<br>'
            + kbd("Click port dot") + ' Start drawing a connection arrow.<br>'
            + kbd("Drag to port dot") + ' Complete a connection &#8212; creates a curved Bezier arrow.<br>'
            + kbd("Right-click block") + ' Context menu: Delete, Rename, Change Role, Configure.<br>'
            + kbd("Right-click canvas") + ' Clear all blocks and connections.<br>'
            + kbd("Double-click port dot") + ' Instantly delete <b>all arrows</b> attached to that port.<br>'
            + kbd("Right-click arrow") + ' Context menu &#8212; <b>Delete Arrow (A &#8594; B)</b> removes that single connection.<br>'
            + kbd("&#9654; Preview Flow") + ' Sidebar button &#8212; animates coloured dots flowing through the pipeline so you can verify routing before running. Click again or press &#9209; Stop Preview to cancel.<br>'
            + kbd("Pill bar") + ' The horizontal strip above the canvas shows all model/logic blocks as clickable pills &#8212; click one to select it on canvas.')
        + tip("The canvas is larger than the visible area. Use the scrollbars to pan around. Make big pipelines by spreading blocks far apart.")

        + h2("&#128309;  Port Dots  ( N &middot; S &middot; E &middot; W )")
        + p("Every block has four port dots sitting on its four edges &#8212; North (top), South (bottom), East (right), West (left). "
            "Click any dot and drag to any dot on another block to create an arrow. "
            "The arrow curves automatically and adapts its control handles to the port direction.")
        + port_table(
            ("E  &#8594;", "Any direction", "On branching blocks: arrow leaving from E = TRUE / LOW arm"),
            ("W  &#8592;", "Any direction", "On branching blocks: arrow leaving from W = FALSE / HIGH arm"),
            ("N  &#8593;", "Any direction", "On LLM SCORE: arrow leaving from N = raw score pass-through"),
            ("S  &#8595;", "Any direction", "On branching blocks: arrow leaving from S = MID arm / pass-through"),
          )
        + note("Execution follows <b>arrow direction</b>, not port name. The port a connection <b>leaves from</b> determines routing on branching blocks (IF/ELSE, SWITCH, LLM logic). "
               "For plain flow blocks (Input, Model, Intermediate, Output) all ports are equivalent &#8212; connect from whichever is convenient.")
        + tip("Use the &#9654; Preview Flow button after connecting your blocks to see exactly which path dots take &#8212; green = TRUE/pass, red = FALSE/drop, purple = SPLIT fan-out, yellow = MID/MERGE.")

        + h2("&#128246;  Flow Blocks")

        + h3(badge("&#9654; INPUT", OK) + " Input Block")
        + p("The mandatory starting point. The text you type in the <i>Input text</i> box on the right panel is injected here as the initial context. "
            "You only ever need one. Every pipeline must have exactly one Input block.")
        + port_table(
            ("E  &#8594;", "Output", "Sends the raw input text to the first connected block"),
            ("S  &#8595;", "Output", "Alternative output &#8212; use when fanning out to multiple first blocks"),
          )
        + example_box("EXAMPLE PIPELINE: Summarise + translate",
            p('&#9654; Input &#8594; &#9889; Summariser model &#8594; &#9672; Intermediate ("Now translate the above to French") &#8594; &#9889; Translator model &#8594; &#9632; Output'))

        + h3(badge("&#9672; INTERMEDIATE", WARN) + " Intermediate Block")
        + p("A pure prompt-injection node &#8212; no model is called. It takes the arriving context and wraps your custom instruction around it before passing it on. "
            "Useful for steering the next model without breaking the data flow.")
        + p('Right-click &#8594; <b>Configure block&#8230;</b> to set:<br>'
            '&nbsp;&nbsp;&bull; <b>Prompt position</b>: above (prompt &#8594; model output) or below (model output &#8594; prompt)<br>'
            '&nbsp;&nbsp;&bull; <b>Prompt text</b>: any instruction, e.g. <i>"Now rewrite the above more concisely."</i>')
        + port_table(
            ("W  &#8592;", "Input",  "Receives context from the previous block"),
            ("E  &#8594;", "Output", "Sends the wrapped context to the next block"),
          )
        + tip("During execution each Intermediate block gets its own live tab in the right panel output area &#8212; you can watch the injected context build up in real time.")
        + note("Leaving the prompt blank makes the Intermediate block a transparent pass-through &#8212; context flows through unchanged. Useful as a visual separator.")

        + h3(badge("&#9632; OUTPUT", ERR) + " Output Block")
        + p("The terminal node. Whatever context arrives here is shown in the <b>&#9632; Output</b> tab on the right panel. "
            "The pipeline stops when it reaches an Output block. You can have multiple Output blocks &#8212; each one terminates its own branch independently.")
        + port_table(
            ("W  &#8592;", "Input",  "Receives the final context &#8212; this becomes the displayed output"),
            ("N  &#8592;", "Input",  "Alternative input port for pipelines where the final arrow comes from above"),
          )

        + h2("&#9889;  Model Block")
        + p("Represents a loaded GGUF model. When the pipeline reaches a Model block, NativeLab starts the model in server mode (or reuses it if already running) and sends the current context as a prompt. The response becomes the new context.")
        + p('<b>How to add:</b><br>'
            '&nbsp;&nbsp;&bull; <b>Drag</b> a model from the sidebar list onto the canvas &#8212; a ghost block shows the drop position.<br>'
            '&nbsp;&nbsp;&bull; <b>Double-click</b> a model in the sidebar list to add it at a default position.')
        + p('<b>Right-click options:</b><br>'
            '&nbsp;&nbsp;&bull; <b>Change Role</b> &#8212; sets the system prompt used for this block:<br>'
            + '&nbsp;&nbsp;&nbsp;&nbsp;' + c_general + ' You are a helpful assistant.<br>'
            + '&nbsp;&nbsp;&nbsp;&nbsp;' + c_reasoning + ' Think step by step.<br>'
            + '&nbsp;&nbsp;&nbsp;&nbsp;' + c_summarization + ' Be clear and concise.<br>'
            + '&nbsp;&nbsp;&nbsp;&nbsp;' + c_coding + ' Write clean, well-commented code.<br>'
            + '&nbsp;&nbsp;&bull; <b>Rename block</b> &#8212; give it a human-readable name.<br>'
            + '&nbsp;&nbsp;&bull; <b>Delete</b> &#8212; removes the block and all its connections.')
        + port_table(
            ("W  &#8592;", "Input",  "Receives the context to use as the prompt"),
            ("E  &#8594;", "Output", "Sends the model response to the next block"),
            ("N  &#8592;", "Input",  "Alternate input &#8212; use when connecting from above"),
            ("S  &#8594;", "Output", "Alternate output &#8212; use when chaining downward"),
          )
        + note("Direct Model &#8594; Model connections are blocked. You must place an &#9672; Intermediate block between two models. "
               "This forces you to explicitly define what the second model should do with the first model's output.")
        + tip("Use different roles on different model blocks in the same pipeline. A 'reasoning' model can analyse, then a 'summarization' model can compress the result.")

        + h2("&#128206;  Context Blocks")
        + p("Context blocks inject fixed text into the pipeline without calling a model. They read configuration you set at design time, not at runtime.")

        + h3(badge("&#128206; REFERENCE", ACC) + " Reference Block")
        + p("Pastes a fixed reference text <b>before</b> the incoming context. Good for injecting background documents, company info, templates, or examples a model should work with.")
        + p('<b>Right-click &#8594; Configure block&#8230;</b> then choose:<br>'
            '&nbsp;&nbsp;&bull; <b>Type / paste text</b> &#8212; opens a multiline input dialog.<br>'
            '&nbsp;&nbsp;&bull; <b>Load from file</b> &#8212; reads any .txt, .md, .py, .json, .yaml file. '
            'Content is truncated to 4,000 characters with a [truncated] marker if longer.')
        + tip("Name your reference block descriptively (e.g. 'Company FAQ'). The injected text is wrapped in [REFERENCE: name] tags so the model can distinguish it from the user content.")

        + h3(badge("&#128161; KNOWLEDGE", ACC2) + " Knowledge Block")
        + p("Identical to Reference but labelled as a <i>knowledge base chunk</i> &#8212; the injected section is prefixed with <i>Knowledge Base:</i> instead of REFERENCE. "
            "Useful when you want to semantically distinguish instructional reference docs from factual knowledge.")
        + p('<b>Right-click &#8594; Configure block&#8230;</b> &#8212; opens a multiline text input. Truncated to 3,000 characters.')

        + h3(badge("&#128196; PDF SUMMARY", PL) + " PDF Summary Block")
        + p("Loads a PDF file, extracts text from all pages, and injects it. If the PDF exceeds 4,500 characters it is automatically chunk-summarised by the primary engine model before injection.")
        + p('<b>Right-click &#8594; Configure block&#8230;</b> to:<br>'
            '&nbsp;&nbsp;1. Select the PDF file.<br>'
            '&nbsp;&nbsp;2. Set the <b>role</b> of the PDF:<br>'
            '&nbsp;&nbsp;&nbsp;&nbsp;&bull; <b>reference</b> &#8212; prior context is MAIN, PDF is supporting reference.<br>'
            '&nbsp;&nbsp;&nbsp;&nbsp;&bull; <b>main</b> &#8212; PDF is the MAIN content, prior context is supporting reference.')
        + note('PyPDF2 must be installed: ' + c_pypdf2 + '. If it is missing the block will show a warning when you try to configure it.')

        + h2("&#9410;  Logic Blocks  ( Python conditions )")
        + p('Logic blocks evaluate Python expressions or perform text operations at runtime. '
            'The variable ' + c_text_var + ' always holds the incoming context string. '
            'These run instantly with no model call &#8212; use them for fast deterministic branching.')

        + h3(badge("&#9410; IF / ELSE", "#f59e0b") + " IF / ELSE")
        + p("Evaluates a Python boolean expression against the incoming text and routes to one of two output arms.")
        + port_table(
            ("E  &#8594;", "Output &#8212; TRUE branch",  "Followed when the condition evaluates to True / YES"),
            ("W  &#8594;", "Output &#8212; FALSE branch", "Followed when the condition evaluates to False / NO"),
            ("W  &#8592;", "Input",                        "Receives incoming context"),
          )
        + p('<b>Configure:</b> Right-click &#8594; Configure block&#8230; and type a Python expression.<br>'
            '<b>Draw two arrows</b> from this block: start from the <b>E port</b> for the TRUE arm and from the <b>W port</b> for the FALSE arm. '
            'Routing is determined by which port the arrow <i>leaves from</i> &#8212; no manual labelling needed.')
        + example_box("CONDITION EXAMPLES",
            p(c_len500 + ' &#8212; route long responses to a summariser<br>'
              + c_err + ' &#8212; catch error messages<br>'
              + c_startscode + ' &#8212; detect code output<br>'
              + c_shortwords + ' &#8212; detect very short answers<br>'
              + c_yescheck + ' &#8212; check if model said yes'))
        + note("The expression has access to: len, str, int, float, bool, list, dict, any, all, min, max, abs, isinstance. No file or network access.")

        + h3(badge("&#9411; SWITCH", "#f97316") + " SWITCH")
        + p("Evaluates a Python expression that returns a string key, then follows the outgoing arrow whose label matches that key.")
        + port_table(
            ("E/S/W  &#8594;", "Output arms", "One per case &#8212; label each arrow with the matching key string"),
            ("W  &#8592;",     "Input",       "Receives incoming context"),
          )
        + p('<b>Configure:</b> Right-click &#8594; Configure block&#8230; and type an expression that returns a string.<br>'
            'Each outgoing arrow is matched by the <b>port it leaves from</b> (E, W, S, N). '
            'To map a port to a named case, add a ' + c_default + ' entry in the block metadata under '
            + code("port_labels") + ' '
            '(e.g. ' + code("port_labels = {'E': 'positive', 'W': 'negative'}") + '). '
            'Without port_labels the port letter itself (E/W/S/N) is used as the key &#8212; make your expression return the port letter directly for the simplest setup.<br>'
            'Add an arm whose key is ' + c_default + ' to catch unmatched results.')
        + example_box("EXPRESSION EXAMPLES",
            p(c_longshort + ' &#8212; length-based routing<br>'
              + c_codeorprose + ' &#8212; format detection<br>'
              + c_firstword + ' &#8212; route on first word / prefix<br>'
              + c_punctheur + ' &#8212; punctuation heuristic'))

        + h3(badge("&#8856; FILTER", "#84cc16") + " FILTER")
        + p("A gate. If the condition is TRUE the text continues through the pipeline unchanged. If FALSE the pipeline terminates immediately with a [FILTER DROPPED] message.")
        + port_table(
            ("E  &#8594;", "Output &#8212; PASS", "Followed only when condition is True"),
            ("W  &#8592;", "Input",               "Receives incoming context"),
          )
        + example_box("USE CASES",
            p('&bull; Drop empty or whitespace-only responses before they reach the next model.<br>'
              '&bull; Stop the pipeline if a safety keyword is detected.<br>'
              '&bull; Gate on minimum length to avoid feeding junk to expensive models.'))
        + note("When FILTER drops text it emits a pipeline_done signal with the original text and a reason &#8212; this appears in the Output tab so you can debug why it was dropped.")

        + h3(badge("&#10551; TRANSFORM", "#06b6d4") + " TRANSFORM")
        + p("Deterministic text operation &#8212; no model involved, instant execution. Modifies the context in a fixed, predictable way.")
        + p('&nbsp;&nbsp;&bull; ' + c_prefix_t + ' &#8212; prepend fixed text before the context<br>'
            '&nbsp;&nbsp;&bull; ' + c_suffix_t + ' &#8212; append fixed text after the context<br>'
            '&nbsp;&nbsp;&bull; ' + c_replace_t + ' &#8212; find a substring and replace it<br>'
            '&nbsp;&nbsp;&bull; ' + c_upper_t + ' / ' + c_lower_t + ' &#8212; change case<br>'
            '&nbsp;&nbsp;&bull; ' + c_strip_t + ' &#8212; remove leading/trailing whitespace and blank lines<br>'
            '&nbsp;&nbsp;&bull; ' + c_truncate_t + ' &#8212; cut text to a maximum number of characters')
        + tip('Chain multiple TRANSFORM blocks together to build a text preprocessing pipeline before it hits a model &#8212; e.g. '
              + c_strip_t + ' &#8594; ' + c_truncate_t + ' &#8594; ' + c_prefix_t + '.')

        + h3(badge("&#8853; MERGE", "#8b5cf6") + " MERGE")
        + p("Waits for all incoming arrows to deliver their context, then joins them all into a single string that flows to the next block. "
            "Essential after a SPLIT or any fan-out pattern.")
        + p('&nbsp;&nbsp;&bull; ' + c_concat + ' &#8212; join with a separator string (default: two newlines + ---)<br>'
            '&nbsp;&nbsp;&bull; ' + c_prepend_m + ' &#8212; newest first, oldest last<br>'
            '&nbsp;&nbsp;&bull; ' + c_append_m + ' &#8212; oldest first, newest last<br>'
            '&nbsp;&nbsp;&bull; ' + c_json_m + ' &#8212; wrap all inputs as a JSON array string')
        + note("MERGE collects all contexts queued for its block ID in the current execution pass. If only one arrow arrives the block still works &#8212; it just returns that single input.")

        + h3(badge("&#9409; SPLIT", "#ec4899") + " SPLIT")
        + p("Broadcasts the exact same text to every single outgoing arrow simultaneously. Useful for running the same context through multiple models in parallel (they execute sequentially internally).")
        + port_table(
            ("E/S/W  &#8594;", "Output &times; N", "All outgoing arrows fire with identical text"),
            ("W  &#8592;",     "Input",            "Receives one context, fans it to all outputs"),
          )
        + tip("SPLIT + MERGE is the classic parallel-processing pattern: split into N model branches, each model processes the same input, MERGE collects all responses.")
        + example_box("PARALLEL REVIEW PATTERN",
            p('&#9654; Input &#8594; &#9409; SPLIT &#8594; &#9889; Reviewer A &#8594; &#8853; MERGE &#8594; &#9672; Intermediate ("Combine the two reviews") &#8594; &#9889; Model &#8594; &#9632; Output<br>'
              '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8600; &#9889; Reviewer B &#8599;'))

        + h3(badge("&#8997; CUSTOM CODE", "#10b981") + " Custom Code")
        + p("Write arbitrary Python that runs inline during pipeline execution. Full code editor with live syntax checking and a test runner.")
        + p('&nbsp;&nbsp;&bull; ' + c_text_ro + ' &#8212; incoming context string (read-only)<br>'
            '&nbsp;&nbsp;&bull; ' + c_result_var + ' &#8212; set this to your output string (defaults to ' + c_text_var + ' if not set)<br>'
            '&nbsp;&nbsp;&bull; ' + c_metadata_var + ' &#8212; block metadata dict, persists across pipeline runs<br>'
            '&nbsp;&nbsp;&bull; ' + c_log_fn + ' &#8212; writes a message to the Pipeline Log tab')
        + p('&nbsp;&nbsp;<b>Safe builtins available:</b> ' + c_safe_builtins)
        + note("No file I/O, no network access, no os/subprocess. The exec() is sandboxed. If your code raises an exception the pipeline stops with the error message shown in the Log tab.")
        + example_box("CODE EXAMPLES",
            p(c_upper + ' &#8212; uppercase<br>'
              + c_sortlines + ' &#8212; sort lines<br>'
              + c_wordcount + ' &#8212; prepend word count<br>'
              + c_stateful + ' &#8212; stateful counter'))

        + h2("&#129504;  LLM Logic Blocks  ( plain English conditions )")
        + p("LLM logic blocks work identically to their Python counterparts except the condition or instruction is written in <b>plain English</b> and the attached model evaluates it. "
            "The model is called with a tight system prompt that demands a specific answer format (YES/NO, a category name, PASS/STOP, etc.).")
        + p('<b>Each LLM logic block requires:</b><br>'
            '&nbsp;&nbsp;1. A <b>GGUF model</b> selected in the config dialog (can be any registered model).<br>'
            '&nbsp;&nbsp;2. A <b>plain English instruction</b> written by you.')
        + p('<b>Advanced settings</b> (expand in the config dialog):<br>'
            '&nbsp;&nbsp;&bull; <b>Max response tokens</b> &#8212; keep small (16&#8211;64) for routing, larger (128&#8211;512) for transforms.<br>'
            '&nbsp;&nbsp;&bull; <b>Temperature</b> &#8212; use 0&#8211;15 for deterministic routing, 30&#8211;60 for creative transforms.<br>'
            '&nbsp;&nbsp;&bull; <b>Show model reasoning in log</b> &#8212; logs the raw model response before it is parsed.<br>'
            '&nbsp;&nbsp;&bull; <b>Passthrough on error</b> &#8212; if the model call fails, pass text through unchanged instead of stopping.')
        + tip("Use a small fast model (e.g. Qwen2.5-0.5B or TinyLlama) for LLM routing blocks. They only need to say YES/NO &#8212; you do not need a 7B model for that.")

        + h3(badge("&#129504; LLM IF / ELSE", "#a855f7") + " LLM IF / ELSE")
        + p("The model reads your condition and the incoming text, then answers with a single word: YES or NO.")
        + port_table(
            ("E  &#8594;", "Output &#8212; TRUE / YES", "Connect from E port &#8212; followed when model answers YES"),
            ("W  &#8594;", "Output &#8212; FALSE / NO", "Connect from W port &#8212; followed when model answers NO"),
            ("N / S",      "Pass-through",              "Arrows from N or S are always followed regardless of answer"),
          )
        + example_box("CONDITION EXAMPLES",
            p('&bull; Does this text contain a complaint or expression of frustration?<br>'
              '&bull; Is the answer longer than a brief paragraph?<br>'
              '&bull; Does the user seem confused or are they asking a follow-up question?<br>'
              '&bull; Is this response in English?<br>'
              '&bull; Does this text contain any personally identifiable information?'))
        + note("The parser accepts: YES Y TRUE 1 PASS POSITIVE as truthy. Everything else is FALSE. Enable 'Show model reasoning' to debug unexpected routing.")

        + h3(badge("&#129504; LLM SWITCH", "#7c3aed") + " LLM SWITCH")
        + p("The model classifies the text into one of the categories you define. The category names are automatically read from the labels you put on the outgoing arrows.")
        + port_table(
            ("E/S/W  &#8594;", "Output arms", "One per category &#8212; label each arrow with the exact category name"),
          )
        + example_box("INSTRUCTION EXAMPLES",
            p('&bull; Classify this text as one of: positive, negative, neutral<br>'
              '&bull; What language is this written in? english, french, spanish, german, other<br>'
              '&bull; Is this a question, a complaint, a compliment, or a general statement?<br>'
              '&bull; Route by topic: technical, billing, account, other<br>'
              '&bull; What kind of content is this: code, prose, data, mixed?'))
        + tip("Always include a 'default' or 'other' labelled arrow to catch cases where the model returns something unexpected. The fallback prevents silent drops.")

        + h3(badge("&#129504; LLM FILTER", "#6366f1") + " LLM FILTER")
        + p("The model decides whether the text should continue (PASS) or be dropped (STOP). When dropped the pipeline ends with a structured message explaining which filter stopped it and the model's reason.")
        + example_box("CONDITION EXAMPLES",
            p('&bull; Only pass if this is a genuine technical support question<br>'
              '&bull; Pass only if the response contains a concrete action item<br>'
              '&bull; Allow through only if the content is safe and not harmful<br>'
              '&bull; Pass only if the answer is at least two sentences long<br>'
              '&bull; Block any response that mentions competitor products by name'))
        + note("The FILTER stop message is displayed in the Output tab with the filter name, condition, model decision, and the original text &#8212; so you can inspect exactly what was dropped and why.")

        + h3(badge("&#129504; LLM TRANSFORM", "#0ea5e9") + " LLM TRANSFORM")
        + p("The model rewrites, reformats, or transforms the incoming text according to your instruction. The result replaces the context for all downstream blocks. "
            "Increase Max response tokens to 256&#8211;512 for this block type.")
        + example_box("INSTRUCTION EXAMPLES",
            p('&bull; Summarise this in exactly three bullet points, each one sentence<br>'
              '&bull; Rewrite in a formal professional business tone<br>'
              '&bull; Extract only the action items as a numbered list<br>'
              '&bull; Translate to Spanish keeping all technical terms in English<br>'
              '&bull; Convert this prose into a structured JSON object with keys: title, summary, keywords<br>'
              '&bull; Remove all filler words and redundant phrases, keep only the core meaning'))
        + tip("The transform block automatically strips common preamble phrases the model might add (Here is..., Result:, Output:) before passing the result downstream.")

        + h3(badge("&#129504; LLM SCORE", "#d946ef") + " LLM SCORE")
        + p("The model rates the incoming text on your criterion from 1 to 10. The score is parsed from the response and used to route to one of three band arms.")
        + port_table(
            ("E  &#8594;", "Output &#8212; LOW (1&#8211;3)",   "Start arrow from E port &#8212; route to escalation or retry"),
            ("S  &#8595;", "Output &#8212; MID (4&#8211;7)",   "Start arrow from S port &#8212; route to standard processing"),
            ("W  &#8594;", "Output &#8212; HIGH (8&#8211;10)", "Start arrow from W port &#8212; route to fast-track or direct output"),
            ("N  &#8593;", "Raw score pass-through",           "Start arrow from N port &#8212; receives the numeric score as text"),
          )
        + p('The port the arrow <b>leaves from</b> determines which score band it handles. '
            'Connect from N port to receive the raw numeric score string in the downstream block &#8212; useful for feeding into a TRANSFORM or OUTPUT.')
        + example_box("CRITERION EXAMPLES",
            p('&bull; Rate the clarity and readability of this explanation (1=very unclear, 10=crystal clear)<br>'
              '&bull; Score the sentiment positivity (1=very negative, 10=very positive)<br>'
              '&bull; Rate the technical complexity (1=trivial, 10=expert-level)<br>'
              '&bull; How complete and thorough is this answer? (1=missing key info, 10=comprehensive)<br>'
              '&bull; Score the urgency of this message (1=low priority, 10=critical/immediate action needed)'))

        + h2("&#128260;  Loops")
        + p("Draw an arrow <b>backwards</b> &#8212; from a downstream block to an upstream block. NativeLab detects the cycle and asks how many times the loop should iterate (min 2, max 999). "
            "Loop arrows are shown as <b>dashed dash-dot lines</b> with a &times;N badge at the midpoint.")
        + p('<b>How execution works:</b><br>'
            'Each time the pipeline reaches the source block of a loop edge it checks how many times that specific edge has already been followed. '
            'Once the limit is reached the edge is skipped and execution continues to the next non-loop outgoing connection.')
        + example_box("REFINEMENT LOOP PATTERN",
            p('&#9654; Input &#8594; &#9889; Model &#8594; &#9672; Intermediate ("Critique the above and list improvements") &#8594; &#9889; Model<br>'
              '(draw a backwards arrow from the second Model back to the first Intermediate, set &times;3)<br>'
              '&#8594; &#9632; Output'))
        + tip("Connect a non-loop arrow from the loop body to an Output block to capture the final result after all iterations complete.")

        + h2("&#9654;  Preview Flow Animation")
        + p('Click <b>&#9654; Preview Flow</b> in the sidebar (below Load Pipeline) to animate coloured dots travelling through your pipeline along every connection arrow. '
            'The preview starts at the INPUT block and fans out to all reachable paths simultaneously &#8212; no model is called, no text is processed. '
            'It is a pure structural dry-run you can run at any time, even before the engine is loaded.')
        + p('<b>Dot colour legend:</b><br>'
            '&nbsp;&nbsp;&bull; <b style="color:#a6e3a1;">&#9679; Green</b> &#8212; normal flow / TRUE arm (E port)<br>'
            '&nbsp;&nbsp;&bull; <b style="color:#f38ba8;">&#9679; Red</b> &#8212; FALSE arm (W port)<br>'
            '&nbsp;&nbsp;&bull; <b style="color:#f9e2af;">&#9679; Yellow</b> &#8212; N/S arms, MERGE output<br>'
            '&nbsp;&nbsp;&bull; <b style="color:#cba6f7;">&#9679; Purple</b> &#8212; SPLIT fan-out broadcast<br>'
            '&nbsp;&nbsp;&bull; <b style="color:#666666;">&#9679; Grey</b> &#8212; FILTER / LLM FILTER drop path')
        + p('For branching blocks (IF/ELSE, SWITCH, LLM logic) the preview fires <b>all</b> outgoing arms at once to show every possible route data could take. '
            'This lets you verify the full graph structure before committing to a run.')
        + tip("Use Preview Flow immediately after wiring a new pipeline. If dots do not reach the Output block you have a disconnected path &#8212; a missing arrow will be obvious from where the dots stop.")

        + h2("&#128190;  Save &amp; Load Pipelines")
        + p('Click <b>&#128190; Save Pipeline&#8230;</b> in the sidebar. Type a name &#8212; existing names are overwritten without warning.<br>'
            'Click <b>&#128194; Load Pipeline&#8230;</b> to restore a saved pipeline. If the canvas has blocks you will be asked to confirm replacement.<br>'
            'Pipelines are stored as JSON in ' + c_pip_json + '.<br>'
            'To delete: Load dialog &#8594; select <i>&#128465; Delete a pipeline&#8230;</i> option.')
        + p('<b>What is saved per block:</b> type, position, size, model path, role, label, all metadata (prompt text, code, conditions, PDF path, port_labels map, etc.).<br>'
            '<b>What is saved per connection:</b> from/to block IDs, from/to port letters, is_loop flag, loop_times. Port letters drive all branch routing at runtime.')
        + note("Model files are saved as absolute paths. If you move a .gguf file the pipeline will load but the model block will show 'no valid file' and validation will fail until you re-attach the model.")

        + h2("&#128027;  Debugging &amp; Troubleshooting")
        + p('<b>&#128203; Log tab</b> &#8212; shows every step: block started, chars processed, decisions made, errors. Always check this first.<br>'
            '<b>&#9672; Intermediate tabs</b> &#8212; each intermediate block gets a live tab showing exactly what text arrived at it.<br>'
            '<b>&#9632; Output tab</b> &#8212; shows the final rendered output with markdown, code highlighting, and bold.')
        + p('<b>Common errors:</b>')
        + p('&bull; <i>No INPUT block</i> &#8212; add &#9654; Input and connect it to something.<br>'
            '&bull; <i>No OUTPUT block</i> &#8212; add &#9632; Output and connect the last block to it.<br>'
            '&bull; <i>No connections drawn</i> &#8212; you must draw at least one arrow between blocks.<br>'
            '&bull; <i>Model block has no valid file</i> &#8212; double-click a model in the sidebar or check the file still exists.<br>'
            '&bull; <i>Reference / Knowledge has no text</i> &#8212; right-click &#8594; Configure before running.<br>'
            '&bull; <i>IF/ELSE has no condition</i> &#8212; right-click &#8594; Configure and type a Python expression.<br>'
            '&bull; <i>LLM logic block: no model</i> &#8212; open config dialog and select a GGUF model.<br>'
            '&bull; <i>LLM logic block: model not found</i> &#8212; the model file was moved; reconfigure the block.<br>'
            '&bull; <i>Engine Not Ready</i> &#8212; wait for the primary model to finish loading in the Server tab.<br>'
            '&bull; <i>Server HTTP 500</i> &#8212; the model is loaded but errored; check the Logs tab in the Server tab for details.<br>'
            '&bull; <i>FILTER DROPPED</i> in output &#8212; the filter condition was FALSE; check the condition logic or increase tolerance.')
        + tip("Press &#9209; Stop Execution at any time. The pipeline will finish its current HTTP request and then halt cleanly.")

        + h2("&#9889;  Performance Tips")
        + p('&bull; Keep context short between models &#8212; truncate with a TRANSFORM block before expensive model calls.<br>'
            '&bull; Use small models (0.5B&#8211;1.5B) for all LLM logic routing blocks. Reserve big models for the actual generation steps.<br>'
            '&bull; PDF blocks auto-summarise large documents &#8212; but this adds extra model calls. Pre-summarise large PDFs offline if speed matters.<br>'
            '&bull; Loop iterations multiply model calls: 3 models &times; 5 loops = 15 server requests. Plan accordingly.<br>'
            '&bull; Set Max response tokens conservatively on routing blocks (16 is enough for YES/NO decisions).')

        + '\n</body></html>'
    )
    return body

PIPELINE_MANUAL_HTML = make_manual_html()