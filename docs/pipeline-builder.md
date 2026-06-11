# Pipeline Builder

NativeLab's pipeline builder is the visual workspace for building repeatable
LLM workflows. It supports manual node editing, shipped example presets, saved
JSON pipelines, native-accelerated graph helpers, and an AI-assisted builder
that can create or revise a pipeline from a plain-English request.

Open it from **Dev > Pipeline**. Developer Mode must be enabled in Settings if
the Dev tab is hidden.

---

## Layout

The builder has three panes:

| Area | Purpose |
| --- | --- |
| Left sidebar | Example presets, block buttons, model list, save/load/preview controls. |
| Canvas | Scrollable block graph editor. |
| Right sidebar | **Execution** tab and **AI Builder** tab. |

Both sidebars are resizable. If a sidebar is dragged too narrow, it snaps into a
thin side rail instead of collapsing to zero. Click the circular arrow on the
rail to reopen it. Text, buttons, and the AI Builder controls scale with the
current sidebar width.

---

## Example Presets

NativeLab ships pipeline examples in
`nativelab/pipelinebuilder/examples/`. They are packaged with the app and shown
in the **Example Presets** dropdown.

Current examples include:

- `quick-answer`
- `clean-summarize`
- `draft-review`
- `triage-router`
- `llm-classify-and-respond`
- `llm-quality-gate`
- `research-synthesis-fanout`
- `briefing-pack-builder`

Select a model in the model list before choosing a preset if you want placeholder
model blocks to be filled automatically. If no model is selected, the preset
still loads and leaves model placeholders for manual assignment.

When a preset or saved JSON has blocks outside the current visible canvas,
NativeLab expands the canvas automatically so the full graph remains reachable.

---

## Canvas Editing

### Basic actions

| Action | How |
| --- | --- |
| Add a block | Click a block button in the left sidebar. |
| Add a model block | Double-click or drag a model from the model list. |
| Move a block | Drag the block; it snaps to the 20 px grid on release. |
| Pan canvas | Click and hold on blank canvas, then drag. |
| Connect blocks | Drag from one port dot to another port dot. |
| Delete/configure | Right-click a block or connection. |
| Preview flow | Click **Preview Flow** after connecting blocks. |
| Save/load | Use **Save Pipeline...** and **Load Pipeline...**. |

### Safety rules

- Pipelines need at least one **Input** and one **Output** block.
- Direct model-to-model connections are blocked; use an Intermediate, Transform,
  Filter, Merge, or other logic block between model calls.
- Duplicate connections are ignored.
- Block IDs are normalized when needed so pasted, generated, or loaded JSON
  cannot accidentally reuse an existing block ID.
- Loop connections have explicit visit limits.
- Custom Code blocks run in a restricted namespace. Imports, filesystem access,
  networking, and subprocesses are not available.

---

## Block Types

### I/O and model

| Block | Use |
| --- | --- |
| Input | Required start point for user text. |
| Output | Required final render target. |
| Model | Runs a loaded local/API/Ollama/HF model. |
| Intermediate | Captures and streams mid-pipeline output. |

### Context

| Block | Use |
| --- | --- |
| Reference | Inject static reference text. |
| Knowledge | Inject reusable knowledge text. |
| PDF Summary | Load a PDF and summarize or inject it depending on settings. |

### Deterministic logic

| Block | Use |
| --- | --- |
| IF / ELSE | Safe Python expression on `text`; routes true/false. |
| SWITCH | Safe expression returns a key matched to labelled exits. |
| FILTER | Pass or stop the pipeline. |
| TRANSFORM | Prefix, suffix, replace, case change, strip, truncate, regex. |
| MERGE | Combine multiple upstream texts. |
| SPLIT | Fan out one input to multiple downstream branches. |
| Custom Code | Restricted deterministic Python block. |

### LLM logic

| Block | Use |
| --- | --- |
| LLM IF / ELSE | Plain-English yes/no routing. |
| LLM SWITCH | LLM classification into labelled exits. |
| LLM FILTER | PASS/STOP decision. |
| LLM TRANSFORM | Rewrite or reformat text. |
| LLM SCORE | Score 1-10 and route low/mid/high. |

Small models are usually best for LLM routing blocks because the expected output
is short and structured.

---

## Execution Tab

The **Execution** tab runs the current canvas through the same validation path
used by saved pipelines and CLI pipeline runs.

Before execution, NativeLab checks:

- Required Input/Output blocks.
- Valid model references for model-backed blocks.
- Configured context/PDF/logic metadata.
- Valid connection endpoints.
- Custom Code presence and syntax.
- LLM logic model attachment and instructions.

The execution log shows block starts, branch decisions, transforms, merges,
filter stops, split fan-out, loop visits, and final output. LLM/runtime errors
are routed through the centralized NativeLab LLM error dialog, so context-window
errors and server/API failures are shown in normal user language instead of only
being logged.

---

## AI Builder Tab

The **AI Builder** tab turns a plain-English request into NativeLab pipeline
JSON, validates it, saves it, and lets you load it for testing.

### Basic flow

1. Load a model first.
2. Open **Dev > Pipeline > AI Builder**.
3. Enter an output JSON name.
4. Describe the pipeline you want.
5. Click **Build & Save**.
6. Click **Load / Test** to place the generated pipeline on the canvas.

The active model receives a compact pipeline-building guide, the requested file
name, the selected/active model label, and your request. The response must be a
single JSON object. NativeLab extracts the JSON, normalizes it, attaches the
active model to empty model-backed blocks when possible, validates it, and saves
through the normal pipeline subsystem.

If the first model response is not valid JSON, NativeLab retries once with a
stricter JSON-only prompt and logs a preview of the invalid response.

### Context preflight

Before sending a request, the AI Builder estimates:

- Input tokens for the guide plus your request/context.
- Reserved output tokens for generated JSON.
- Total projected tokens versus the loaded model context limit.

If the request would exceed the current context window, NativeLab blocks the send
before the model call and asks you to increase context/reload the model or
shorten the request.

### Smart context

The AI Builder supports iterative editing:

- Empty history + empty canvas: your prompt is sent as-is.
- Empty history + existing canvas: NativeLab includes the current canvas JSON
  with the first prompt so the model can edit the graph instead of starting over.
- Existing history: NativeLab includes compact prior context, recent turns, and
  current canvas data.

Supported commands in the request box:

| Command | Effect |
| --- | --- |
| `/get_data` | Prints the current pipeline canvas state as JSON. |
| `/context` | Compacts AI Builder history using the active model when possible, with a deterministic local fallback. |

History is saved under `localllm/pipeline_builder_history/default.json`.

### Generated JSON rules

The AI Builder accepts the same persisted schema used by saved pipelines:

```json
{
  "version": 2,
  "title": "short name",
  "description": "short purpose",
  "blocks": [
    {
      "bid": 1,
      "btype": "input",
      "x": 80,
      "y": 120,
      "w": 148,
      "h": 76,
      "model_path": "",
      "role": "general",
      "label": "Input",
      "metadata": {}
    }
  ],
  "connections": [
    {
      "from_block_id": 1,
      "from_port": "E",
      "to_block_id": 2,
      "to_port": "W",
      "is_loop": false,
      "loop_times": 1
    }
  ]
}
```

Model-backed blocks can leave `model_path` empty. NativeLab fills them from the
selected model or active engine when possible.

---

## Native Pipeline Core

Python remains the UI/orchestration layer. Deterministic hot paths are routed
through focused native helpers when available:

- `nativelab/native/pipeline_core.c` handles block ID normalization, connection
  remapping, cycle checks, transform/merge helpers, route selection, loop visit
  limits, and validation records.
- `nativelab/native/pipeline_core.py` is the Python wrapper/fallback, so the app
  still works when the native extension is unavailable.
- `nativelab/pipelinebuilder/graph_ops.py` centralizes graph behavior.
- `nativelab/pipelinebuilder/execution_core.py` centralizes deterministic
  execution helpers.
- `nativelab/pipelinebuilder/validation.py` centralizes validation messages and
  keeps UI, CLI, and generated pipelines on the same guardrails.

AI Builder helper paths also have native C/Rust implementations for token
estimation and JSON-object span detection, with Python fallback behavior.

The native layer is optional. It improves deterministic pipeline work without
owning Qt widgets, model calls, subprocesses, plugin behavior, or user-facing
error dialogs.

---

## Troubleshooting

### AI Builder says the context is too small

Increase the model context limit and reload the model, or shorten the request.
Generated pipeline JSON reserves output tokens, so a request can exceed context
even when the visible prompt looks short.

### The model did not return JSON

NativeLab retries once automatically. If it still fails, make the request more
direct, for example:

```text
Make a 3 block input -> model -> output pipeline.
```

### Generated pipeline fails validation

The validator rejects unsafe or incomplete graphs. Check the error text, then
ask the AI Builder to revise the pipeline or edit the blocks manually.

### Sidebar content is hidden when resized

The sidebars should scale text and controls automatically. If the right sidebar
gets too tight, it switches tab labels to `Exec` and `AI`; dragging further snaps
the whole sidebar into a rail with a reopen arrow.
