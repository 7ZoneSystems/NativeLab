# NativeLab CLI - Beginner's Guide

Talk to a language model that runs on **your own computer**, right from the
terminal. No browser, no cloud account, no monthly bill. This guide assumes
you've never used a terminal LLM tool before.

---

## What you'll need

- A computer with **Python 3.10+** installed.
- About **5 GB of free disk space** for a starter model.
- Roughly **8 GB of RAM** for the smallest models we recommend (more is
  better for quality and speed).
- A **terminal** (Terminal on macOS, Windows Terminal on Windows, or any
  Linux terminal - they all work).

You don't need to know any programming. The CLI walks you through everything
the first time.

---

## Step 1 - Open a terminal in the project folder

Open your terminal and `cd` into the NativeLab folder. For example:

```bash
cd ~/Desktop/NativeLabGithub/NativeLab
```

If `cd` isn't a familiar command yet, just remember: it stands for "change
directory," and you point it at wherever you cloned NativeLab.

---

## Step 2 - Launch the CLI

Type this and press **Enter**:

```bash
python -m nativelab --cli
```

The first time you run it, you'll see the NativeLab logo and a setup wizard.
Don't worry - the wizard handles everything for you.

---

## Step 3 - The setup wizard, explained

The wizard will ask you four things in plain English. Here's what each one
means and what's safe to pick:

### 1. "Binary check"

The CLI looks for two helper programs called **llama-server** and
**llama-cli** in `./llama/bin/`. These are what actually run the model.

- **If both show in green OK** - perfect, hit any answer to continue.
- **If they're red ERR** - open the GUI once with just `python -m nativelab`,
  click the **Download** tab, and let it install llama.cpp for you. Then come
  back here.

### 2. "Select a model"

A *model* is the language model file (typically a few GB). The CLI will
either show models you already have, or offer to download one.

If the list is empty, pick **"Download a new model from HuggingFace"**.

### 3. "Pick a model to download"

Three quick picks are pre-selected for you. If you're not sure, pick the
first one - **Mistral 7B Instruct (~4.4 GB)**. It's a great general-purpose
model.

The download will show a progress bar. If your internet drops, just re-run
the wizard later - it picks up where it left off.

### 4. "Context size"

Context is how much text (tokens) the model can "remember" at once. Bigger
numbers use more memory.

Safe defaults:

- **2048** - fastest, low RAM, fine for short questions.
- **4096** - good balance. Recommended for first-timers.
- **8192** - better for long documents, uses more RAM.

Just press **Enter** to accept the default if you're unsure.

When the wizard finishes, you'll see:

```
OK  Setup saved.
```

You're done with setup forever. Next time you run `nativelab --cli`, it opens
the terminal control center.

---

## Step 4 - Your first chat

After setup, choose **Chat** from the menu. You'll see a prompt that looks like
this:

```
you ▸
```

Type a question and press **Enter**:

```
you ▸ Explain what a Python list comprehension is, with one example.
bot ▸ A list comprehension is a compact way to build a list…
```

That's it. You're chatting with a language model running locally.

To leave, type:

```
you ▸ /quit
```

---

## Useful commands inside chat

Anything that **starts with a `/`** is a command, not a message to the model.

| Type this        | What it does                                              |
| ---------------- | --------------------------------------------------------- |
| `/help`          | Shows all the commands.                                   |
| `/status`        | Shows which model is loaded and how it's running.         |
| `/models`        | Lists local models.                                       |
| `/api-models`    | Lists saved API model profiles.                           |
| `/skills on`     | Enables shared skill instructions in chat.                |
| `/skills off`    | Disables shared skill instructions in chat.               |
| `/pipelines`     | Lists saved GUI pipelines.                                |
| `/labs`          | Lists available Labs features.                            |
| `/endpoint /snapshot` | Shows the integration endpoint catalog.              |
| `/reset`         | Clears the conversation memory and starts fresh.          |
| `/system You are a Python tutor.` | Sets a "personality" for the model.       |
| `/save chat.json` | Saves the whole conversation to a file.                  |
| `/load <path>`   | Switches to a different model file or API ref.            |
| `/ctx 8192`      | Changes the memory window size and reloads.               |
| `/quit`          | Exits.                                                    |

You don't need to memorize these. Just type `/help` any time.

---

## Embedding a file in your message

You can paste in the contents of a file by typing **`@`** followed by the
file path - no quotes, no spaces:

```
you ▸ Explain what @nativelab/labs/endpoints.py does in plain English.
```

The CLI silently reads the file and slips it into the prompt. The model sees
the actual code and gives you a real answer.

This works for any text file: `.py`, `.md`, `.txt`, `.json`, `.yaml`,
`.html`, etc. Files larger than ~60 KB are trimmed automatically so you don't
blow past the model's memory.

---

## Linting Python files

Linting checks Python files for mistakes. There are two ways to do it:

**As a one-shot command:**

```bash
python -m nativelab --cli lint my_script.py another.py
```

**Inside a chat:**

```
you ▸ /lint my_script.py
```

The CLI uses whichever linter you have installed: `pyflakes`, `flake8`, or
`pylint`. None installed? It still catches **syntax errors** with Python's
built-in compiler - that's the most important thing.

To install a linter:

```bash
pip install pyflakes        # smallest and fastest
```

---

## All the ways to start the CLI

```bash
# Setup-or-menu - the one you'll use most
python -m nativelab --cli

# Skip the wizard, jump straight to chat
python -m nativelab --cli chat

# Re-run the setup wizard (e.g. to switch models)
python -m nativelab --cli setup

# Force a clean setup, throwing away saved choices
python -m nativelab --cli setup --reset

# Lint files without entering chat
python -m nativelab --cli lint <file.py> ...

# Just show what's currently saved
python -m nativelab --cli status

# List models and API profiles
python -m nativelab --cli models list
python -m nativelab --cli api-models list

# Use Labs, pipelines, skills, and integrations from the terminal
python -m nativelab --cli skills list
python -m nativelab --cli labs list
python -m nativelab --cli pipeline list
python -m nativelab --cli endpoint /snapshot --json
```

You can also pass options to `chat`:

```bash
python -m nativelab --cli chat \
    --model ./localllm/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
    --ctx 4096 \
    --system "Be very concise."
```

---

## Where files are saved

The CLI keeps everything inside the project folder, so nothing is hidden in
strange system directories.

| File / folder                    | Why it's there                                |
| -------------------------------- | --------------------------------------------- |
| `./localllm/`                    | Where downloaded GGUF models live.            |
| `./localllm/cli_prefs.json`      | Your saved model + ctx choice.                |
| `./localllm/custom_models.json`  | Extra models you've added by hand.            |
| `./localllm/api_models.json`     | Saved API model profiles from the GUI/CLI.    |
| `./localllm/skill/skills.json`   | Shared model skills.                          |
| `./localllm/temp_code_edit.json` | Structured Code Edit Lab history.             |
| `./localllm/server_config.json`  | Paths to llama.cpp binaries (shared with GUI).|
| `./localllm/integrations/`       | Discord and WhatsApp bot profiles.            |
| `./llama/bin/`                   | The llama.cpp binaries themselves.            |
| `./sessions/`                    | Saved chat sessions from the GUI.             |

To start over from scratch, you can safely delete `./localllm/cli_prefs.json`
- the wizard will run again next launch.

---

## Common problems & fixes

**"command not found: python"**
Try `python3` instead, or install Python from <https://python.org/downloads>.

**"No module named nativelab"**
You're running the command from the wrong folder. `cd` into the folder that
contains the `NativeLab/nativelab/` directory, then try again.

**"No llama.cpp binary found."**
Open the GUI once (`python -m nativelab`), go to the Download tab, and click
**Download** under "llama.cpp release." It installs into `./llama/bin/`.
Re-launch the CLI afterwards.

**Download stalls or fails.**
Just re-run the wizard. It detects the half-finished `.part` file and resumes
from where it stopped.

**The model is slow.**
Lower the context size with `/ctx 2048` and pick a smaller model next time
(Qwen 2.5 3B or Phi-3.5 Mini are both excellent and fast).

**Replies are cut off.**
That's the model running out of "n_predict" tokens. The default is fine for
most questions; for long answers, just say "continue" as your next message.

---

## What about the GUI?

The CLI and the GUI use the **same** model, the **same** binaries, and the
**same** config. Switching between them is friction-free:

```bash
python -m nativelab          # GUI
python -m nativelab --cli    # CLI
```

Anything you set up in one shows up in the other.

---

## Where to go next

Once you're comfortable:

- Try `/system "You are a code reviewer."` and paste files with `@`.
- Try `/code-edit my_file.py -- add input validation` for structured edits.
- Use `python -m nativelab --cli serve --port 8765` when testing Discord,
  WhatsApp, or other external integrations.
- Open `nativelab/cli/cli_guide.md` (this file) any time - it's bundled with
  the project so it travels with the code.

Have fun. Local LLMs are quietly amazing once you get the hang of them.
