<div align="center">

<img src="../nativelab/icon.png" alt="NativeLab" width="120" height="120" />

# NativeLab Documentation

</div>

Welcome. The docs are organised so each page covers a single concern - pick the one that matches what you're trying to do and skip the rest.

---

## 🚀 Getting started

| Page | When to read it |
|---|---|
| [installation.md](installation.md) | First-time install, llama.cpp setup, workspace folder layout. |
| [cli.md](cli.md) | You want to use the terminal client (`nativelab --cli`). |
| [../nativelab/cli/cli_guide.md](../nativelab/cli/cli_guide.md) | **Beginner walkthrough** of the CLI - friendly, step-by-step. |
| [troubleshooting.md](troubleshooting.md) | Something's broken and you want it fixed fast. |

---

## 🧭 What's in NativeLab

| Page | Topic |
|---|---|
| [features.md](features.md) | Full feature catalogue. Latest release notes live in `changelog.txt`. |
| [architecture.md](architecture.md) | Layered design, engine layer, project structure. |
| [labs.md](labs.md) | The Labs experimentation layer + how to add a feature. |
| [integrations.md](integrations.md) | External endpoint routes, local HTTP bridge, and Discord/WhatsApp bot connectors. |
| [models.md](models.md) | Model registry, family detection, quants, API models. |
| [workflows.md](workflows.md) | Pipelines, references, summarization, MCP, model/runtime downloads. |
| [ui.md](ui.md) | GUI components, theming, shortcuts, persistence. |

---

## 🗂️ Quick links by task

**"I want to chat with a local model from my terminal."**
→ [cli.md](cli.md) and the [CLI beginner guide](../nativelab/cli/cli_guide.md).

**"I want to build a multi-step pipeline with branches and loops."**
→ [workflows.md#visual-pipeline-builder](workflows.md#visual-pipeline-builder).

**"I want to feed long PDFs to a model."**
→ [workflows.md#summarization-pipeline](workflows.md#summarization-pipeline).

**"I want to plug in OpenAI / Anthropic / a local Ollama."**
→ [models.md#local-and-api-backend-support](models.md#local-and-api-backend-support).

**"I want to write a new experimental feature."**
→ [labs.md](labs.md).

**"I want to connect NativeLab to Discord, WhatsApp, webhooks, or a local script."**
→ [integrations.md](integrations.md).

**"I want to understand how the codebase is organised."**
→ [architecture.md](architecture.md).

---

## 📜 Other documents

- [LICENSE](../LICENSE) - AGPL v3.
- [CONTRIBUTING.md](../CONTRIBUTING.md) - How to send a PR.
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) - Community guidelines.
- [SECURITY.md](../SECURITY.md) - Reporting vulnerabilities.
