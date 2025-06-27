# 🧠 Robust Annotator Framework

## Overview

**Robust Annotator** is a modular framework designed to automate **data engineering tasks** using **Separated Mnolithic agents**, each dedicated to a single, isolated processing task.

Agents currently operate using either:
- Remote LLMs via API (e.g., OpenAI, Gemini, Claude)
- Local quantized models (e.g., C++ backends via `llama.cpp` with `gguf`)

Future extensions will enable integration with **enterprise endpoints** through HTTP/REST interfaces.

---

## 🗂️ Project Structure

Each subfolder in the project corresponds to an **independent monolithic agent**, specialized for one well-defined task.

| Folder                   | Agent Name              | Description                                                                 |
|--------------------------|--------------------------|-----------------------------------------------------------------------------|
| `langgraph_agents/NER_annotator_agent`   | NER Annotation Agent     | Annotates named entities in raw texts 
| `langgraph_agents/SPAN_annotator_agent`  | Span Detection Agent     | Identifies relevant spans (e.g., disinformation signals) in documents.     |
| `langgraph_agents/DataAndLabel_linker_agent`   | Linker Agent             | Aligns noisy raw data (e.g., OCR texts) with pre-annotated labels to produce a clean dataset. |

---

## 🧩 Agent Details

### 1. NER Annotation Agent

- Task: Extract named entities from raw text using LLM-based classification.
- Output: Token-labeled in json format.
- Usage: Supports custom prompts and multiple languages.

### 2. Span Detection Agent

- Task: Identify semantically significant spans descibed in prompts (e.g., rhetorical markers, disinformation indicators).
- Input: Raw unstructured documents.
- Output: Annotated span intervals with optional reasoning metadata, and also a refinement strategy loop.

### 3. Linking Agent

The agents operate on the following data transformation pattern:

**( X, Y ) → D( X′, Y′ )**

- **X**: Input raw data, often noisy or unstructured (e.g., OCR-extracted text).
- **Y**: Manually curated labels, following a defined schema.
- **D(X′, Y′)**: Cleaned, restructured, and prompt-aligned dataset, ready for analysis or training.

This pipeline enables:
- Creation of high-quality structured datasets
- Training of domain-specific classifiers
- Downstream analytics or integration into ML pipelines
---

## ⚙️ Current Capabilities

- ✅ Compatible with API-based LLMs (configurable via YAML)
- ✅ Supports local execution via quantized models (GGUF + C++)
- ✅ Modular design per agent
- ✅ Error handling and retry policies
- ✅ Prompt-level control and auditability

---

## 🔜 Planned Extensions

- Integration with authenticated enterprise REST endpoints for input/output
- Workflow orchestration for chaining multiple agents
- Use this suite as starting of small controlled agents to deploy a verticalized ecosystem to acomplish advanced data engineering tasks.
- integrate advanced mechanism for Human-in-the-loop and MCP integration.

---

## 📁 Configuration

- Each agent reads its configuration from a YAML file in the `/config` folder.

- Prompt files are also externalized for modularity. Each agent have its configuration prompts in YAML files in the `/promnpt` folder. 

- Additional configurable PATH can be placed into `/config/config.yml` file, use it to reference your data file names in a correct position way to start to use a specific agent.

- Additional instruction in specific agents folders
