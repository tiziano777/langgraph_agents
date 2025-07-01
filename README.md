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
| `langgraph_agents/NER_annotator_agent`   | NER Annotation Agent     | Annotates named entities from raw texts 
| `langgraph_agents/SPAN_annotator_agent`  | Span Detection Agent     | Identifies relevant spans (e.g., disinformation signals) in documents.     |

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


This pipelines enables:
- Creation of high-quality structured datasets
- Training of domain-specific classifiers
- Downstream analytics or integration into ML pipelines

---

## ⚙️ Current Capabilities

- ✅ Compatible with API-based LLMs (configurable via YAML)
- ✅ Supports local execution via quantized models (GGUF + C++)
- ✅ Modular design per agent
- ✅ Error handling and retry policies for API calls
- ✅ Prompt-level control and auditability

---

## 🔜 Planned Extensions

- Integration with authenticated enterprise REST endpoints for input/output messaging
- integrate advanced mechanism for Human-in-the-loop and MCP integration.
- Create a GUI in Streamlit in which Human can use easily the provided agentic tasks
- Workflow orchestration for chaining multiple agents
- Use this suite as starting of small controlled agents to deploy a verticalized ecosystem to acomplish advanced data engineering tasks.


---

## 📁 Configuration

- Each agent reads its configuration from a YAML file in the `/config` folder.

- Prompt files are also externalized for modularity. Each agent have its configuration prompts in YAML files in the `/prompt` folder. 

- Additional configurable PATH can be placed into `/config/config.yml` file, use it to reference your data file names in a correct position way to start to use a specific agent.

- Additional instruction in specific agents folders
