# SPAN Annotator Agent

Welcome to the `SPAN_annotator_agent` repository!
This project implements a **LangGraph-based pipeline** for **logical span annotation** within arbitrary text inputs. Its goal is to identify and mark semantically relevant spans, providing a robust foundation for downstream NLP tasks such as **information extraction**, **argument mining**, or **structured annotation**.

<p align="center">
  <img src="SPAN_annotator_agent/graph.png" alt="Pipeline Graph" width="600"/>
</p>

---

## Table of Contents

* [Key Features](#key-features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Configuration](#configuration)
* [How to Use](#how-to-use)
* [Customization](#customization)
* [License](#license)

---

## Key Features

* **Logical Span Annotation**: Detects and annotates semantically relevant spans of text, rather than restricting to entities only.
* **LangGraph Architecture**: Built entirely with **LangGraph**, ensuring modularity, explicit state handling, and robust execution of pipeline logic.
* **Directory Structure for Modularity**:

  * `prompt/`: prompt templates for span extraction and validation.
  * `states/`: definitions of graph states and transitions.
  * `utils/`: helper functions and general utilities.
  * `pipelines/`: main orchestration pipelines.
  * `nodes/`: atomic operations for preprocessing, annotation, evaluation.
  * `config/`: configuration files for model and runtime parameters.
  * `data/{input, output, checkpoint}`: test inputs, generated outputs, and checkpoint states.

---

## Requirements

* **Python 3.8+** (Python 3.10+ recommended).
* **LangGraph** and other dependencies listed in `requirements.txt`.
* Access to a **Google Large Language Model (LLM)** (GEMINI_API_KEY).

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/tiziano777/langgraph_agents.git
cd langgraph_agents/SPAN_annotator_agent
```

### Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

### Environment Variables

Copy `.env.template` to `.env` in the project root and configure your system prompts and model paths. Example:

```bash
api_key="YOUR GEMINI API KEY"
```

### Config Directory

* `config/` contains YAML/JSON configs for inference, pipelines, and model settings.
* Update these files according to your deployment environment.

---

## How to Use

Activate your virtual environment:

```bash
source .venv/bin/activate
```

Run the default pipeline:

```bash
python main.py
```

* Modify `data/input/` files for your test samples.
* The outputs will be saved under `data/output/` in JSON or structured format.
* Ensure that the fiel is a file.jsonl and  initialize a checpoint.json file as follows to start reading fiel from starting line:
```json
{
    "checkpoint":0
}
```

---

## Customization

1. **Prompts**

   * Adapt prompt templates inside `prompt/` to reflect your annotation schema.

2. **Pipelines**

   * Modify or extend pipelines in `pipelines/` for domain-specific logic.
   * We have defined two different pipeliens, one with refinement cycle and one without it.

3. **Nodes**

   * Create or extend processing nodes in `nodes/` to support additional preprocessing, validation, or span alignment.

4. **State Management**

   * Update state definitions in `states/` to handle new control flows or error cases.

5. **Model Configuration**

   * Update `config/` files and `.env` for different models, domains, or deployment contexts.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
