# NER Annotator Agent

Welcome to the `NER_annotator_agent` repository! This project implements a robust and flexible Named Entity Recognition (NER) pipeline. It's powered by a fine-tuned Large Language Model (LLM), it implements various LLM connections, including HF, remote hosted LLM, thid party Gemini API LLM and a local quantized model cpp; orchestrated using **LangGraph**. Our goal is to provide a reliable tool for extracting named entities, adaptable to many contexts.

---

## Table of Contents

- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [How to Use](#how-to-use)
- [Customization for New NER Domains](#customization-for-new-ner-domains)
- [License](#license)

---

## Key Features

- **LangGraph Architecture**: The core pipeline is built with **LangGraph**, providing a flexible execution logic that supports state management, conditional decisions, and retries for robust operation.
- **JSONL-Robust**: We've included mechanisms for handling imperfect LLM outputs (like malformed JSON) to ensure reliability.
- **Spacy Integration**: It utilizes **Spacy's** capabilities for advanced preprocessing, tokenization, and precise entity alignment.
- **Adaptability**: Designed with flexibility in mind, this agent can be easily modified and reused for NER tasks across various domains.

---

## Requirements

To get this project up and running, you'll need:

- **Python 3.8+** (Python 3.10 or newer is recommended).
- A **fine-tuned Mistral model** saved locally or accessible via the Hugging Face Hub.
- See **.env.template** file to visualize .env settings.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/tiziano777/langgraph_agents.git
cd langgraph_agents/NER_annotator_production_agent
```

### Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
Install Dependencies
```

```bash
pip install -r requirements.txt
```

### Create a .env file in the project's root directory (NER_annotator_agent). Example:

```bash
api_key=YOUR_API_KEY
hf_token= "your HF token"
prompt_path= 'your path to the specific_ner_prompts.yml file'
input_path="your path to the data.jsonl file"
output_path="your path to output.jsonl file"
checkpoint_path=" your path to checkpoint.json"
remote_llm="https://api.linkToYourLLM"
cost_log="your path to log/cost_log.jsonl file"
```

### config/ Directory

The config/ directory contains subfolders that hold specific settings for the models used during inference.

### How to Use
Ensure your virtual environment is active:

```bash
source .venv/bin/activate

```
Run the example pipeline:

```bash
python main.py
```

You can modify the input text in main.py to test different samples. The output will be printed to the console.

### Customization for New NER Domains
To adapt the agent to a new NER domain:

0. Select Your LLM source

1. Domain-Specific Preprocessing
Update the logic in:

```bash
nodes/preprocessing/ner_preprocessing.py
```
to reflect domain-specific tokenization, cleaning, or normalization.

2. Entity Schema Definition
Modify the Formatter node (typically in nodes/evaluators/) and update:

```bash
pipelines/tender_ner_pipeline.py
to reflect new target entities and validation logic.
```

3. Prompt Engineering
Adapt prompts within:

```bash
pipelines/tender_ner_pipeline.py
```

to match the new NER task and expected output schema.