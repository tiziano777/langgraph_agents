# NER Annotator Production Agent

Welcome to the `NER_annotator_production_agent` repository! This project implements a robust and flexible Named Entity Recognition (NER) pipeline. It's powered by a fine-tuned Large Language Model (LLM), specifically **Mistral**, and orchestrated using **LangGraph**. Our goal is to provide a reliable tool for extracting named entities, initially optimized for the **"tender" domain**, but easily adaptable to other contexts.

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

- **Fine-tuned LLM-based NER**: This project leverages an optimized **Mistral model** for high-performance entity extraction.
- **LangGraph Architecture**: The core pipeline is built with **LangGraph**, providing a flexible execution logic that supports state management, conditional decisions, and retries for robust operation.
- **Production Readiness**: We've included mechanisms for handling imperfect LLM outputs (like malformed JSON) to ensure reliability in production environments.
- **Spacy Integration**: It utilizes **Spacy's** capabilities for advanced preprocessing, tokenization, and precise entity alignment.
- **Adaptability**: Designed with flexibility in mind, this agent can be easily modified and reused for NER tasks across various domains.

---

## Requirements

To get this project up and running, you'll need:

- **Python 3.8+** (Python 3.10 or newer is recommended).
- A **fine-tuned Mistral model** saved locally or accessible via the Hugging Face Hub.
- A **Hugging Face Token (`HF_TOKEN`)** with read permissions.

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

### Download Spacy Language Models (Optional but Recommended)

```bash
python -m spacy download en_core_web_lg
python -m spacy download it_core_news_lg
Configuration
.env File
```

### Create a .env file in the project's root directory (NER_annotator_production_agent). Example:

```bash
HF_TOKEN="your_huggingface_token_here"
MODEL_PATH="langgraph_agents/NER_annotator_agent/local_llm/fine_tuned_mistral_ner_model_v1.3"
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

1. Fine-tuned LLM
Training: Fine-tune a Mistral model on your domain-specific NER dataset.

Loading: Update MODEL_PATH in your .env file.

2. Domain-Specific Preprocessing
Update the logic in:

```bash
nodes/preprocessing/ner_preprocessing.py
```
to reflect domain-specific tokenization, cleaning, or normalization.

3. Entity Schema Definition
Modify the Formatter node (typically in nodes/evaluators/) and update:

```bash
pipelines/tender_ner_pipeline.py
to reflect new target entities and validation logic.
```

4. Prompt Engineering
Adapt prompts within:

```bash
pipelines/tender_ner_pipeline.py
```

to match the new NER task and expected output schema.

5. Spacy Integration (if applicable)
If needed, replace generic Spacy models with domain-specific ones (e.g., biomedical models).