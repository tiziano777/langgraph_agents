# Robust Annotator

> **LLM-based Label Linking node for LangGraph pipelines**  
> Handles robust extraction, normalization, and disambiguation of entities from text chunks labeled in IOB format (or others), using an external LLM API.
> The purpose of this Agent is to fill the gap between OCR text data noisy injection, and span matching respect referenced labels.
> OCR injecting errors, but also annotators can inject noise, defining a prompted LLM system with a pre-defined schema can help data engineers to acquire high quality data without huge effort.

---

## 🔍 Overview

This module implements an annotation linking node in LangGraph, responsible for coherent dataset linking with a set of labels (in IOB format in our case) into a structured list of entity mentions with semantic links, using a Large Language Model (LLM) as personal agent. It ensures robustness via retry mechanisms, exception handling, and JSON repair strategies.

---

## 📦 Project Architecture

annotation-agent/
├── config/             # Model configurations, API call setups,
├── prompt/             # Prompt definitions
├── data/
│   ├── checkpoint/     # Stores checkpoint.json for resumable execution
│   ├── input/          # Input data in JSONL format to be annotated
│   │   ├── labels/     # Contains files of annotations, with an identificator 1:1 with the data in the input data file
│   └── output/         # Annotated output in append mode (JSONL extension)
├── log/                # Logging of token/cost usage and execution metadata
├── nodes/              # Definition of custom LangGraph node classes, every classe take in input a State object
├── states/             # Graph state definitions, It is a message exchanged between nodes in the graph
├── pipelines/          # Pipelines, graph definitions of flows of nodes and edges
├── utils/              # Cost estimator, helpers, error handler etc...
├── main.py             # Execution script to run one or more pipelines
├── requirements.txt    # Required Python dependencies
└── README.md           # Project documentation
└── .env.template       # use this template to create a .env file with your secrets

---

## 🧠 Core CUSTOM Functionalities 

### 1. `Annotator.process_text(text: str) -> str`
    - Normalizes raw text from OCR (e.g., lowercase, diacritic removal, punctuation cleanup).
    - Preserves and restores date patterns or other entities with some quick fixed errors.
    - Mitigates issues from irregular whitespace and control characters.
    - you can modify it if you have to handle different type of texts.

### 2. `Annotator.extract_iob_labels(state: State) -> List[Dict[str, str]]`
    - Parses IOB-tagged JSON from `state.labels_path`.
    - Groups contiguous `B-`/`I-` tags into entity mentions.
    - Returns a list of dict {"entity_name": "span string"}:  
    ```json
    [{"ORG": "AS Roma"}, {"DATE": "15.06.2023"}, ...]
    ```
### 3. Annotator.annotate(state: State) -> dict
    Apply previous functions and then perform an LLM API call using a prompt and both text and related labels.

    Invokes LLM with retry logic via ErrorHandler.
    Repairs and parses JSON output from LLM.
    Returns dictionary with:
    - initial_labels: extracted IOB mentions

    - labels: LLM-linked entity structure

    - input_tokens, output_tokens: token usage metadata

**Checkpointing**:

* `data/checkpoint/checkpoint.json` contains a file that keep track of advancements:

  ```json
  {
      "checkpoint": 0
  }
  ```
* This index tracks the item in the dataset where annotation should resume in case of interruption, if not present, the code generates and initialize it to zero.

## Initialization Commands

To initialize the project correctly, run the following commands:

```bash
mkdir -p data/checkpoint
mkdir -p data/input
mkdir -p data/output
mkdir -p logs

# Initialize the checkpoint file
cat <<EOF > data/checkpoint/checkpoint.json
{
    "checkpoint": 0
}
EOF
```

## Cost Estimation

A utility is provided in `utils/` to estimate API usage cost. Initially configured for Google's Gemini model (input/output inference), this can be easily adapted to any API service of choice.

## Customization Notes

Significant portions of the codebase are marked with `### CUSTOM LOGIC ###` comments. These are placeholders for adapting the agent to a specific use case. In particular, they should be revised to:

* Reflect the correct input keys for your data format
* Apply logic specific to your annotation schema

## Applications

This linking annotation agent has been applied to various tasks, but the core purpose is joinind data extracted from documents, eventually correct it to be alligned with the anotations, related to the doc files, and allign it by some pre-defined rules thanks to a prompted LLM.


## Installation

To install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Next Advancements

Add an Annotator node that use a cpp quantized local model instead of an LLM API service,
for companies that requires privacy, or low reasoning is enough for the task and we not havce access to a commercial LLM.

## Final Remarks

Feel free to use and adapt this linker agent for any span-based annotation match needs in your projects. Its flexible design allows integration with diverse language models and annotation schemes.
