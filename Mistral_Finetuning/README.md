# Mistral Fine-Tuning with Unsloth

Welcome to the `Mistral_Finetuning` repository!
This project provides a collection of fine-tuning strategies for **Mistral LLMs** using the **Unsloth framework**. It includes multiple pipelines and trainers for supervised fine-tuning (SFT), reinforcement learning with GRPO, and hybrid strategies with early stopping and retrieval-augmented examples. The repository also provides evaluation utilities and model merging scripts to stabilize checkpoints.

---

## Table of Contents

* [Key Features](#key-features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Project Structure](#project-structure)
* [Configuration](#configuration)
* [How to Use](#how-to-use)
* [Customization](#customization)
* [License](#license)

---

## Key Features

* **Supervised Fine-Tuning (SFT)**: Standard fine-tuning pipelines with AdamW optimizer.
* **GRPO Training**: Reinforcement-learning style training using **Grouped Reward Policy Optimization (GRPO)**.
* **Early Stopping with Visual Evaluation**: Integrated callbacks for early stopping and on-the-fly sample evaluations for visual inspection.
* **RAG + In-Context Learning**: GRPO trainer with similarity-based few-shot examples integrated into prompts.
* **Checkpoint Stabilization**: Model merging script to average weights across checkpoints to reduce training variance.
* **Evaluation Pipelines**: Test scripts for fine-tuned models both with and without similarity-based smart retrieval.

---

## Requirements

* **Python 3.10+**
* **Unsloth** framework
* **Transformers** (Hugging Face)
* **Accelerate**
* **PEFT** (for parameter-efficient fine-tuning)
* Additional dependencies listed in `requirements.txt`

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/tiziano777/langgraph_agents.git
cd langgraph_agents/Mistral_Finetuning
```

### Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Project Structure

* **Training Pipelines**

  * `chat_template_unsloth_finetuning.py`: SFT trainer with AdamW optimizer.
  * `chat_template_unsloth_reasoning_personalized_finetuning.py`: GRPO trainer specialized for reasoning tasks.
  * `early_stopping_unsloth_finetuning.py`: SFT pipeline with early stopping callbacks and visual evaluation of samples.
  * `chat_template_unsloth_rag_reasoning_finetuning.py`: GRPO trainer with in-context learning and similarity-based few-shot examples.

* **Evaluation Scripts**

  * `test_reasoning_models.py`: Evaluate a fine-tuned Mistral model trained with GRPO.
  * `tunsloth_test_models.py`: Evaluate the model treined with SFT.

* **Utilities**

  * `model_merging.py`: Script to average weights across multiple checkpoints for stabilization.

* **Directories**

  * `config/`: model and training hyperparameter settings.
  * `data/`: datasets for fine-tuning and evaluation.
  * `models/`: saved checkpoints from different training runs.
  * `examples/`: documented examples for some methods.
  * `scripts/`: utils functions or one-shot functions.
  * `logs/`: log of execution params.

---

## Configuration

1. **Environment Variables**: Copy `.env.template` to `.env` and set optional paths for model API source.

2. **Config Files**: Modify YAML/JSON configs under `config/` for dataset paths, optimizer settings, and GRPO parameters.

---

## How to Use

### Run Supervised Fine-Tuning

```bash
python chat_template_unsloth_finetuning.py
```

### Run GRPO Training

```bash
python chat_template_unsloth_reasoning_personalized_finetuning.py
```

### Fine-Tuning with Early Stopping

```bash
python early_stopping_unsloth_finetuning.py \
  --config config/sft_earlystop.yaml
```

### GRPO with RAG + Few-Shot

```bash
python chat_template_unsloth_rag_reasoning_finetuning.py
```

### Merge Checkpoints

```bash
python model_merging.py 
```
---

## Customization

1. **Prompts**: Adjust templates in training scripts to align with your domain-specific tasks.
2. **Optimizers & Callbacks**: Modify config files to test different optimizers, schedulers, and callbacks.
3. **Evaluation Logic**: Extend evaluation scripts to include domain-specific metrics, especially for GRPO RL reward functions.
4. **Merging Strategy**: Adapt `model_merging.py` to try alternative averaging schemes (e.g., weighted averages).

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
