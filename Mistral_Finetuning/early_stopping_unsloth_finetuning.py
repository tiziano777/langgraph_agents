import torch
from dotenv import load_dotenv
import os
import json
import yaml
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from scripts.training_callback import GenerationCallback
from datasets import load_dataset
from huggingface_hub import login
import optuna

# --- CUDA Setup ---
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("Warning: CUDA is not available. Training will be slow.")

# --- Hugging Face Token Login ---
load_dotenv()
HF_TOKEN = os.environ.get("hf_token")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("Successfully logged in to Hugging Face Hub.")
    except Exception as e:
        print(f"Warning: Failed to log in to Hugging Face Hub: {e}")
else:
    print("Warning: HF_TOKEN not found in environment variables. Model download/access might be limited.")

# --- Load Configuration ---
CONFIG_FILE = "config/mistral7B_instruct_v3.yml"
try:
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded successfully from {CONFIG_FILE}")
except FileNotFoundError:
    print(f"Error: Configuration file '{CONFIG_FILE}' not found. Please create it.")
    exit()
except yaml.YAMLError as e:
    print(f"Error parsing YAML configuration file: {e}")
    exit()

# --- Extract Configurations ---
TENDER_PROMPT = config["TENDER_PROMPT"]
BID_PROMPT = config["BID_PROMPT"]
ORDER_PROMPT = config["ORDER_PROMPT"]

DATASET_PATH = config["dataset_path"]
OUTPUT_TEST_FILE_PATH = config["output_test_file_path"]
MODEL_CHECKPOINT_DIR = config["model_checkpoint_dir"]

MODEL_NAME = config["model"]["name"]
MAX_SEQ_LENGTH = config["model"]["max_seq_length"]
DTYPE = config["model"]["dtype"]
LOAD_IN_4BIT = config["model"]["load_in_4bit"]

PEFT_CONFIG = config["peft"]
TRAINING_ARGS_DICT = config["trainer_args"]

MAX_NEW_TOKENS = config['callback']['max_new_tokens']
EVAL_NUM_EXAMPLES=config['callback']['num_examples']
LOG_STEP_INTERVAL=config['callback']["log_steps_interval"]

GRID_SEARCH_CONFIG = config.get("grid_search", {})

# Estrai i parametri di early stopping separatamente
EARLY_STOPPING_CONFIG = config.get("early_stopping", {}) # Nuova sezione nel YAML
EARLY_STOPPING_PATIENCE = EARLY_STOPPING_CONFIG.get("patience", 3)
EARLY_STOPPING_THRESHOLD = EARLY_STOPPING_CONFIG.get("threshold", 0.001)

# --- Dataset Formatting Functions ---
def format_ner_example_for_training(example):
    """Formats an example for training, including prompt and ground truth output."""
    input_text = example["text"]
    chunk_id = example["chunk_id"]
    document_id = example["id"]

    cleaned_ner_list = [
        {k: v for k, v in entity_dict.items() if v is not None} 
        for entity_dict in example["ner"]
    ]
    cleaned_ner_list = [item for item in cleaned_ner_list if item] # Remove empty dicts

    output_json_string = json.dumps(cleaned_ner_list, ensure_ascii=False, separators=(',', ':')) if cleaned_ner_list else "[]"

    if "BID" in document_id:
        current_prompt_content = BID_PROMPT
    elif "TENDER" in document_id:
        current_prompt_content = TENDER_PROMPT
    elif "ORDER" in document_id:
        current_prompt_content = ORDER_PROMPT
    else:
        current_prompt_content = TENDER_PROMPT # Default
        print(f"Warning: No specific prompt found for ID: {document_id}. Using TENDER_PROMPT as default.")

    messages = [
        {"role": "user", "content": f"{current_prompt_content}\nUser Input:\nchunk_id: {chunk_id}\n{input_text}\nOutput:\n"},
        {"role": "assistant", "content": output_json_string}
    ]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    return {"text": formatted_text, "ground_truth_output": output_json_string}

def format_ner_example_for_inference(example):
    """Formats an example for inference, creating the input prompt."""
    input_text = example["text"]
    chunk_id = example["chunk_id"]
    document_id = example["id"]

    if "BID" in document_id:
        current_prompt_content = BID_PROMPT
    elif "TENDER" in document_id:
        current_prompt_content = TENDER_PROMPT
    elif "ORDER" in document_id:
        current_prompt_content = ORDER_PROMPT
    else:
        current_prompt_content = TENDER_PROMPT # Default
        print(f"Warning: No specific prompt found for ID: {document_id}. Using TENDER_PROMPT as default.")
    
    messages = [
        {"role": "user", "content": f"{current_prompt_content}\nUser Input:\nchunk_id: {chunk_id}\n{input_text}"},
    ]
    input_for_inference = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    cleaned_ner_list = [
        {k: v for k, v in entity_dict.items() if v is not None} 
        for entity_dict in example["ner"]
    ]
    cleaned_ner_list = [item for item in cleaned_ner_list if item]
    expected_output_json_string = json.dumps(cleaned_ner_list, ensure_ascii=False, separators=(',', ':')) if cleaned_ner_list else "[]"

    return {
        "id": example["id"],
        "chunk_id": example["chunk_id"],
        "text": input_for_inference,
        "output": expected_output_json_string
    }

# --- Load and Split Dataset ---
try:
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"Dataset loaded from: {DATASET_PATH}")
except Exception as e:
    print(f"Error loading dataset from {DATASET_PATH}: {e}")
    exit()

train_temp_split = dataset.train_test_split(test_size=0.10, seed=PEFT_CONFIG["random_state"])
train_dataset = train_temp_split["train"]
temp_dataset = train_temp_split["test"]

eval_test_split = temp_dataset.train_test_split(test_size=0.50, seed=PEFT_CONFIG["random_state"])
eval_dataset = eval_test_split["train"]
test_dataset = eval_test_split["test"]

print(f"Training Dataset size: {len(train_dataset)} examples")
print(f"Validation Dataset size: {len(eval_dataset)} examples")
print(f"Test Dataset size: {len(test_dataset)} examples")

# --- Initialize Model and Tokenizer (for initial processing and if not using Optuna) ---
# This initial loading is used for dataset processing. For Optuna, it's re-loaded per trial.
model_init_for_dataset, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
    token = HF_TOKEN,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# --- Process Datasets ---
processed_train_dataset = train_dataset.map(format_ner_example_for_training, batched=False)
processed_eval_dataset = eval_dataset.map(format_ner_example_for_training, batched=False)

columns_to_keep_train_eval = ["id", "chunk_id", "text", "ground_truth_output"]
processed_train_dataset = processed_train_dataset.remove_columns(
    [col for col in processed_train_dataset.column_names if col not in columns_to_keep_train_eval]
)
processed_eval_dataset = processed_eval_dataset.remove_columns(
    [col for col in processed_eval_dataset.column_names if col not in columns_to_keep_train_eval]
)

print("\nExample of 'text' column formatted for TRAINING (input+output):")
print(processed_train_dataset[0]["text"])
print("\nExample of 'ground_truth_output' column for EVAL:")
print(processed_eval_dataset[0]["ground_truth_output"])

processed_test_dataset = test_dataset.map(format_ner_example_for_inference, batched=False)
columns_to_keep_test_final = ["id", "chunk_id", "text", "output"]
processed_test_dataset = processed_test_dataset.remove_columns(
    [col for col in processed_test_dataset.column_names if col not in columns_to_keep_test_final]
)

processed_test_dataset.to_json(
    OUTPUT_TEST_FILE_PATH,
    orient="records",
    lines=True,
    force_ascii=False
)
print(f"\nTest dataset saved to: {OUTPUT_TEST_FILE_PATH}")
print(f"\nExample of TEST dataset (id, chunk_id, text, output columns):")
print(processed_test_dataset[0])
print(f"Columns in processed_test_dataset: {processed_test_dataset.column_names}")

# --- Optuna Training Function ---
def run_training_optuna(trial):
    """Runs a single training trial for Optuna hyperparameter optimization."""
    current_training_args_dict = TRAINING_ARGS_DICT.copy()
    current_training_args_dict["learning_rate"] = trial.suggest_loguniform(
        "learning_rate", 
        GRID_SEARCH_CONFIG.get("learning_rate_min", 1e-9),
        GRID_SEARCH_CONFIG.get("learning_rate_max", 5e-9)
    )
    current_training_args_dict["per_device_train_batch_size"] = trial.suggest_categorical(
        "per_device_train_batch_size", GRID_SEARCH_CONFIG.get("per_device_train_batch_size_options", [4])
    )
    
    # Model and Tokenizer initialization for the trial
    model, current_tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
        token = HF_TOKEN,
    )
    if current_tokenizer.pad_token is None:
        current_tokenizer.pad_token = current_tokenizer.eos_token
    if current_tokenizer.pad_token_id is None:
        current_tokenizer.pad_token_id = current_tokenizer.eos_token_id

    # Apply PEFT with trial-specific parameters
    r_val = trial.suggest_categorical("peft_r", GRID_SEARCH_CONFIG.get("r_options", [PEFT_CONFIG["r"]]))
    lora_alpha_val = trial.suggest_categorical("peft_lora_alpha", GRID_SEARCH_CONFIG.get("lora_alpha_options", [PEFT_CONFIG["lora_alpha"]]))

    model = FastLanguageModel.get_peft_model(
        model,
        r = r_val,
        target_modules = PEFT_CONFIG["target_modules"],
        lora_alpha = lora_alpha_val,
        lora_dropout = PEFT_CONFIG["lora_dropout"],
        bias = PEFT_CONFIG["bias"],
        use_gradient_checkpointing = PEFT_CONFIG["use_gradient_checkpointing"],
        random_state = PEFT_CONFIG["random_state"],
        max_seq_length = MAX_SEQ_LENGTH,
    )

    current_training_args_dict["fp16"] = not torch.cuda.is_bf16_supported()
    current_training_args_dict["bf16"] = torch.cuda.is_bf16_supported()

    callbacks = [
        GenerationCallback(
            info=current_training_args_dict,
            model=model,
            tokenizer=current_tokenizer, # Pass trial's tokenizer
            max_new_tokens=MAX_NEW_TOKENS,
            eval_dataset_for_inference=processed_eval_dataset,
            num_examples=EVAL_NUM_EXAMPLES,
            log_steps_interval=LOG_STEP_INTERVAL
        ),
        EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD
        )
    ]
    
    current_training_args_dict["metric_for_best_model"] = "f1_score"
    current_training_args_dict["greater_is_better"] = True
    
    #IMPORTANTE!
    current_training_args_dict['output_dir']=f"{TRAINING_ARGS_DICT['output_dir']}/trial_{trial.number}"
    
    training_args = TrainingArguments(
        **current_training_args_dict
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = current_tokenizer, # Use trial's tokenizer
        train_dataset = processed_train_dataset,
        eval_dataset = processed_eval_dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        packing = True,
        args = training_args,
        callbacks = callbacks,
    )

    print(f"\nStarting training for trial {trial.number} with HPs: {trial.params}")
    trainer.train()

    eval_results = trainer.evaluate()
    return eval_results["eval_f1_score"]

# --- Main Execution ---
if __name__ == "__main__":
    if GRID_SEARCH_CONFIG.get("num_trials", 0) > 0:
        print(f"Starting Optuna hyperparameter optimization with {GRID_SEARCH_CONFIG['num_trials']} trials.")
        study = optuna.create_study(direction="maximize", study_name="mistral_ner_finetuning")
        
        study.optimize(run_training_optuna, n_trials=GRID_SEARCH_CONFIG["num_trials"])

        print("\nOptimization finished!")
        print("Best trial:")
        best_trial = study.best_trial
        print(f"  Value: {best_trial.value:.4f} (F1-score)")
        print("  Params:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        best_model_path = os.path.join(TRAINING_ARGS_DICT['output_dir'], f"trial_{best_trial.number}", "checkpoint-best")
        print(f"\nBest model from trial {best_trial.number} should be saved at: {best_model_path}")

    else:
        # Standard training without Optuna
        model = FastLanguageModel.get_peft_model(
            model_init_for_dataset, # Use the initially loaded model
            r = PEFT_CONFIG["r"],
            target_modules = PEFT_CONFIG["target_modules"],
            lora_alpha = PEFT_CONFIG["lora_alpha"],
            lora_dropout = PEFT_CONFIG["lora_dropout"],
            bias = PEFT_CONFIG["bias"],
            use_gradient_checkpointing = PEFT_CONFIG["use_gradient_checkpointing"],
            random_state = PEFT_CONFIG["random_state"],
            max_seq_length = MAX_SEQ_LENGTH,
        )

        TRAINING_ARGS_DICT["fp16"] = not torch.cuda.is_bf16_supported()
        TRAINING_ARGS_DICT["bf16"] = torch.cuda.is_bf16_supported()

        training_args = TrainingArguments(**TRAINING_ARGS_DICT)
        print(training_args)
        callbacks = [
            GenerationCallback(
                info=training_args,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
                eval_dataset_for_inference=processed_eval_dataset,
                num_examples=EVAL_NUM_EXAMPLES,
                log_steps_interval=LOG_STEP_INTERVAL
            ),
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD
            )
        ]
        
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = processed_train_dataset,
            eval_dataset = processed_eval_dataset,
            dataset_text_field = "text",
            max_seq_length = MAX_SEQ_LENGTH,
            packing = True,
            args = training_args,
            callbacks = callbacks
        )

        print("\nStarting standard training...")
        trainer.train()

        print(f"\nTraining completed. Saving fine-tuned model to: {MODEL_CHECKPOINT_DIR}")
        trainer.save_model(MODEL_CHECKPOINT_DIR)
        tokenizer.save_pretrained(MODEL_CHECKPOINT_DIR)

        print("Script finished successfully!")