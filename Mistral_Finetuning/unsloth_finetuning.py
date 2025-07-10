import torch
from dotenv import load_dotenv
import os
import json
import yaml
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from huggingface_hub import login

# --- CUDA ---
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("Warning: CUDA is not available. Training will be slow.")

# --- HF TOKEN ---
load_dotenv()
HF_TOKEN = os.environ.get("hf_token")

# --- YAML CONFIG FILE ---
CONFIG_FILE = "config/mistral7B_instruct_v3.yml" # Assicurati che questo percorso sia corretto

# --- HF LOGIN ---
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("Successfully logged in to Hugging Face Hub.")
    except Exception as e:
        print(f"Warning: Failed to log in to Hugging Face Hub: {e}")
        print("Please ensure your HF_TOKEN is valid and restart your environment.")
else:
    print("Warning: HF_TOKEN not found in environment variables. Model download/access might be limited.")

# --- LOAD CONFIG ---
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

# --- Estrazione delle configurazioni ---
DATASET_PATH = config["dataset_path"]
OUTPUT_TEST_FILE_PATH = config["output_test_file_path"]
MODEL_CHECKPOINT_DIR = config["model_checkpoint_dir"]

MODEL_NAME = config["model"]["name"]
MAX_SEQ_LENGTH = config["model"]["max_seq_length"]
DTYPE = config["model"]["dtype"]
LOAD_IN_4BIT = config["model"]["load_in_4bit"]

PEFT_CONFIG = config["peft"]

# Carica i nuovi parametri per i token del prompt
START_INSTRUCTION_TOKEN = config["START_INSTRUCTION_TOKEN"]
END_INSTRUCTION_TOKEN = config["END_INSTRUCTION_TOKEN"]
# Contenuto del prompt senza i token di inizio/fine istruzione
PROMPT_TEMPLATE_CONTENT = config["TENDER_PROMPT"]

TRAINING_ARGS_DICT = config["trainer_args"]

# --- Inizializzazione Modello e Tokenizer ---
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
        token = HF_TOKEN,
    )
    print(f"Model and tokenizer loaded from: {MODEL_NAME}")
except Exception as e:
    print(f"Error loading model from {MODEL_NAME}: {e}")
    print("Please ensure the path is correct and the model is accessible.")
    exit()

# Applica PEFT (LoRA) al modello
model = FastLanguageModel.get_peft_model(
    model,
    r = PEFT_CONFIG["r"],
    target_modules = PEFT_CONFIG["target_modules"],
    lora_alpha = PEFT_CONFIG["lora_alpha"],
    lora_dropout = PEFT_CONFIG["lora_dropout"],
    bias = PEFT_CONFIG["bias"],
    use_gradient_checkpointing = PEFT_CONFIG["use_gradient_checkpointing"],
    random_state = PEFT_CONFIG["random_state"],
    max_seq_length = MAX_SEQ_LENGTH, # Utilizza MAX_SEQ_LENGTH
)

# --- Funzione di Formattazione per Training/Validation Dataset ---
# Questa funzione crea la singola stringa 'text' che il modello userà per imparare.
def format_ner_example(example):
    input_text = example["text"]
    chunk_id = example["chunk_id"] # Prendi il chunk_id dall'esempio

    cleaned_ner_list = []
    for entity_dict in example["ner"]:
        cleaned_entity_dict = {k: v for k, v in entity_dict.items() if v is not None}
        if cleaned_entity_dict:
            cleaned_ner_list.append(cleaned_entity_dict)

    if not cleaned_ner_list:
        output_json_string = "[]"
    else:
        output_json_string = json.dumps(cleaned_ner_list, ensure_ascii=False, separators=(',', ':'))

    # Il prompt completo che include i token di inizio/fine istruzione, il contenuto dell'istruzione,
    # l'input (con chunk_id) e l'output atteso.
    formatted_text = (
        f"{START_INSTRUCTION_TOKEN}\n" + # Inizia con <s>[INST] e una nuova riga
        f"{PROMPT_TEMPLATE_CONTENT}\n" + # Aggiungi il contenuto untokenized del prompt
        f"{END_INSTRUCTION_TOKEN}\n" + # Chiudi con [/INST] e una nuova riga
        f"Input:\nchunk_id: {chunk_id}\n{input_text}\n" + # Input con chunk_id
        f"Output:\n{output_json_string}</s>" # Output atteso e fine della sequenza del modello
    )
    return {"text": formatted_text}

# --- Caricamento e Suddivisione del Dataset ---
try:
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"Dataset loaded from: {DATASET_PATH}")
except Exception as e:
    print(f"Error loading dataset from {DATASET_PATH}: {e}")
    exit()

# Suddivisione del dataset
train_temp_split = dataset.train_test_split(test_size=0.10, seed=PEFT_CONFIG["random_state"])
train_dataset = train_temp_split["train"]
temp_dataset = train_temp_split["test"]

eval_test_split = temp_dataset.train_test_split(test_size=0.50, seed=PEFT_CONFIG["random_state"])
eval_dataset = eval_test_split["train"]
test_dataset = eval_test_split["test"]

print(f"Training Dataset size: {len(train_dataset)} examples")
print(f"Validation Dataset size: {len(eval_dataset)} examples")
print(f"Test Dataset size: {len(test_dataset)} examples")

# --- Preparazione Dataset per Training e Validation ---
processed_train_dataset = train_dataset.map(format_ner_example, batched=False)
processed_eval_dataset = eval_dataset.map(format_ner_example, batched=False)

# Rimuovi le colonne non necessarie per il training/validation
columns_to_keep_train_eval = ["id", "chunk_id", "text"] # Assuming id and chunk_id are useful for logging/tracking
columns_to_remove_train_eval = [col for col in processed_train_dataset.column_names if col not in columns_to_keep_train_eval]
processed_train_dataset = processed_train_dataset.remove_columns(columns_to_remove_train_eval)
processed_eval_dataset = processed_eval_dataset.remove_columns(columns_to_remove_train_eval)

print("\nExample of 'text' column formatted for TRAINING (input+output):")
print(processed_train_dataset[0]["text"])

# --- Preparazione Dataset per Test (con colonne separate) ---
# Primo passaggio: formatta con la stessa funzione del training per ottenere la 'text' completa
temp_processed_test_dataset = test_dataset.map(format_ner_example, batched=False)

# Secondo passaggio: estrai le colonne desiderate ('text' come input, 'output' come ground truth)

def extract_test_columns(example):
    full_sequence = example["text"] # Questa è la sequenza completa generata da format_ner_example

    # Il punto di divisione tra input (per inferenza) e output (ground truth)
    # è la stringa "Output:\n"
    output_marker = "Output:\n"
    output_marker_pos = full_sequence.rfind(output_marker)

    if output_marker_pos != -1:
        # L'input per l'inferenza è tutto ciò che precede la risposta effettiva del modello
        # In questo caso, include "Output:\n"
        inference_input_text = full_sequence[:output_marker_pos + len(output_marker)].strip() # <<< MODIFICA QUI
        
        # L'output atteso è solo la parte JSON più il token </s>
        expected_output_text = full_sequence[output_marker_pos + len(output_marker):].strip() # <<< E QUI

        # Rimuovi il token </s> dalla ground truth se presente
        if expected_output_text.endswith("</s>"):
            expected_output_text = expected_output_text[:-len("</s>")].strip()
            
    else:
        # Fallback se il formato del prompt non è quello atteso (dovrebbe essere raro)
        inference_input_text = full_sequence
        expected_output_text = ""
        print(f"Warning: '{output_marker}' not found in full sequence for example ID: {example.get('id', 'N/A')}")

    return {
        "id": example["id"],
        "chunk_id": example["chunk_id"],
        "text": inference_input_text, # Colonna 'text' per l'input di inferenza (include "Output:\n")
        "output": expected_output_text # Colonna 'output' per la ground truth (solo JSON)
    }

processed_test_dataset = temp_processed_test_dataset.map(extract_test_columns, batched=False)

# Rimuovi tutte le colonne che non sono quelle desiderate per il dataset di test finale
columns_to_keep_test_final = ["id", "chunk_id", "text", "output"]
columns_to_remove_test_final = [col for col in processed_test_dataset.column_names if col not in columns_to_keep_test_final]
processed_test_dataset = processed_test_dataset.remove_columns(columns_to_remove_test_final)

# Salva il dataset di test formattato
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

exit(0) 

# --- Inizializzazione Trainer ---
# Imposta i parametri di precisione dinamicamente per TrainingArguments
TRAINING_ARGS_DICT["fp16"] = not torch.cuda.is_bf16_supported()
TRAINING_ARGS_DICT["bf16"] = torch.cuda.is_bf16_supported()

training_args = TrainingArguments(**TRAINING_ARGS_DICT)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = processed_train_dataset,
    eval_dataset = processed_eval_dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    packing = True,
    args = training_args,
)

# --- Avvio Training ---
print("\nStarting training...")
trainer.train()

# --- Salvataggio Modello Fine-Tuned ---
print(f"\nTraining completed. Saving fine-tuned model to: {MODEL_CHECKPOINT_DIR}")
trainer.save_model(MODEL_CHECKPOINT_DIR)
tokenizer.save_pretrained(MODEL_CHECKPOINT_DIR)

print("Script finished successfully!")