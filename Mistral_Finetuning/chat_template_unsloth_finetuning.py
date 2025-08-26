import torch
from dotenv import load_dotenv
import os
import json
import yaml
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from scripts.training_callback import GenerationCallback
from datasets import load_dataset
#from huggingface_hub import login

# --- CUDA ---
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("Warning: CUDA is not available. Training will be slow.")

### --- LOAD ENV FILE SECRETS ---

load_dotenv()

#HF_TOKEN = os.environ.get("hf_token")

# --- YAML CONFIG FILE ---
CONFIG_FILE = "config/mistral7B_instruct_v3.yml" # Assicurati che questo percorso sia corretto

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

### PROMPTS ### 

TENDER_PROMPT = config["TENDER_PROMPT"]
BID_PROMPT = config["BID_PROMPT"]
ORDER_PROMPT = config["ORDER_PROMPT"]


# --- CONFIG FILE SETTINGS ---

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

# --- Inizializzazione Modello e Tokenizer ---
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
        #token=HF_TOKEN
    )
    # Imposta il pad_token_id per il tokenizer, spesso utile per il training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
    max_seq_length = MAX_SEQ_LENGTH,
)

# --- TRAINING DATA PROCESSING: Funzione di Formattazione per Training/Validation Dataset ---

def format_ner_example_for_training(example): # Rinominata per chiarezza
    input_text = example["text"]
    chunk_id = example["chunk_id"]
    document_id = example["id"] # Prendi l'ID per determinare il prompt


    cleaned_ner_list = []
    for entity_dict in example["ner"]:
        cleaned_entity_dict = {k: v for k, v in entity_dict.items() if v is not None}
        if cleaned_entity_dict:
            cleaned_ner_list.append(cleaned_entity_dict)

    if not cleaned_ner_list:
        output_json_string = "[]"
    else:
        output_json_string = json.dumps(cleaned_ner_list, ensure_ascii=False, separators=(',', ':'))

    # Determina quale prompt usare basandosi sul campo 'id'
    if "BID" in document_id:
        current_prompt_content = BID_PROMPT
    elif "TENDER" in document_id:
        current_prompt_content = TENDER_PROMPT
    elif "ORDER" in document_id:
        current_prompt_content = ORDER_PROMPT
    else:
        # Fallback o gestione di ID non previsti
        print(f"Warning: No specific prompt found for ID: {document_id}. Using TENDER_PROMPT as default.")
        current_prompt_content = TENDER_PROMPT # Puoi scegliere un prompt di default o sollevare un errore

    
    messages = [
        {"role": "user", "content": f"{current_prompt_content}\nUser Input:\nchunk_id: {chunk_id}\n{input_text}\nOutput:\n"},
        {"role": "assistant", "content": output_json_string}
    ]
    
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    return {"text": formatted_text}

# --- INFERNCE DATA PROCESSING: Funzione per Preparare il Dataset di Test per l'Inferenza ---
def format_ner_example_for_inference(example):
    input_text = example["text"]
    chunk_id = example["chunk_id"]
    document_id = example["id"] # Prendi l'ID per determinare il prompt

    # Determina quale prompt usare basandosi sul campo 'id'
    if "BID" in document_id:
        current_prompt_content = BID_PROMPT
    elif "TENDER" in document_id:
        current_prompt_content = TENDER_PROMPT
    elif "ORDER" in document_id:
        current_prompt_content = ORDER_PROMPT
    else:
        # Fallback o gestione di ID non previsti
        print(f"Warning: No specific prompt found for ID: {document_id}. Using TENDER_PROMPT as default.")
        current_prompt_content = TENDER_PROMPT # Puoi scegliere un prompt di default o sollevare un errore

    
    # Generiamo la stringa di input esattamente come la passeremmo al modello per l'inferenza
    messages = [
        {"role": "user", "content": f"{current_prompt_content}\nUser Input:\nchunk_id: {chunk_id}\n{input_text}"},
    ]
    
    # add_generation_prompt=True aggiungerà lo spazio necessario dopo [/INST] per la generazione
    input_for_inference = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Prepariamo la ground truth come stringa JSON
    cleaned_ner_list = []
    for entity_dict in example["ner"]:
        cleaned_entity_dict = {k: v for k, v in entity_dict.items() if v is not None}
        if cleaned_entity_dict:
            cleaned_ner_list.append(cleaned_entity_dict)

    if not cleaned_ner_list:
        expected_output_json_string = "[]"
    else:
        expected_output_json_string = json.dumps(cleaned_ner_list, ensure_ascii=False, separators=(',', ':'))

    return {
        "id": example["id"],
        "chunk_id": example["chunk_id"],
        "text": input_for_inference,
        "output": expected_output_json_string
    }


# --- SPLIT TRAIN/EVAL/TEST Dataset ---
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
eval_dataset = eval_test_split["train"] # EVAL
test_dataset = eval_test_split["test"] # Questo è il dataset che formatteremo diversamente

print(f"Training Dataset size: {len(train_dataset)} examples")
print(f"Validation Dataset size: {len(eval_dataset)} examples")
print(f"Test Dataset size: {len(test_dataset)} examples")

# --- TRAIN/VALIDATION PREPARATION ---

# Qui si applica format_ner_example_for_training
processed_train_dataset = train_dataset.map(format_ner_example_for_training, batched=False)
processed_eval_dataset = eval_dataset.map(format_ner_example_for_training, batched=False)

# Rimuovi le colonne non necessarie per il training/validation
columns_to_keep_train_eval = ["id", "chunk_id", "text"]
columns_to_remove_train_eval = [col for col in processed_train_dataset.column_names if col not in columns_to_keep_train_eval]
processed_train_dataset = processed_train_dataset.remove_columns(columns_to_remove_train_eval)
processed_eval_dataset = processed_eval_dataset.remove_columns(columns_to_remove_train_eval)

print("\nExample of 'text' column formatted for TRAINING (input+output):")
print(processed_train_dataset[0]["text"])


# --- TEST DATASET PREPARATION ---

# Applichiamo format_ner_example_for_inference al test_dataset originale
processed_test_dataset = test_dataset.map(format_ner_example_for_inference, batched=False)

# Rimuovi tutte le colonne che non sono quelle desiderate per il dataset di test finale
# Le colonne desiderate ora sono "id", "chunk_id", "text", "output"
columns_to_keep_test_final = ["id", "chunk_id", "text", "output"]
columns_to_remove_test_final = [col for col in processed_test_dataset.column_names if col not in columns_to_keep_test_final]
processed_test_dataset = processed_test_dataset.remove_columns(columns_to_remove_test_final)

# SAVE TEST DATASET FOR LATER
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


# --- TRAINER SETUP ---

# Imposta i parametri di precisione dinamicamente per TrainingArguments
TRAINING_ARGS_DICT["fp16"] = not torch.cuda.is_bf16_supported()
TRAINING_ARGS_DICT["bf16"] = torch.cuda.is_bf16_supported()

training_args = TrainingArguments(**TRAINING_ARGS_DICT)

generation_callback = GenerationCallback(
    info=training_args,
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=MAX_NEW_TOKENS,
    eval_dataset_for_inference=processed_eval_dataset, # Passa il tuo dataset di validazione processato
    num_examples=EVAL_NUM_EXAMPLES, # Numero di esempi da stampare ad ogni intervallo
    log_steps_interval=LOG_STEP_INTERVAL # Ogni quanti step stampare gli esempi (regola in base alle tue esigenze)
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = processed_train_dataset,
    eval_dataset = processed_eval_dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    packing = True,
    args = training_args,
    callbacks = [generation_callback]
)

# --- Avvio Training ---
print("\nStarting training...")
trainer.train()

# --- Salvataggio Modello Fine-Tuned ---
print(f"\nTraining completed. Saving fine-tuned model to: {MODEL_CHECKPOINT_DIR}")
trainer.save_model(MODEL_CHECKPOINT_DIR)
tokenizer.save_pretrained(MODEL_CHECKPOINT_DIR)

print("Script finished successfully!")