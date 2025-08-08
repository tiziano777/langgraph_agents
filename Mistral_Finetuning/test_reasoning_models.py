from unsloth import FastLanguageModel
import os
import json
from datasets import load_dataset
from tqdm import tqdm
from json_repair import repair_json
import yaml
from huggingface_hub import login
from dotenv import load_dotenv

# --- Caricamento Variabili d'Ambiente e Login a Hugging Face ---
load_dotenv()
#hf_token = os.environ.get("hf_token")

# --- Caricamento Configurazione da YAML ---
CONFIG_FILE = "config/mistral7B_instruct_v3_reasoning.yml" # Assicurati che il percorso del file config sia corretto

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

# Estrai le configurazioni per l'inferenza
TEST_DATASET_PATH = config["output_test_file_path"]
MODEL_CHECKPOINT_PATH = config["model_test_dir"]
OUTPUT_INFERENCE_FILE = config["inference"]["output_results_file"]
INFERENCE_GENERATION_PARAMS = config["inference"]["generation_params"]

# Parametri del modello dal config (usati per caricare il modello)
MAX_SEQ_LENGTH = config["model"]["max_seq_length"]
DTYPE = config["model"]["dtype"]
LOAD_IN_4BIT = config["model"]["load_in_4bit"]
TENDER_PROMPT = config["TENDER_PROMPT"]
BID_PROMPT = config["BID_PROMPT"]
ORDER_PROMPT = config["ORDER_PROMPT"]


'''if hf_token:
    try:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face Hub.")
    except Exception as e:
        print(f"Warning: Failed to log in to Hugging Face Hub: {e}")
        print("Please ensure your HF_TOKEN is valid and restart your environment.")
else:
    print("Warning: HF_TOKEN not found in environment variables. Model download/access might be limited.")
'''

def extract_and_repair_json(json_text: str) -> list:
    """
    Ripara e deserializza l'output JSON generato da LLM.
    Restituisce una lista di dizionari in caso di successo o una lista vuota in caso di errore.
    """
    try:
        repaired_text = repair_json(json_text)
        
        parsed_json = json.loads(repaired_text)

        if isinstance(parsed_json, dict):
            return [parsed_json]
        elif isinstance(parsed_json, list):
            return [item for item in parsed_json if isinstance(item, dict)]
        else:
            print(f"Warning: Repaired JSON is not a list or dict. Type: {type(parsed_json)}. Content: {repaired_text[:100]}...")
            return []

    except Exception as e:
        print(f"Error repairing/parsing JSON: {e}")
        print(f"Original malformed JSON text: {json_text[:200]}...")
        return []

def get_document_type_from_id(document_id: str) -> str:
    """Helper function to determine document type from ID."""
    if "TENDER" in document_id.upper():
        return "TENDER"
    elif "ORDER" in document_id.upper():
        return "ORDER"
    elif "BID" in document_id.upper():
        return "BID"
    return "TENDER"

def create_grpo_prompt(example: dict) -> str:
    """Crea il prompt nel formato GRPO usato nel training."""
    input_text = example["text"]
    chunk_id = example["chunk_id"]
    document_id = example["id"]
    
    def _select_prompt(doc_id: str) -> str:
        if "BID" in doc_id:
            return BID_PROMPT
        elif "TENDER" in doc_id:
            return TENDER_PROMPT
        elif "ORDER" in doc_id:
            return ORDER_PROMPT
        else:
            raise ValueError(f"Unknown document type in ID: {doc_id}")

    prompt_template = _select_prompt(document_id)
    user_prompt = (
        f"{prompt_template}\n"
        f"User Input:\n"
        f"chunk_id: {chunk_id}\n"
        f"{input_text}\n"
        f"Output:\n"
    )
    return user_prompt


try:
    raw_test_dataset = load_dataset("json", data_files=TEST_DATASET_PATH, split="train")
    print(f"Dataset di test caricato da: {TEST_DATASET_PATH}")
    print(f"Dimensione del dataset di test: {len(raw_test_dataset)} esempi")
except Exception as e:
    print(f"Errore nel caricamento del dataset di test da {TEST_DATASET_PATH}: {e}")
    exit()

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_CHECKPOINT_PATH,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
        #token = hf_token,
    )
    
    # Aggiusta il pad token se non è impostato
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Modello e tokenizer caricati da: {MODEL_CHECKPOINT_PATH}")
except Exception as e:
    print(f"Errore nel caricamento del modello da {MODEL_CHECKPOINT_PATH}: {e}")
    print("Assicurati che il percorso sia corretto e che il modello sia stato salvato.")
    exit()

# --- Esegui l'Inferenza ---
print("Avvio inferenza sul dataset di test...")
inference_results = []

generation_config = INFERENCE_GENERATION_PARAMS.copy() 
generation_config["eos_token_id"] = tokenizer.eos_token_id
generation_config["pad_token_id"] = tokenizer.pad_token_id or tokenizer.eos_token_id


for i, example in enumerate(tqdm(raw_test_dataset, desc="Inference")):
    # Crea il prompt nel formato GRPO
    input_prompt = create_grpo_prompt(example)
    
    inputs = tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
    outputs = model.generate(**inputs, **generation_config)

    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    # Estraggo il JSON grezzo dall'output del modello
    predicted_json_str_raw = ""
    output_prefix = "Output:\n"
    if output_prefix in input_prompt: # Il prefix `Output:\n` è nel prompt
        if output_prefix in generated_text:
            predicted_json_str_raw = generated_text.split(output_prefix, 1)[1].strip()
        else:
            predicted_json_str_raw = generated_text.strip()
    else: # Fallback se il modello non genera il prefisso
        predicted_json_str_raw = generated_text.strip()

    # Rimuovi eventuali token di fine sequenza (come '</s>')
    if predicted_json_str_raw.endswith("</s>"):
        predicted_json_str_raw = predicted_json_str_raw[:-len("</s>")].strip()
    
    predicted_ner_entities = extract_and_repair_json(predicted_json_str_raw)

    inference_results.append({
        "id": example['id'],
        "chunk_id": example['chunk_id'],
        "ner": predicted_ner_entities,
        "ground_truth_output": json.loads(example['answer']), # 'answer' contiene la ground truth
        "raw_generated_text": generated_text
    })

with open(OUTPUT_INFERENCE_FILE, "w", encoding="utf-8") as f:
    for item in inference_results:
        f.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + "\n")

print(f"\nInferenza completata. I risultati sono stati salvati in: {OUTPUT_INFERENCE_FILE}")