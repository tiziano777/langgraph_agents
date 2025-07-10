from unsloth import FastLanguageModel
import os
import json
from datasets import load_dataset
from tqdm import tqdm
from json_repair import repair_json # Importa repair_json
import yaml # Aggiungi l'importazione di yaml

# --- Caricamento Variabili d'Ambiente e Login a Hugging Face ---
from dotenv import load_dotenv
from huggingface_hub import login 
load_dotenv()
hf_token = os.environ.get("hf_token")

# --- Caricamento Configurazione da YAML ---
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

# Estrai le configurazioni per l'inferenza
TEST_DATASET_PATH = config["output_test_file_path"] # Usa il percorso dal training config
MODEL_CHECKPOINT_PATH = config["model_checkpoint_dir"] # Percorso del modello fine-tuned
OUTPUT_INFERENCE_FILE = config["inference"]["output_results_file"]
INFERENCE_GENERATION_PARAMS = config["inference"]["generation_params"]

# Parametri del modello dal config (usati per caricare il modello)
MAX_SEQ_LENGTH = config["model"]["max_seq_length"]
DTYPE = config["model"]["dtype"]
LOAD_IN_4BIT = config["model"]["load_in_4bit"]


if hf_token:
    try:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face Hub.")
    except Exception as e:
        print(f"Warning: Failed to log in to Hugging Face Hub: {e}")
        print("Please ensure your HF_TOKEN is valid and restart your environment.")
else:
    print("Warning: HF_TOKEN not found in environment variables. Model download/access might be limited.")


def extract_and_repair_json(json_text: str) -> list:
    """
    Ripara e deserializza l'output JSON generato da LLM.
    Restituisce una lista di dizionari in caso di successo o una lista vuota in caso di errore.
    """
    try:
        repaired_text = repair_json(json_text)
        
        parsed_json = json.loads(repaired_text)

        # Normalizza l'output: assicurati che sia sempre una lista di dizionari
        if isinstance(parsed_json, dict):
            return [parsed_json] # Se il JSON riparato è un singolo dizionario, lo avvolgiamo in una lista
        elif isinstance(parsed_json, list):
            # Filtra per assicurarti che tutti gli elementi siano dizionari validi
            return [item for item in parsed_json if isinstance(item, dict)]
        else:
            # Se l'output non è né una lista né un dizionario (es. "null", "true", "")
            print(f"Warning: JSON riparato non è né una lista né un dizionario. Tipo: {type(parsed_json)}. Contenuto: {repaired_text[:100]}...")
            return []

    except Exception as e:
        print(f"Errore durante riparazione/parsing JSON: {e}")
        print(f"Testo JSON malformato originale: {json_text[:200]}...")
        return [] # Restituisce una lista vuota in caso di qualsiasi errore


try:
    raw_test_dataset = load_dataset("json", data_files=TEST_DATASET_PATH, split="train")
    print(f"Dataset di test caricato da: {TEST_DATASET_PATH}")
    print(f"Dimensione del dataset di test: {len(raw_test_dataset)} esempi")
except Exception as e:
    print(f"Errore nel caricamento del dataset di test da {TEST_DATASET_PATH}: {e}")
    exit()

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_CHECKPOINT_PATH, # Carica dal percorso del checkpoint
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
        token = hf_token # Ancora utile per l'accesso ai componenti del modello base se necessario
    )
    print(f"Modello e tokenizer caricati da: {MODEL_CHECKPOINT_PATH}")
except Exception as e:
    print(f"Errore nel caricamento del modello da {MODEL_CHECKPOINT_PATH}: {e}")
    print("Assicurati che il percorso sia corretto e che il modello sia stato salvato.")
    exit()

# --- Esegui l'Inferenza ---
print("Avvio inferenza sul dataset di test...")
inference_results = []

# Aggiungo pad_token_id qui in caso non sia già nel config, è importante per la generazione
generation_config = INFERENCE_GENERATION_PARAMS.copy() 
generation_config["eos_token_id"] = tokenizer.eos_token_id
generation_config["pad_token_id"] = tokenizer.pad_token_id or tokenizer.eos_token_id


for i, example in enumerate(tqdm(raw_test_dataset, desc="Inference")):
    input_prompt = example["text"] # Prompt formattato per l'inferenza
    # Prepara l'input per il modello
    inputs = tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
    # Genera l'output
    outputs = model.generate(**inputs, **generation_config)

    # Decodifica il testo generato dal modello, escludendo l'input iniziale
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    # print('generated_text: ' + str(generated_text)) # Decommenta per debugging

    # --- Estrai la stringa JSON grezza dall'output completo del modello ---
    # Il modello genererà: "...Output:\n[{"entity": "value"}]</s>"
    # Cerchiamo la parte dopo "Output:\n"
    predicted_json_str_raw = ""
    output_prefix = "Output:\n" # Questo è parte del tuo prompt template
    if output_prefix in generated_text:
        # Prende tutto ciò che segue "Output:\n"
        predicted_json_str_raw = generated_text.split(output_prefix, 1)[1].strip()
        # Rimuovi eventuali token di fine sequenza (come '</s>')
        if predicted_json_str_raw.endswith("</s>"): # A volte tokenizer.eos_token è '</s>'
            predicted_json_str_raw = predicted_json_str_raw[:-len("</s>")].strip()
    else:
        # Se il modello non segue il formato, prendiamo tutto il testo generato e proviamo a ripararlo
        # Questo è un fallback, idealmente il modello dovrebbe seguire il formato Output:\n
        predicted_json_str_raw = generated_text.strip()
    
    # --- Usa la funzione per riparare ed estrarre il JSON ---
    predicted_ner_entities = extract_and_repair_json(predicted_json_str_raw)

    inference_results.append({
        "id":example['id'],
        "chunk_id":example['chunk_id'],
        "ner": predicted_ner_entities,
        "ground_truth_output": example['output'], # Aggiungi anche la ground truth per confronto
        "raw_generated_text": generated_text # Utile per debugging
    })

with open(OUTPUT_INFERENCE_FILE, "w", encoding="utf-8") as f:
    for item in inference_results:
        # Salva in formato JSON compatto, come nel training
        f.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + "\n")

print(f"\nInferenza completata. I risultati sono stati salvati in: {OUTPUT_INFERENCE_FILE}")