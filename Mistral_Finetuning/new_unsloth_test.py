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
MODEL_CHECKPOINT_PATH = config["model_test_dir"] # Percorso del modello fine-tuned
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
    # Imposta il pad_token_id per il tokenizer se non già presente, è importante per la generazione
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

# Aggiungo pad_token_id e eos_token_id alla configurazione di generazione,
# sovrascrivendo se già presenti nel YAML (preferibile per questi specifici token)
generation_config = INFERENCE_GENERATION_PARAMS.copy() 
generation_config["eos_token_id"] = tokenizer.eos_token_id
# Fallback per pad_token_id se non definito, utile per batching durante la generazione
generation_config["pad_token_id"] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


for i, example in enumerate(tqdm(raw_test_dataset, desc="Inference")):
    input_prompt = example["input_for_inference"] 
    
    inputs = tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
    outputs = model.generate(**inputs, **generation_config)

    # --- Refined Decoding and Extraction Logic ---
    # 1. Decode the entire generated sequence (including the prompt)
    full_decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 2. Find the end of the input_prompt within the full_decoded_text
    #    The input_prompt should end with "[/INST]" and potentially a space.
    #    We need to make sure we accurately find the start of the model's *actual* generation.
    
    # Mistral Instruct template: <s>[INST] USER_PROMPT [/INST] ASSISTANT_RESPONSE
    # Your input_for_inference is: <s>[INST] <SCENARIO>...</EXAMPLES> User Input:...</INST>
    # So we expect the model to start generating *after* the final [/INST].
    
    # Clean the input_prompt from any leading <s> token for string matching
    clean_input_prompt = input_prompt.replace(tokenizer.bos_token, "").strip() # Remove <s>
    
    # Try to find the clean_input_prompt in the full_decoded_text
    # Add a space after [/INST] as that's what `apply_chat_template(add_generation_prompt=True)` does
    # This assumes your `input_for_inference` already has this structure.
    # Check if the full_decoded_text actually starts with the input prompt.
    if full_decoded_text.startswith(clean_input_prompt):
        # The generated part is everything after the input prompt
        generated_text_only = full_decoded_text[len(clean_input_prompt):].strip()
    else:
        # Fallback: If the generated text doesn't start with the input prompt,
        # it means the model might have truncated or altered the prompt during generation.
        # In this specific case where it repeats the whole template, the issue is more fundamental.
        # For now, let's keep the previous slicing logic as a primary method for the *actual* output portion
        # but apply robust cleaning.
        
        # This slicing `outputs[0][inputs["input_ids"].shape[1]:]` is generally the most accurate
        # for getting *only* the new tokens generated by the model.
        generated_output_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text_only = tokenizer.decode(generated_output_ids, skip_special_tokens=True).strip()

    # Further clean the extracted generated_text_only
    # Remove any potential trailing EOS token
    if generated_text_only.endswith(tokenizer.eos_token):
        generated_text_only = generated_text_only[:-len(tokenizer.eos_token)].strip()

    # The most critical part: if the model is repeating the entire template,
    # we need to extract only the *last* "Output:\n[...]" block.
    # This is a heuristic to handle misbehaving models.
    output_prefix_in_template = "Output:\n"
    
    # Split by the output_prefix and take the last part.
    # This handles cases where the template might be repeated multiple times.
    parts = generated_text_only.split(output_prefix_in_template)
    if len(parts) > 1:
        # The last part should be the desired JSON.
        predicted_json_str_raw = parts[-1].strip()
    else:
        # If "Output:\n" wasn't found as expected,
        # it might mean the model generated the template without the explicit "Output:\n",
        # or just generated garbage. Try to repair the whole thing as a last resort.
        predicted_json_str_raw = generated_text_only.strip()

    print('generated_text_raw (pre-repair): ' + str(predicted_json_str_raw)) # Decommenta per debugging

    # --- Usa la funzione per riparare ed estrarre il JSON ---
    predicted_ner_entities = extract_and_repair_json(predicted_json_str_raw)

    inference_results.append({
        "id": example['id'],
        "chunk_id": example['chunk_id'],
        "ner": predicted_ner_entities,
        "ground_truth_output": example['expected_output'], 
        "raw_generated_text": generated_text_only # Useful for debugging what the model actually generated
    })


with open(OUTPUT_INFERENCE_FILE, "w", encoding="utf-8") as f:
    for item in inference_results:
        # Salva in formato JSON compatto, come nel training
        f.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + "\n")

print(f"\nInferenza completata. I risultati sono stati salvati in: {OUTPUT_INFERENCE_FILE}")