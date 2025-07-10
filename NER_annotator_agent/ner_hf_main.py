import yaml
import os
import logging
from dotenv import load_dotenv
#from transformers import AutoTokenizer
from huggingface_hub import login

# === Load ENV Variables ===

load_dotenv()

# === Setup Logging ===

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"api_ner_pipeline.log")

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(log_filename, encoding='utf-8'),logging.StreamHandler()])
logger = logging.getLogger(__name__)

###############################################################

### CHOOSE YOUR LLM STRATEGY ###
# 1) Local LLM from HF, Inference
# 2) TODO: Local LLM from HF, Inference + Refinement Loop

from pipelines.hf_ner_pipeline import run_pipeline
#from pipelines.hf_ner_refine_pipeline import run_pipeline

###############################################################

# === LOCAL MODEL SETUP (MISTRAL)===

MISTRAL_PATH = "/home/tiziano/langgraph_agents/NER_annotator_agent/config/config_finetuned_mistral7B-0.3.yml"
with open(MISTRAL_PATH, "r", encoding="utf-8") as f:
    llm_config = yaml.safe_load(f)
logger.info(f"Configurazione caricata da {MISTRAL_PATH}")

hf_token=os.environ.get("hf_token")
llm_config['hf_token']=hf_token

# --- HF LOGIN ---
if hf_token:
    try:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face Hub.")
    except Exception as e:
        print(f"Warning: Failed to log in to Hugging Face Hub: {e}")
        print("Please ensure your HF_TOKEN is valid and restart your environment.")
else:
    print("Warning: HF_TOKEN not found in environment variables. Model download/access might be limited.")

##########################################################

### TOKENIZER TO SEE PROMPTS LENGTH ###

'''def count_tokens(text: str) -> int:
    """
    Conta il numero di token in un dato testo utilizzando il tokenizzatore Mistral-7B-Instruct-v0.3.

    Args:
        text (str): Il testo di cui contare i token.

    Returns:
        int: Il numero di token nel testo. Restituisce -1 se il tokenizzatore non può essere caricato.
    """
    # Carica le variabili d'ambiente dal file .env (necessario se questa funzione è usata in uno script separato)
    load_dotenv()
    hf_token = os.getenv("hf_token")

    # Inizializzazione del tokenizzatore (lo facciamo qui per rendere la funzione autonoma)
    try:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", token=hf_token)
    except Exception as e:
        print(f"Errore durante l'inizializzazione del tokenizzatore: {e}")
        print("Assicurati che il token HF sia valido e che il modello esista.")
        return -1 # Restituisce -1 per indicare un errore

    if tokenizer is None:
        return -1

    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)
'''

### CUSTOM PATHS TO:  - READ INPUT - STORE OUTPUT - CHECKPOINTING ###

# UNIQUE INPUT

input_path=os.environ.get('test_input_path')

# LOCAL HF OUTPUT AND CHECKPOINTING

output_path=os.environ.get('hf_test_output_path')
checkpoint_path=os.environ.get('hf_test_checkpoint_path')

######################################################################

run_pipeline(input_path=input_path,output_path=output_path,checkpoint_path=checkpoint_path,llm_config=llm_config,hf_token=hf_token)
