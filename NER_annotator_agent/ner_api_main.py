import yaml
import os
import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer
# === Load ENV Variables ===

load_dotenv()

# === Setup Logging ===

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"api_ner_pipeline.log")

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(log_filename, encoding='utf-8'),logging.StreamHandler()])
logger = logging.getLogger(__name__)

### TOKENIZER TO SEE PROMPTS LEN ###

def count_tokens(text: str) -> int:
    """
    Conta il numero di token in un dato testo utilizzando il tokenizzatore Mistral-7B-Instruct-v0.3.

    Args:
        text (str): Il testo di cui contare i token.

    Returns:
        int: Il numero di token nel testo. Restituisce -1 se il tokenizzatore non può essere caricato.
    """
    # Carica le variabili d'ambiente dal file .env (necessario se questa funzione è usata in uno script separato)
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

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

###############################################################

### CHOOSE YOUR LLM STRATEGY ###
# 1) GEMINI API Direct Inference to the task
# 2) GEMINI API Inference + Refinement Loop (x2 API call usage)

from pipelines.gemini_api_ner_pipeline import run_pipeline
#from pipelines.gemini_api_ner_refine_pipeline import run_pipeline


###############################################################

# === GEMINI API MODEL SETUP ===

MODEL_CONFIG = "./config/gemini2.0-flash.yml"

# LOAD CONFIGS
with open(MODEL_CONFIG, "r", encoding="utf-8") as f:
    llm_config = yaml.safe_load(f)

api_key = os.environ.get("gemini_api_key")
llm_config["gemini_api_key"] = api_key

##########################################################

### CUSTOM PATHS TO:  - READ INPUT - STORE OUTPUT - CHECKPOINTING ###

# INPUT AND PROMPTS

PROMPT_PATH= os.environ.get('prompt_path')
input_path=os.environ.get('input_path')

# GEMINI OUTPUT AND CHECKPOINTING

checkpoint_path=os.environ.get('gemini_checkpoint_path')
output_path=os.environ.get('gemini_output_path')


####################################################################

### CUSTOM PROMPTS REFERENCE ###

# PRETRAINING DATASET CREATION FOR TRAINING A BERT CLASSIFICATOR
general_ner_prompts=["SL_NER_PROMPT","IT_NER_PROMPT","UK_NER_PROMPT"]

# FINETUNING DATASET PROMPTS
# USED TO CREATE AN AUTOANNOTATED DATASET FROM POWERFULL LLMS
# AND THEN USE THESE DATASET TO FINETUNE A MODEL
specific_ner_prompts=["TENDER_PROMPT","TENDER_REFINEMENT_PROMPT","BID_PROMPT","ORDER_PROMPT"]

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)
    
base_prompt=prompts[specific_ner_prompts[3]] # Main task
refinement_prompt=prompts[specific_ner_prompts[1]] # Refinement prompt

print('base prompt len: '+str(count_tokens(base_prompt)))
print('refinement prompt len: '+str(count_tokens(refinement_prompt)))

### END CUSTOM PROMPTS REFERENCE ###

######################################################################

run_pipeline(input_path=input_path,output_path=output_path,checkpoint_path=checkpoint_path,base_prompt=base_prompt,refinement_prompt=refinement_prompt,llm_config=llm_config)
