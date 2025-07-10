import yaml
import os
import logging
from dotenv import load_dotenv

# === Load ENV Variables ===

load_dotenv()

# === Setup Logging ===

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"api_ner_pipeline.log")

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(log_filename, encoding='utf-8'),logging.StreamHandler()])
logger = logging.getLogger(__name__)

###############################################################

### YOUR LLM STRATEGY ###
# cpp quantized local model

#TODO:
#from pipelines.cpp_ner_pipeline import run_pipeline


###############################################################

# === LOCAL MODEL SETUP (GEMMA3)===

GEMMA_PATH = "/home/tiziano/langgraph_agents/NER_annotator_agent/config/config_gemma3.yml"
with open(GEMMA_PATH, "r", encoding="utf-8") as f:
    llm_config = yaml.safe_load(f)
logger.info(f"Configurazione caricata da {GEMMA_PATH}")

##########################################################

### CUSTOM PATHS TO:  - READ INPUT - STORE OUTPUT - CHECKPOINTING ###

# UNIQUE INPUT AND PROMPTS

PROMPT_PATH= os.environ.get('prompt_path')
input_path=os.environ.get('input_path')

# CPP OUTPUT AND CHECKPOINTING

checkpoint_path=os.environ.get('cpp_checkpoint_path')
output_path=os.environ.get('cpp_output_path')

####################################################################

### CUSTOM PROMPTS REFERENCE ###

# PRETRAINING DATASET CREATION FOR TRAINING A BERT CLASSIFICATOR
general_ner_prompts=["SL_NER_PROMPT","IT_NER_PROMPT","UK_NER_PROMPT"]

# FINETUNING DATASET PROMPTS
# USED TO CREATE AN AUTOANNOTATED DATASET FROM POWERFULL LLMS
# AND THEN USE THESE DATASET TO FINETUNE A MODEL
specific_ner_prompts=["TENDER_PROMPT","TENDER_REFINEMENT_PROMPT"]

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

base_prompt=prompts[specific_ner_prompts[0]] # Main task
refinement_prompt=prompts[specific_ner_prompts[1]] # Refinement prompt

### END CUSTOM PROMPTS REFERENCE ###

######################################################################

run_pipeline(input_path=input_path,output_path=output_path,checkpoint_path=checkpoint_path,base_prompt=base_prompt,refinement_prompt=refinement_prompt,llm_config=llm_config)
