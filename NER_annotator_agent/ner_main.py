import yaml

### CHOSE YOUR LLM STRATEGY (LOCAL OR API) ###

#from pipelines.cpp_ner_pipeline import run_pipeline
from pipelines.api_ner_pipeline import run_pipeline

##############################################

'''
# === LOCAL MODEL SETUP ===

GEMMA_PATH = "/home/tiziano/langgraph_agents/NER_annotator_agent/config/config_gemma3.yml"
with open(GEMMA_PATH, "r", encoding="utf-8") as f:
    llm_config = yaml.safe_load(f)
logger.info(f"Configurazione caricata da {GEMMA_PATH}")

'''

# === API MODEL SETUP ===

import os
from dotenv import load_dotenv
MODEL_CONFIG = "./config/gemini2.0-flash.yml"

# Carica la configurazione del modello
with open(MODEL_CONFIG, "r", encoding="utf-8") as f:
    llm_config = yaml.safe_load(f)
load_dotenv()
api_key = os.environ.get("api_key")
llm_config["api_key"] = api_key

###########################

PROMPT_PATH= os.environ.get('prompt_path')
input_path=os.environ.get('input_path')
output_path=os.environ.get('output_path')
checkpoint_path=os.environ.get('checkpoint_path')

### CUSTOM PROMPTS REFERENCE ###

general_ner_prompts=["SL_NER_PROMPT","IT_NER_PROMPT","UK_NER_PROMPT"]
specific_ner_prompts=["TENDER_PROMPT","TENDER_REFINEMENT_PROMPT"]

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

base_prompt=prompts[specific_ner_prompts[0]] # Main task
refinement_prompt=prompts[specific_ner_prompts[1]] # Refinement prompt

### END CUSTOM PROMPTS REFERENCE ###

run_pipeline(input_path=input_path,output_path=output_path,checkpoint_path=checkpoint_path,base_prompt=base_prompt,refinement_prompt=refinement_prompt,llm_config=llm_config)