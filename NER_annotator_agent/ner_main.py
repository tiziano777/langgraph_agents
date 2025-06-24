import yaml
from pipelines.cpp_ner_pipeline import run_pipeline

from pipelines.api_ner_pipeline import run_pipeline

'''
import os
from dotenv import load_dotenv
MODEL_CONFIG = "./config/gemini2.0-flash.yml"
# Carica la configurazione del modello
with open(MODEL_CONFIG, "r", encoding="utf-8") as f:
    llm_config = yaml.safe_load(f)
load_dotenv()
api_key = os.environ.get("api_key")
llm_config["api_key"] = api_key
'''


PROMPT_PATH= "/home/tiziano/langgraph_agents/NER_annotator_agent/prompt/ner_prompts.yml"

choiches=["SL_NER_PROMPT","IT_NER_PROMPT","UK_NER_PROMPT"]

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)
    
input_path="/home/tiziano/AutoAnnotator/data/input/slovenian_corpus.json"
output_path="/home/tiziano/AutoAnnotator/data/output/sl_ner_dataset.jsonl"
checkpoint_path="/home/tiziano/AutoAnnotator/data/checkpoint/sl_ner_checkpoint.json"

prompt=prompts[choiches[0]] # Slovene system prompt

run_pipeline(input_path=input_path,output_path=output_path,checkpoint_path=checkpoint_path,prompt=prompt)