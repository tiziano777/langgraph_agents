import os
import json
import yaml
from utils.CostAnalyze import CostAnalyze

from pipelines.api_disinfo_pipeline_refiner import run_pipeline 
# from pipelines.api_disinfo_pipeline_no_refiner import run_pipeline 
# from pipelines.api_ner_pipeline import run_pipeline 
# from pipelines.cpp_ner_pipeline import run_pipeline 

# CONFIG
TOPIC = "election"
INPUT_PATH = f'./data/raw/{TOPIC}_articles_deduplicated.jsonl'
OUTPUT_PATH = f'./data/output/{TOPIC}_articles_annotated.jsonl'
CHECKPOINT_PATH = f'./data/checkpoint/{TOPIC}_checkpoint.json'

PROMPTS_PATH = './config/disinfo_prompts.yml'
MODEL_CONFIG = "./config/gemini2.0-flash.yml"

def main():
    # Carica i prompt
    with open(PROMPTS_PATH, 'r', encoding='utf-8') as file:
        prompts = yaml.safe_load(file)

    # Carica la configurazione del modello
    with open(MODEL_CONFIG, "r", encoding="utf-8") as f:
        llm_config = yaml.safe_load(f)

    

    # Crea file se non esistono
    if not os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            pass
    if not os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": 0}, f, ensure_ascii=False, indent=4)

    try:
        run_pipeline(INPUT_PATH, OUTPUT_PATH, CHECKPOINT_PATH, llm_config, prompts)
        #run_pipeline(INPUT_PATH, OUTPUT_PATH, CHECKPOINT_PATH, llm_config, prompts)
        
        #os.remove(CHECKPOINT_PATH)
        
        #costAnalyzer non richiesto per i modelli locali
        CostAnalyze().daily_cost_log()
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")


if __name__ == "__main__":
    main()
