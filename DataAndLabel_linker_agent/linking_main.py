import os
from dotenv import load_dotenv
import json
import yaml
import traceback

from utils.CostAnalyze import CostAnalyze

from pipelines.api_linking_pipeline import run_pipeline 


# CONFIG
TOPIC = "linking"
LANG="sl"
DATA_PATH = f'./data/raw/{TOPIC}_data.jsonl'
LABELS_PATH = f'./data/raw/labels/'

OUTPUT_PATH = f'./data/output/{TOPIC}_dataset.jsonl'
CHECKPOINT_PATH = f'./data/checkpoint/{TOPIC}_checkpoint.json'

PROMPTS_PATH = './config/linking_prompts.yml'
MODEL_CONFIG = "./config/gemini2.0-flash.yml"

def main():
    # Carica i prompt
    with open(PROMPTS_PATH, 'r', encoding='utf-8') as file:
        prompts = yaml.safe_load(file)

    # Carica la configurazione del modello
    with open(MODEL_CONFIG, "r", encoding="utf-8") as f:
        llm_config = yaml.safe_load(f)
    
    #Load api key from environment variables
    load_dotenv()
    api_key = os.environ.get("api_key")
    llm_config["api_key"] = api_key


    # Crea file se non esistono
    if not os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            pass
    if not os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": 0}, f, ensure_ascii=False, indent=4)

    try:
        run_pipeline(DATA_PATH, LABELS_PATH, OUTPUT_PATH, CHECKPOINT_PATH, LANG, llm_config, prompts)
        
        #os.remove(CHECKPOINT_PATH)
        
        #costAnalyzer non richiesto per i modelli locali
        CostAnalyze().daily_cost_log()
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        traceback.print_exc()  # <-- stack trace completo


if __name__ == "__main__":
    main()
