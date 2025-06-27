import os
from dotenv import load_dotenv
import json
import yaml
import traceback

from utils.CostAnalyze import CostAnalyze

from pipelines.api_linking_pipeline import run_pipeline 

# === CONFIGURATION ===
CONFIG_PATH = "./config/config.yml"

def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # Load global config
    config = load_config()

    TOPIC = config["TOPIC"]
    LANG = config["LANG"]
    DATA_PATH = config["DATA_PATH"].format(TOPIC=TOPIC)
    LABELS_PATH = config["LABELS_PATH"]
    OUTPUT_PATH = config["OUTPUT_PATH"].format(TOPIC=TOPIC)
    CHECKPOINT_PATH = config["CHECKPOINT_PATH"].format(TOPIC=TOPIC)
    PROMPTS_PATH = config["PROMPTS_PATH"]
    MODEL_CONFIG = config["MODEL_CONFIG"]

    # Load prompts
    with open(PROMPTS_PATH, 'r', encoding='utf-8') as file:
        prompts = yaml.safe_load(file)

    # Load model config
    with open(MODEL_CONFIG, "r", encoding="utf-8") as f:
        llm_config = yaml.safe_load(f)

    # Load API key
    load_dotenv()
    api_key = os.environ.get("api_key")
    llm_config["api_key"] = api_key

    # Ensure file existence
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    if not os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            pass

    if not os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": 0}, f, ensure_ascii=False, indent=4)

    # Run main pipeline
    try:
        run_pipeline(DATA_PATH, LABELS_PATH, OUTPUT_PATH, CHECKPOINT_PATH, LANG, llm_config, prompts)
        CostAnalyze().daily_cost_log()
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()