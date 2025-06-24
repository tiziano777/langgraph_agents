import yaml
from pipelines.cpp_ner_pipeline import run_pipeline


PROMPT_PATH= "/home/tiziano/AutoAnnotator/src/config/ner_prompts.yml"
choiches=["SL_NER_PROMPT","IT_NER_PROMPT","UK_NER_PROMPT"]
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

input_path="/home/tiziano/AutoAnnotator/data/raw/slovenian_corpus.json"
output_path="/home/tiziano/AutoAnnotator/data/outputs/sl_ner_dataset.jsonl"
checkpoint_path="/home/tiziano/AutoAnnotator/data/checkpoint/sl_ner_checkpoint.json"

prompt=prompts[choiches[0]] # Slovene system prompt
run_pipeline(input_path=input_path,output_path=output_path,checkpoint_path=checkpoint_path,prompt=prompt)