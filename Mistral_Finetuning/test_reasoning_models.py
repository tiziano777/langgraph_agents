import json
from vllm import SamplingParams
from unsloth import FastLanguageModel
import yaml

# --- LOAD CONFIG ---
CONFIG_FILE = "config/mistral7B_instruct_v3_reasoning.yml"
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

MODEL_CHECKPOINT_DIR = config["model_checkpoint_dir"]
MODEL_NAME = config["model"]["name"]
MAX_SEQ_LENGTH = config["model"]["max_seq_length"]
DTYPE = config["model"]["dtype"]
LOAD_IN_4BIT = config["model"]["load_in_4bit"]
PEFT_CONFIG = config["peft"]

# Inferenza da config
INFERENCE_CONFIG = config.get("inference", {})
OUTPUT_RESULTS_FILE = INFERENCE_CONFIG.get("output_results_file")
GENERATION_PARAMS = INFERENCE_CONFIG.get("generation_params", {})

# Default params di generazione se non presenti nel YAML
temperature = GENERATION_PARAMS.get("temperature", 0.3)
top_p = GENERATION_PARAMS.get("top_p", 0.95)
max_new_tokens = GENERATION_PARAMS.get("max_new_tokens", 512)
top_k = GENERATION_PARAMS.get("top_k", None)
do_sample = GENERATION_PARAMS.get("do_sample", True)
repetition_penalty = GENERATION_PARAMS.get("repetition_penalty", 1.0)

# --- Initialize Model and Tokenizer ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    fast_inference=False,
    max_lora_rank=PEFT_CONFIG["r"],
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = FastLanguageModel.get_peft_model(
    model,
    r=PEFT_CONFIG["r"],
    target_modules=PEFT_CONFIG["target_modules"],
    lora_alpha=PEFT_CONFIG["lora_alpha"],
    lora_dropout=PEFT_CONFIG["lora_dropout"],
    bias=PEFT_CONFIG["bias"],
    use_gradient_checkpointing=PEFT_CONFIG["use_gradient_checkpointing"],
    random_state=PEFT_CONFIG["random_state"],
)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\n" + "="*50)
print("RUNNING INFERENCE ON TEST SET WITH LoRA")
print("="*50)

test_data = load_jsonl("processed_test_dataset.jsonl")

try:
    lora_adapter = model.load_lora(f"{MODEL_CHECKPOINT_DIR}/grpo_lora")
except FileNotFoundError:
    print(f"LoRA adapter not found at {MODEL_CHECKPOINT_DIR}/grpo_lora. Aborting.")
    exit(1)

sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_new_tokens,
    top_k=top_k,
    do_sample=do_sample,
    repetition_penalty=repetition_penalty,
)

results = []
for idx, example in enumerate(test_data):
    try:
        prompt = example["prompt"]
        test_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        outputs = model.fast_generate([test_text], sampling_params=sampling_params, lora_request=lora_adapter)
        generated_text = outputs[0].outputs[0].text

        results.append({
            "document_id": example.get("document_id"),
            "chunk_id": example.get("chunk_id"),
            "prompt": prompt,
            "raw_text": example.get("raw_text"),
            "generated_answer": generated_text,
            "reference_answer": example.get("answer"),
        })

        if idx % 10 == 0:
            print(f"Inferred {idx + 1}/{len(test_data)} examples")

    except Exception as e:
        print(f"Error processing example {idx}: {e}")

save_jsonl(results, OUTPUT_RESULTS_FILE)
print(f"Inference completed. Results saved to {OUTPUT_RESULTS_FILE}")
