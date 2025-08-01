import torch
import re
import json
import yaml
from dotenv import load_dotenv
import os
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from huggingface_hub import login
from json_repair import repair_json

# --- CUDA ---
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("Warning: CUDA is not available. Training will be slow.")

# --- HF TOKEN ---
load_dotenv()
HF_TOKEN = os.environ.get("hf_token")

# --- YAML CONFIG FILE ---
CONFIG_FILE = "config/TENDER_mistral7B_v3_reasoning.yml"

# --- HF LOGIN ---
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("Successfully logged in to Hugging Face Hub.")
    except Exception as e:
        print(f"Warning: Failed to log in to Hugging Face Hub: {e}")
        print("Please ensure your HF_TOKEN is valid and restart your environment.")
else:
    print("Warning: HF_TOKEN not found in environment variables. Model download/access might be limited.")

# --- LOAD CONFIG ---
try:
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded successfully from {CONFIG_FILE}")
except FileNotFoundError:
    print(f"Error: Configuration file '{CONFIG_FILE}' not found. Please create it.")
    exit()
except yaml.YAMLError as e:
    print(f"Error parsing YAML configuration file: {e}")
    exit()

# --- Extract configurations ---
DATASET_PATH = config["dataset_path"]
MODEL_CHECKPOINT_DIR = config["model_checkpoint_dir"]

MODEL_NAME = config["model"]["name"]
MAX_SEQ_LENGTH = config["model"]["max_seq_length"]
DTYPE = config["model"]["dtype"]
LOAD_IN_4BIT = config["model"]["load_in_4bit"]

PEFT_CONFIG = config["peft"]
GRPO_CONFIG = config["grpo"]
VALID_ENTITY_KEYS_CONFIG = config.get("valid_entity_keys", {})

TENDER_PROMPT = config["TENDER_PROMPT"]
BID_PROMPT = config["BID_PROMPT"]
ORDER_PROMPT = config["ORDER_PROMPT"]

# --- Model and Tokenizer Initialization ---
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
        token = HF_TOKEN,
        fast_inference = False,
        max_lora_rank = PEFT_CONFIG["r"]
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Model and tokenizer loaded from: {MODEL_NAME}")
except Exception as e:
    print(f"Error loading model from {MODEL_NAME}: {e}")
    exit()

# Apply PEFT (LoRA) to model
model = FastLanguageModel.get_peft_model(
    model,
    r = PEFT_CONFIG["r"],
    target_modules = PEFT_CONFIG["target_modules"],
    lora_alpha = PEFT_CONFIG["lora_alpha"],
    lora_dropout = PEFT_CONFIG["lora_dropout"],
    bias = PEFT_CONFIG["bias"],
    use_gradient_checkpointing = PEFT_CONFIG["use_gradient_checkpointing"],
    random_state = PEFT_CONFIG["random_state"],
)

# --- Helper functions for JSON validation ---
def is_valid_json(text: str) -> bool:
    """Check if text is valid JSON"""
    try:
        json.loads(text)
        return True
    except:
        return False

def get_input_text_from_prompt(prompt_messages):
    """Estrae il testo di input utente da una lista di messaggi del prompt."""
    try:
        content = prompt_messages[0]["content"]
        if "User Input:" in content and "Output:" in content:
            start_index = content.find("User Input:") + len("User Input:")
            end_index = content.find("Output:")
            input_text = content[start_index:end_index].strip()
            input_text = re.sub(r'chunk_id: \d+\n', '', input_text, flags=re.IGNORECASE).strip()
            return input_text.lower().replace(" ", "").replace(".", "")
    except (TypeError, IndexError):
        return ""
    return ""

def get_document_type_from_id(document_id: str) -> str:
    if "TENDER" in document_id.upper():
        return "TENDER"
    elif "ORDER" in document_id.upper():
        return "ORDER"
    elif "BID" in document_id.upper():
        return "BID"
    return "TENDER"

# --- GRPO Dataset Preparation ---
def format_ner_example_for_grpo(example):
    """Format example for GRPO training"""
    input_text = example["text"]
    chunk_id = example["chunk_id"]
    document_id = example["id"]

    cleaned_ner_list = []
    for entity_dict in example["ner"]:
        cleaned_entity_dict = {k: v for k, v in entity_dict.items() if v is not None}
        if cleaned_entity_dict:
            cleaned_ner_list.append(cleaned_entity_dict)

    if "BID" in document_id:
        current_prompt_content = BID_PROMPT
    elif "TENDER" in document_id:
        current_prompt_content = TENDER_PROMPT
    elif "ORDER" in document_id:
        current_prompt_content = ORDER_PROMPT
    else:
        current_prompt_content = TENDER_PROMPT

    messages = [
        {"role": "user", "content": f"{current_prompt_content}\nUser Input:\nchunk_id: {chunk_id}\n{input_text}\nOutput:\n"}
    ]
    
    return {
        "prompt": messages,
        "answer": cleaned_ner_list,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "raw_text": input_text
    }

# --- Funzione per l'estrazione JSON con riparazione ---
def extract_json(json_text: str) -> list:
    """
    Ripara e deserializza l'output JSON generato da LLM. Restituisce una lista oppure {} in caso di errore.
    """
    try:
        repaired_text = repair_json(json_text)
        
        parsed_json = json.loads(repaired_text)

        if not isinstance(parsed_json, list):
            # Se il JSON riparato non è una lista, proviamo a metterlo in una lista
            if isinstance(parsed_json, dict):
                return [parsed_json]
            raise ValueError("Parsed JSON is not a list")

        return parsed_json

    except Exception as e:
        return {}

def is_list_of_single_key_dicts(data):
    """
    Verifica se un oggetto è una lista di dizionari,
    dove ogni dizionario contiene una sola coppia chiave-valore.
    """
    if not isinstance(data, list):
        return False
    
    for item in data:
        if not isinstance(item, dict) or len(item) != 1:
            return False
            
    return True

# --- GRPO Reward Functions ---
# --- Funzioni di Reward Persnalizzate ---

def json_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for valid JSON format"""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    r=0
    
    for response in responses:
        # Uso della funzione extract_json per la riparazione
        parsed_data = extract_json(response)
        
        if is_list_of_single_key_dicts(parsed_data):
            rewards.append(5.0)
        else:
            rewards.append(0.1)
    
    return rewards

def entity_correctness_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Reward function for entity extraction correctness"""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    answers = kwargs.get("answers")
    if answers is None:
        return [0.0] * len(completions)

    for i, response in enumerate(responses):
        try:
            # Uso della funzione extract_json per la riparazione
            extracted_entities = extract_json(response)
            expected_entities = answers[i]
            
            if not is_list_of_single_key_dicts(extracted_entities):
                rewards.append(0.0)
                continue
            
            score = 0
            
            extracted_tuples = {(k, v) for d in extracted_entities for k, v in d.items()}
            expected_tuples = {(k, v) for d in expected_entities for k, v in d.items()}

            intersection_count = len(extracted_tuples.intersection(expected_tuples))
            score += intersection_count

            incorrect_extractions = len(extracted_tuples - expected_tuples)
            score -= incorrect_extractions

            missing_extractions = len(expected_tuples - extracted_tuples)
            score -= missing_extractions

            if len(expected_tuples) > 0:
                normalized_score = score / len(expected_tuples)
                rewards.append(max(-2.0, min(2.0, normalized_score * 2.0)))
            else:
                if len(extracted_tuples) == 0:
                    rewards.append(2.0)
                else:
                    rewards.append(0.0)
            
        except Exception:
            rewards.append(0.0)
    
    return rewards

def entity_type_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for extracting valid entity types"""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    document_ids = kwargs.get("document_ids")
    valid_entity_keys_config = kwargs.get('valid_entity_keys_config', {})

    if document_ids is None or valid_entity_keys_config is None:
        return [0.0] * len(completions)

    for i, response in enumerate(responses):
        extracted_json = extract_json(response)
        doc_type = get_document_type_from_id(document_ids[i])
        
        try:

            if extracted_json=={}:
                rewards.append(0.0)
                continue
            
            valid_types = set(valid_entity_keys_config.get(doc_type, []))
            
            valid_types_count = 0
            total_entities = len(extracted_json)
            
            for entity in extracted_json:
                if isinstance(entity, dict):
                    for key in entity.keys():
                        if key in valid_types:
                            valid_types_count += 1
            
            if total_entities == 0:
                reward = -2.0
            else:
                reward = (valid_types_count / total_entities) * 3.0
            
            rewards.append(reward)
            
        except:
            rewards.append(0.0)
    
    return rewards

def response_length_reward_func(completions, **kwargs) -> list[float]:
    """Reward function to discourage overly long responses"""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response in responses:
        length = len(response)
        if length < 31:
            reward = -1.0
        elif length < 350 and length > 30:
            reward = 2.0
        elif length >= 350:
            reward = -1.0
        else:
            reward = 0.1
        
        rewards.append(reward)
    
    return rewards

def anti_hallucination_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Funzione di reward per penalizzare le allucinazioni.
    Assegna un punteggio alto se le entità estratte sono presenti nel testo originale.
    """
    responses = [completion[0]['content'] for completion in completions]
    rewards = []

    for i, response in enumerate(responses):
        input_text_lower_clean = get_input_text_from_prompt(prompts[i])
        extracted_json = extract_json(response)
        
        if extract_json=={}:
            rewards.append(0.0)
            continue
        
        try:
            extracted_entities = extracted_json
            total_entities = 0
            correct_entities = 0
            
            for entity in extracted_entities:
                if isinstance(entity, dict):
                    for _, value in entity.items():
                        total_entities += 1
                        entity_value_lower_clean = str(value).lower().replace(" ", "").replace(".", "")
                        
                        if entity_value_lower_clean in input_text_lower_clean:
                            correct_entities += 1
            
            if total_entities == 0:
                reward = 1.0
            else:
                reward = correct_entities / total_entities

            system_prompt_indicators = [" The provided example","User Input:", "Output:", "chunk_id", "SCHEMA", "<SCENARIO>","</SCENARIO>","<RULES>","</RULES>"]
            for indicator in system_prompt_indicators:
                if indicator.lower() in response.lower():
                    reward -= 1
            
            rewards.append(min(3.0, reward))
            
        except (json.JSONDecodeError, TypeError):
            rewards.append(0.0)
    
    return rewards

# --- Load and prepare dataset ---
try:
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"Dataset loaded from: {DATASET_PATH}")
except Exception as e:
    print(f"Error loading dataset from {DATASET_PATH}: {e}")
    exit()

# Split dataset
train_split = dataset.train_test_split(test_size=0.1, seed=PEFT_CONFIG["random_state"])
train_dataset = train_split["train"]

print(f"Training Dataset size: {len(train_dataset)} examples")

# Format dataset for GRPO
processed_train_dataset = train_dataset.map(format_ner_example_for_grpo, batched=False)

# --- GRPO Training Configuration ---
training_args = GRPOConfig(
    learning_rate = GRPO_CONFIG["learning_rate"],
    adam_beta1 = GRPO_CONFIG["adam_beta1"],
    adam_beta2 = GRPO_CONFIG["adam_beta2"],
    weight_decay = GRPO_CONFIG["weight_decay"],
    warmup_ratio = GRPO_CONFIG["warmup_ratio"],
    lr_scheduler_type = GRPO_CONFIG["lr_scheduler_type"],
    optim = GRPO_CONFIG["optim"],
    logging_steps = GRPO_CONFIG["logging_steps"],
    per_device_train_batch_size = GRPO_CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps = GRPO_CONFIG["gradient_accumulation_steps"],
    num_generations = GRPO_CONFIG["num_generations"],
    max_prompt_length = GRPO_CONFIG["max_prompt_length"],
    max_completion_length = MAX_SEQ_LENGTH - GRPO_CONFIG["max_prompt_length"],
    max_steps = GRPO_CONFIG["max_steps"],
    save_steps = GRPO_CONFIG["save_steps"],
    max_grad_norm = GRPO_CONFIG["max_grad_norm"],
    report_to = GRPO_CONFIG["report_to"],
    output_dir = MODEL_CHECKPOINT_DIR,
)

# --- Initialize GRPO Trainer ---
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        json_format_reward_func,
        entity_type_reward_func,
        response_length_reward_func,
        entity_correctness_reward_func,
        anti_hallucination_reward_func
    ],
    args = training_args,
    train_dataset = processed_train_dataset,
    answers=processed_train_dataset["answer"],
    document_ids=processed_train_dataset["document_id"],
    valid_entity_keys_config=VALID_ENTITY_KEYS_CONFIG,
)

# --- Start Training ---
print("\nStarting GRPO training...")
print("Note: You might see 0 rewards for the first 100+ steps. This is normal!")
trainer.train()

# --- Save Model ---
print(f"\nTraining completed. Saving GRPO fine-tuned model to: {MODEL_CHECKPOINT_DIR}")
model.save_lora(f"{MODEL_CHECKPOINT_DIR}/grpo_lora")

print("GRPO training finished successfully!")

# --- Test the trained model ---
print("\n" + "="*50)
print("TESTING THE TRAINED MODEL")
print("="*50)

test_text = tokenizer.apply_chat_template([
    {"role": "user", "content": f"{TENDER_PROMPT}\nUser Input:\nchunk_id: 0\npostopek 165 2023 da/sp povabilo k oddaji ponudbe narocnik vabi ponudnike da v skladu z navodili ponudnikom izdelajo ponudbo za popravilo centrifuge heraeus cryofuge 6000 naziv aparata centrifuga proizvajalec heraeus tip cryofuge 6000 inv.st. kljuke za oddelcne lekarne do najkasneje 28.04.2023 do 12 ure. vodja nabavne sluzbe matjaz stinek.) univ.dipl.ekon\nOutput:\n"}
], tokenize = False, add_generation_prompt = True)

try:
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature = 0.3,
        top_p = 0.95,
        max_tokens = 512,
    )

    print("Testing without LoRA:")
    output_base = model.fast_generate(
        [test_text],
        sampling_params = sampling_params,
        lora_request = None,
    )[0].outputs[0].text
    print(output_base)

    print("\nTesting with GRPO-trained LoRA:")
    output_grpo = model.fast_generate(
        [test_text],
        sampling_params = sampling_params,
        lora_request = model.load_lora(f"{MODEL_CHECKPOINT_DIR}/grpo_lora"),
    )[0].outputs[0].text
    print(output_grpo)
except ImportError:
    print("vllm not installed, skipping fast generation test.")
except FileNotFoundError:
    print(f"LoRA adapter not found at {MODEL_CHECKPOINT_DIR}/grpo_lora. Skipping test.")
except Exception as e:
    print(f"An error occurred during the test: {e}")