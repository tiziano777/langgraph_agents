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
#HF_TOKEN = os.environ.get("hf_token")

# --- YAML CONFIG FILE ---
CONFIG_FILE = "config/TENDER_mistral7B_v3_reasoning.yml"

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
        fast_inference = False,  # Enable vLLM fast inference for GRPO
        max_lora_rank = PEFT_CONFIG["r"],
        #token = HF_TOKEN
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

def extract_json_from_response(json_text: str) -> list:
    """
    Ripara e deserializza l'output JSON generato da LLM. Restituisce una lista oppure {} in caso di errore.
    """
    try:
        print("json text: ", json_text)
        repaired_text = repair_json(json_text)
            
        parsed_json = json.loads(repaired_text)

        if not isinstance(parsed_json, list):
            raise ValueError("Parsed JSON is not a list")

        return parsed_json

    except Exception as e:
        return e

def validate_ner_entities(json_str: str, expected_entities: list) -> bool:
    """Validate if extracted entities match expected format"""
    try:
        extracted = json.loads(json_str)
        if not isinstance(extracted, list):
            return False
        
        # Check if all extracted entities have valid keys
        valid_keys = set()
        for exp_entity in expected_entities:
            valid_keys.update(exp_entity.keys())
        
        for entity in extracted:
            if not isinstance(entity, dict):
                return False
            for key in entity.keys():
                if key not in valid_keys:
                    return False
        
        return True
    except:
        return False

# --- GRPO Dataset Preparation ---
def format_ner_example_for_grpo(example):
    """Format example for GRPO training"""
    input_text = example["text"]
    chunk_id = example["chunk_id"]
    document_id = example["id"]

    # Clean NER entities
    cleaned_ner_list = []
    for entity_dict in example["ner"]:
        cleaned_entity_dict = {k: v for k, v in entity_dict.items() if v is not None}
        if cleaned_entity_dict:
            cleaned_ner_list.append(cleaned_entity_dict)

    # Select prompt based on document ID
    if "BID" in document_id:
        current_prompt_content = BID_PROMPT
    elif "TENDER" in document_id:
        current_prompt_content = TENDER_PROMPT
    elif "ORDER" in document_id:
        current_prompt_content = ORDER_PROMPT
    else:
        current_prompt_content = TENDER_PROMPT

    # Create prompt messages
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

# --- GRPO Reward Functions ---

def json_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for valid JSON format"""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response in responses:
        extracted_json = extract_json_from_response(response)
        if is_valid_json(extracted_json):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards

def entity_correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function for entity extraction correctness"""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for i, response in enumerate(responses):
        extracted_json = extract_json_from_response(response)
        
        try:
            extracted_entities = json.loads(extracted_json)
            expected_entities = answer[i]
            
            if not isinstance(extracted_entities, list):
                rewards.append(0.0)
                continue
            
            # Calculate precision and recall for entities
            extracted_set = set()
            expected_set = set()
            
            for entity in extracted_entities:
                if isinstance(entity, dict):
                    for k, v in entity.items():
                        extracted_set.add(f"{k}:{v}")
            
            for entity in expected_entities:
                if isinstance(entity, dict):
                    for k, v in entity.items():
                        expected_set.add(f"{k}:{v}")
            
            if len(expected_set) == 0:
                reward = 1.0 if len(extracted_set) == 0 else 0.5
            else:
                intersection = len(extracted_set.intersection(expected_set))
                precision = intersection / len(extracted_set) if len(extracted_set) > 0 else 0
                recall = intersection / len(expected_set) if len(expected_set) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                reward = f1 * 2.0  # Scale to match other rewards
            
            rewards.append(reward)
            
        except Exception as e:
            rewards.append(0.0)
    
    print(f"Entity correctness rewards: {rewards}")
    return rewards

def entity_type_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for extracting valid entity types"""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    # Valid entity types for different document types
    valid_entity_types = {
        'TenderType', 'TenderYear', 'TenderNumber', 'TenderCode', 
        'TenderPerson', 'TenderOrg', 'TenderTel', 'TenderFax', 'TenderDeadline',
        'OrderID', 'OrderCompanyName', 'OrderCompanyAddress', 'OrderTaxNumber', 
        'OrderDate', 'OrderPerson', 'TenderCompanyName', 'TenderCompanyAddress', 
        'TenderTaxNumber', 'BidId', 'BidCompanyName', 'BidCompanyAddress', 
        'BidTaxNumber', 'BidOrderDate', 'BidPerson'
    }
    
    for response in responses:
        extracted_json = extract_json_from_response(response)
        
        try:
            extracted_entities = json.loads(extracted_json)
            if not isinstance(extracted_entities, list):
                rewards.append(0.0)
                continue
            
            valid_types_count = 0
            total_entities = len(extracted_entities)
            
            for entity in extracted_entities:
                if isinstance(entity, dict):
                    for key in entity.keys():
                        if key in valid_entity_types:
                            valid_types_count += 1
            
            if total_entities == 0:
                reward = 0.5  # Neutral reward for empty but valid arrays
            else:
                reward = (valid_types_count / total_entities) * 1.0
            
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
        if length < 50:  # Too short
            reward = 0.2
        elif length < 200:  # Good length
            reward = 0.5
        elif length < 500:  # Acceptable
            reward = 0.3
        else:  # Too long
            reward = 0.1
        
        rewards.append(reward)
    
    return rewards


# --- LOAD AND SPLIT DATASET ---
try:
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"Dataset loaded from: {DATASET_PATH}")
except Exception as e:
    print(f"Error loading dataset from {DATASET_PATH}: {e}")
    exit()

# SPLIT
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
    ],
    args = training_args,
    train_dataset = processed_train_dataset,
)

# --- Start Training ---
print("\nStarting GRPO training...")
print("Note: You might see 0 rewards for the first 100+ steps. This is normal!")
trainer.train()

# --- Save Model ---
print(f"\nTraining completed. Saving GRPO fine-tuned model to: {MODEL_CHECKPOINT_DIR}")
model.save_lora(f"{MODEL_CHECKPOINT_DIR}/grpo_lora")

print("GRPO training finished successfully!")




# --- TEST THE MODEL ---
from vllm import SamplingParams

print("\n" + "="*50)
print("TESTING THE TRAINED MODEL")
print("="*50)

# Load the LoRA for inference
test_text = tokenizer.apply_chat_template([
    {"role": "user", "content": f"{TENDER_PROMPT}\nUser Input:\nchunk_id: 0\npostopek 165 2023 da/sp povabilo k oddaji ponudbe narocnik vabi ponudnike da v skladu z navodili ponudnikom izdelajo ponudbo za popravilo centrifuge heraeus cryofuge 6000 naziv aparata centrifuga proizvajalec heraeus tip cryofuge 6000 inv.st. kljuke za oddelcne lekarne do najkasneje 28.04.2023 do 12 ure. vodja nabavne sluzbe matjaz stinek.) univ.dipl.ekon\nOutput:\n"}
], tokenize = False, add_generation_prompt = True)


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