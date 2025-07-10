import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
from dotenv import load_dotenv
import os
import json

# Importazioni per il fine-tuning standard (senza Unsloth)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# === Load ENV Variables ===

load_dotenv()
hf_token = os.environ.get("hf_token")

# Scegli il modello Mistral che vuoi fine-tunare
model_name = "mistralai/Mistral-7B-Instruct-v0.3" 

# --- CONFIGURAZIONE QLORA SENZA UNSLOTH ---
# Definisci il tipo di dati per l'addestramento (bfloat16 è preferibile se supportato, altrimenti float16)
max_seq_length = 4096 # Lunghezza massima delle sequenze (puoi aumentarla se hai VRAM sufficiente)

# 1. Configurazione della Quantizzazione (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                 # Carica il modello in 4 bit
    bnb_4bit_quant_type="nf4",         # Tipo di quantizzazione (NormalFloat 4)
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # Tipo di dati per la computazione
    bnb_4bit_use_double_quant=True,    # Usa la doppia quantizzazione
)

# 2. Caricamento del modello e del tokenizer
# Per caricare il modello quantizzato, passare la bnb_config
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # Importante per la compatibilità con bnb_4bit_compute_dtype
    device_map="auto",                 # Distribuisce il modello automaticamente sulle GPU disponibili
    token=hf_token,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

# Aggiungi un token pad se il tokenizer non ne ha uno (comune con alcuni modelli)
# Questo è importante per l'impacchettamento del dataset e l'allineamento dei batch
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # O un altro token appropriato

# 3. Preparazione del modello per il K-bit training (QLoRA)
# Abilita il gradient checkpointing e converte i moduli per il training
model = prepare_model_for_kbit_training(model)

# 4. Configurazione LoRA
lora_config = LoraConfig(
    r=16,                                    # Rank LoRA
    lora_alpha=16,                           # Scalatura per LoRA
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Moduli da "loraficare"
    lora_dropout=0.05,                       # Dropout per LoRA (un po' di dropout è spesso utile)
    bias="none",                             # Nessun bias per LoRA
    task_type="CAUSAL_LM",                   # Tipo di task (Generazione di testo)
)

# 5. Ottieni il modello LoRA
model = get_peft_model(model, lora_config)

# Stampa la percentuale di parametri addestrabili (utile per debug)
model.print_trainable_parameters()

# --- FINE CONFIGURAZIONE QLORA SENZA UNSLOTH ---


# Esempio di dataset (devrai sostituirlo con il tuo)
# Questo è il formato per l'addestramento "instruction-following"
# NOTA: Ho rimosso l'assegnazione multipla `alpaca_prompt = full_prompt_template`
# e uso solo `full_prompt_template` per chiarezza.
full_prompt_template = """<SCENARIO>
This is a Named Entity Recognition (NER) task on noisy, partial OCR-derived chunk of text.
Not invent any entity, and if you don't recognize one or more entities, skip them, it can be found in next chunks.
Extract specified entities if happens, and eventually correcting them to match the schema below.
Unique fields cannot be present multiple times in output.
Persons goods and services can appear multiple times.
</SCENARIO>

<SCHEMA>
- TenderPurchasedGoodsServices: free-text description of the good(s) or service(s) being acquired, preserve OCR text
- TenderContractType: one character among n, m, x. Usually is not present in the text.
    {n = non medical goods, m = medical goods, x = services}
    you have to decide this symbol based on goods/services types that eventually occur in chunks, if no good/services appear, ignore this field.
- TenderOrg: free-text name of the oganization issuing the tender, preserve OCR text
- TenderTel and TenderFax: phone numbers, follow the format in the examples (e.g. "(03) 42 33 000").
- TenderDeadline: date in format dd.mm.yyyy
- TenderContactPerson: name like "firstname lastname", it may appear "firstname.lastnameOtherWords" (correct any dots or irregularities)
- TenderProcedureNumber: string int between 0 and 999, delete zeros in front, e.g. "009" becomes '9'
- TenderProcedureCode: code in the format aa/bb or aaa/bb
- TenderYear: four-digit year positioned usually near the procedure number and Procedure Code.
                If you don't detect TenderProcedureNumber, don't extract any TenderYear enitity.
</SCHEMA>

<FORMAT>
Output Format: List[Dict[str, str]]]) or empty list.
Return:
[] if no relevant entity is found
Otherwise, a list of dictionaries, add one dict for eah schema's entity tha you find, if no entity detected in current chunk, skip it.
[
    { "TenderProcedureNumber": "309" },
    { "TenderDeadline":"28.04.2023" },
    { "TenderPurchasedGoodsServices":"str1" },
    { "TenderPurchasedGoodsServices":"str2" },
    ...
]
</FORMAT>

<EXAMPLES>

Input:
splosna bolnisnica celje oblakova ulica 5 3000 celje tel (03) 42 33 000 fax: (03) 42 33 757 postopek 309 2022 da/vh povabilo k oddaji ponudbe narocnik vabi ponudnike
Output:
[
    { "TenderOrg": "splosna bolnisnica celje" },
    { "TenderTel": "(03) 42 33 000" },
    { "TenderProcedureNumber": "309" },
    { "TenderYear": "2022" },
    { "TenderProcedureCode": "da/vh" },
]

Example 1
Input:
postopek 165 2023 da/sp povabilo k oddaji ponudbe narocnik vabi ponudnike da v skladu z navodili ponudnikom izdelajo ponudbo za popravilo centrifuge heraeus cryofuge 6000 naziv aparata centrifuga proizvajalec heraeus tip cryofuge 6000 inv.st. kljuke za oddelcne lekarne do najkasneje 28.04.2023 do 12 ure. vodja nabavne sluzbe matjaz stinek.) univ.dipl.ekon
Output:
[
    { "TenderProcedureNumber": "165" },
    { "TenderYear": "2023" },
    { "TenderProcedureCode": "da/sp" },
    { "TenderDeadline": "28.04.2023" },
    { "TenderContactPerson": "matjaz stinek" },
    { "TenderPurchasedGoodsServices": "popravilo centrifuge heraeus cryofuge 6000" },
    { "TenderContractType": "x"} 
]

Example 2 (with dotted name, spacing in deadline, single non medical good)
Input:
en n 123 2022 sp/hv rok za oddajo 10. 07. 2024 kontaktna oseba ana.novak odgovorna oseba ana.novak predmet javnega narocila: dobava pisarniskega materiala tonerjev e pisal
Output:
[
    { "TenderProcedureNumber": "123" },
    { "TenderYear": "2022" },
    { "TenderProcedureCode": "sp/hv" },
    { "TenderDeadline": "10.07.2024" },
    { "TenderContactPerson": "ana novak" },
    { "TenderPurchasedGoodsServices": "dobava pisarniskega materiala tonerjev e pisal" },
    { "TenderContractType": "n" }
]

Example 3
Input:
dokumentacija za postopek 045 2023 mvk/vh mora biti predlozena pravocasno za dobavo ultrazvocnih gelov
Output:
[
    { "TenderProcedureNumber": "45" },
    { "TenderYear":"2023" },
    { "TenderProcedureCode":"mvk/vh" },
    { "TenderPurchasedGoodsServices":"dobavo ultrazvocnih gelov" }
    { "TenderContractType": "m" }
]

Example 4 (no one match)
Input:
navodila za uporabo opreme so navedena v prilozenem dokumento. pred montazo jih natancno preberite
Output:
    []

Example 5 (no ID, multiple non-contiguous goods/services)
Input:
kontaktna oseba luka.zajc predmet javnega narocila je najem tiskalnikov za obdobje treh anni. poleg questo sarà fatto anche consegna di cartucce e carta per tutti i reparti.
Output:
[
    { "TenderContactPerson": "luka zajc" },
    { "TenderPurchasedGoodsServices": "najem tiskalnikov za obdobje treh let" },
    { "TenderPurchasedGoodsServices": "dobava kartus e papirja za vse oddelke" },
    { "TenderContractType": "n" }
]
</EXAMPLES>

Input:
{}
Output:
{}"""

# Funzione di formattazione per il tuo dataset
def format_ner_example(example):
    input_text = example["text"]
    
    # Filtra i valori null da ogni dizionario di entità
    cleaned_ner_list = []
    for entity_dict in example["ner"]:
        cleaned_entity_dict = {k: v for k, v in entity_dict.items() if v is not None}
        if cleaned_entity_dict: # Assicurati che il dizionario non sia vuoto dopo il filtraggio
            cleaned_ner_list.append(cleaned_entity_dict)
    
    # Se la lista finale è vuota, assicurati che output_json_string sia "[]"
    if not cleaned_ner_list:
        output_json_string = "[]"
    else:
        # Converte la lista filtrata in una stringa JSON
        output_json_string = json.dumps(cleaned_ner_list, ensure_ascii=False, separators=(',', ':'))

    formatted_text = (
        full_prompt_template +
        f"\nInput:\n{input_text}\nOutput:\n{output_json_string}" +
        tokenizer.eos_token
    )
    
    return {"text": formatted_text}


# Carica il tuo dataset JSONL
# Assicurati che 'your_dataset.jsonl' sia il nome corretto del tuo file
dataset = load_dataset("json", data_files="/home/tiziano/langgraph_agents/Finetuning/data/gemini_tender_ner_dataset.jsonl", split="train")


# --- DATASET PREPARATION ---

# step 1: 90% train, 10% per validation + test
train_temp_split = dataset.train_test_split(test_size=0.10, seed=42)
train_dataset = train_temp_split["train"]
temp_dataset = train_temp_split["test"] # Questo è il 10% che verrà ulteriormente diviso

# step 2: (5% validation, 5% test)
eval_test_split = temp_dataset.train_test_split(test_size=0.50, seed=42) # 0.50 di 0.10 è 0.05
eval_dataset = eval_test_split["train"] # Ora questo è il 5% per la validazione
test_dataset = eval_test_split["test"]   # E questo è il 5% per il test finale

print(f"Dimensione del Training Dataset: {len(train_dataset)} esempi")
print(f"Dimensione del Validation Dataset: {len(eval_dataset)} esempi")
print(f"Dimensione del Test Dataset: {len(test_dataset)} esempi")

# Save test data to use it after trianing
output_test_file_path = "/home/tiziano/langgraph_agents/Finetuning/data/gemini_tender_ner_test_split.jsonl"
processed_test_dataset = test_dataset.map(format_ner_example, batched=False)
processed_test_dataset.to_json(output_test_file_path, orient="records", lines=True, force_ascii=False)
# Applica la funzione di formattazione al dataset di training E di validazione
processed_train_dataset = train_dataset.map(format_ner_example, batched=False)
processed_eval_dataset = eval_dataset.map(format_ner_example, batched=False) # Formatta anche il set di validazione

print(f"Test dataset salvato in: {output_test_file_path}")
print("Esempio di riga del dataset di training formattato:")
print(processed_train_dataset[0]["text"])
exit(0)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = processed_train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    # packing = True è molto utile per l'efficienza, specialmente con lunghezze variabili
    # ma potrebbe causare problemi se il dataset è molto piccolo o con formattazione particolare
    packing = True, 
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "paged_adamw_8bit", # Usiamo paged_adamw_8bit per una migliore gestione della memoria
        seed = 3407,
        # Aggiungi queste righe per salvare il modello in formato LoRA, più leggero
        save_strategy="epoch", # Salva a ogni epoca
        save_steps=500, # O ogni 500 step, scegli quello che preferisci
        save_total_limit=3, # Mantieni solo gli ultimi 3 checkpoint
    ),
)

# Avvia l'addestramento
trainer.train()

# --- Opzionale: Salvataggio del modello LoRA addestrato ---
# Puoi salvare solo gli adapter LoRA, che sono molto più piccoli dell'intero modello.
# Saranno necessari il modello base e gli adapter per l'inferenza.
# model.save_pretrained("mistral_7b_ner_finetuned") 
# tokenizer.save_pretrained("mistral_7b_ner_finetuned")

# --- Opzionale: Unione degli adapter al modello base per deployment (richiede molta VRAM) ---
# Se hai abbastanza VRAM, puoi unire gli adapter al modello base per un modello standalone.
# from peft import AutoPeftModelForCausalLM
# from transformers import AutoTokenizer

# model_path = "./mistral_7b_ner_finetuned" # Il percorso dove hai salvato gli adapter
# model = AutoPeftModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# merged_model_path = "./mistral_7b_ner_merged"
# model.save_pretrained(merged_model_path, safe_serialization=True)
# tokenizer.save_pretrained(merged_model_path)