import torch
from unsloth import FastLanguageModel
#from transformers import AutoTokenizer # Anche se Unsloth carica il tokenizer, è utile per chiarezza
import os
import yaml
from glob import glob
import gc # Per la garbage collection

def merge_lora_checkpoints(
    config_file: str,
    output_merged_dir: str,
    num_checkpoints_to_average: int = 5,
    save_method: str = "merged_16bit",
    hf_token: str = None
):
    """
    Esegue il checkpoint averaging dei pesi LoRA da checkpoint multipli
    e merge il risultato nel modello base per salvare un modello completo.

    Args:
        config_file (str): Percorso al file di configurazione YAML (es. "config/mistral7B_instruct_v3.yml").
        output_merged_dir (str): Directory dove salvare il modello mergiato finale.
        num_checkpoints_to_average (int): Numero di checkpoint più recenti da mediare. Di default 5.
        save_method (str): Metodo di salvataggio per il modello mergiato ("merged_16bit", "merged_4bit", ecc.).
                           Di default "merged_16bit".
        hf_token (str, optional): Token di Hugging Face per l'autenticazione, se necessario per il modello base.
                                  Di default None.
    """
    print(f"Starting LoRA checkpoint averaging and merging process...")
    print(f"Configuration file: {config_file}")
    print(f"Output directory for merged model: {output_merged_dir}")
    print(f"Number of checkpoints to average: {num_checkpoints_to_average}")
    print(f"Save method: {save_method}")

    # --- Carica la configurazione ---
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from {config_file}")
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found. Please create it.")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}")
        return

    MODEL_NAME = config["model"]["name"]
    MODEL_CHECKPOINT_DIR = config["model_checkpoint_dir"] # Directory che contiene i checkpoint numerati

    # Carica le configurazioni per il modello
    MAX_SEQ_LENGTH = config["model"]["max_seq_length"]
    DTYPE = getattr(torch, config["model"]["dtype"]) # Converti stringa in tipo torch.dtype
    LOAD_IN_4BIT = config["model"]["load_in_4bit"]

    # --- 1. Trova tutti i checkpoint salvati ---
    checkpoint_paths = sorted(glob(os.path.join(MODEL_CHECKPOINT_DIR, "checkpoint-*")))

    if not checkpoint_paths:
        print(f"Error: No checkpoints found in {MODEL_CHECKPOINT_DIR}.")
        print("Please ensure `save_strategy` and `save_total_limit` are set in your training args in the config file.")
        return

    # Seleziona gli ultimi N checkpoint da mediare
    checkpoints_to_load = checkpoint_paths[-num_checkpoints_to_average:]
    if len(checkpoints_to_load) < num_checkpoints_to_average:
        print(f"Warning: Found only {len(checkpoints_to_load)} checkpoints, averaging all of them.")
    
    print(f"Selected checkpoints for averaging: {checkpoints_to_load}")

    # --- 2. Inizializza un dizionario per accumulare la somma dei pesi LoRA ---
    # Carichiamo il primo checkpoint solo per ottenere la struttura dei pesi LoRA
    print(f"Loading first checkpoint {checkpoints_to_load[0]} to get LoRA weights structure...")
    try:
        # Carica il modello base con l'adapter dal primo checkpoint
        # FastLanguageModel quando carica da un checkpoint LoRA, lo fa su un modello base "sotto il cofano"
        temp_model, _ = FastLanguageModel.from_pretrained(
            model_name=checkpoints_to_load[0],
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
            token=hf_token,
        )
    except Exception as e:
        print(f"Error loading initial checkpoint {checkpoints_to_load[0]}: {e}")
        return

    # Estrai i nomi dei parametri LoRA e inizializza la somma a zero
    # Inizializza il tensore somma per i pesi mediati
    sum_lora_weights = {}
    for name, param in temp_model.named_parameters():
        if "lora" in name:
            sum_lora_weights[name] = torch.zeros_like(param.data)

    del temp_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # --- 3. Carica i pesi LoRA da ogni checkpoint e somma ---
    for i, cp_path in enumerate(checkpoints_to_load):
        print(f"Processing checkpoint {i+1}/{len(checkpoints_to_load)}: {cp_path}")
        try:
            current_model, _ = FastLanguageModel.from_pretrained(
                model_name=cp_path,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=DTYPE,
                load_in_4bit=LOAD_IN_4BIT,
                token=hf_token,
            )
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {cp_path}. Skipping. Error: {e}")
            continue

        for name, param in current_model.named_parameters():
            if "lora" in name and name in sum_lora_weights:
                sum_lora_weights[name].add_(param.data.to(sum_lora_weights[name].device)) # Assicura che siano sullo stesso device

        del current_model # Rilascia memoria
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- 4. Calcola la media ---
    averaged_lora_weights = {}
    if not sum_lora_weights:
        print("Error: No LoRA weights were accumulated. Checkpoints might be empty or corrupted.")
        return

    num_successfully_loaded_checkpoints = len(checkpoints_to_load) # Se ci sono stati errori, potresti volerlo aggiornare

    for name, param_sum in sum_lora_weights.items():
        averaged_lora_weights[name] = param_sum / num_successfully_loaded_checkpoints

    # --- 5. Applica i pesi mediati a un nuovo modello base LoRA ---
    print(f"Loading base model '{MODEL_NAME}' to apply averaged LoRA weights.")
    try:
        model_for_merge, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
            token=hf_token,
        )
    except Exception as e:
        print(f"Error loading base model {MODEL_NAME}: {e}")
        return

    # Applica PEFT per creare la struttura dell'adapter sul modello base
    model_for_merge = FastLanguageModel.get_peft_model(
        model_for_merge,
        r = config["peft"]["r"],
        target_modules = config["peft"]["target_modules"],
        lora_alpha = config["peft"]["lora_alpha"],
        lora_dropout = config["peft"]["lora_dropout"],
        bias = config["peft"]["bias"],
        use_gradient_checkpointing = config["peft"]["use_gradient_checkpointing"],
        random_state = config["peft"]["random_state"],
        max_seq_length = MAX_SEQ_LENGTH,
    )

    # Sostituisci i pesi dell'adapter con quelli mediati
    print("Injecting averaged LoRA weights into the model's adapter.")
    for name, param in model_for_merge.named_parameters():
        if "lora" in name and name in averaged_lora_weights:
            # Assicurati che il tensore destinazione sia scrivibile
            param.data.copy_(averaged_lora_weights[name].to(param.device)) # Copia i dati


    # --- 6. Esegui il merge finale e salva ---
    os.makedirs(output_merged_dir, exist_ok=True)
    print(f"Performing final merge of averaged adapter into base model and saving to: {output_merged_dir}")
    try:
        model_for_merge.save_pretrained(output_merged_dir, save_method=save_method)
        tokenizer.save_pretrained(output_merged_dir)
        print(f"Averaged and merged model saved successfully!")
    except Exception as e:
        print(f"Error saving merged model: {e}")
        return
    
    del model_for_merge
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("LoRA checkpoint averaging and merging process completed!")

# Questo blocco viene eseguito solo se il file viene eseguito direttamente
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv() # Carica le variabili d'ambiente (es. HF_TOKEN)

    # Parametri di esempio (dovresti adattarli ai tuoi)
    CONFIG_FILE_PATH = "config/mistral7B_instruct_v3.yml" # Il tuo file di configurazione
    OUTPUT_MERGED_MODEL_DIR = "merged_model_checkpoint_averaged" # Dove verrà salvato il modello finale mergiato

    # Assicurati di avere il token HF se il modello base non è pubblico
    HF_TOKEN = os.environ.get("hf_token")

    # Esegui la funzione di merging
    merge_lora_checkpoints(
        config_file=CONFIG_FILE_PATH,
        output_merged_dir=OUTPUT_MERGED_MODEL_DIR,
        num_checkpoints_to_average=8, # Puoi modificare questo valore
        save_method="merged_16bit",
        hf_token=HF_TOKEN
    )