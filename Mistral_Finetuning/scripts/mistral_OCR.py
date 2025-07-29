import os
import json
import re
import unicodedata
import base64
from dotenv import load_dotenv
from transformers import AutoTokenizer
import spacy
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# ============================
# 📌 PARAMETRI MODULARI
# ============================
YEAR_FILTER = "2023"
FILENAME_KEYWORD = "ORDER"
MAX_TOKENS = 1200
SPACY_MODEL = "sl_core_news_sm"

# Assicurati che queste directory esistano o modificale secondo le tue esigenze
# input_root ora può contenere sia PDF che DOCX
input_root = "/home/tiziano/GLiNER_fine_tuned/model/finetuning_data/sl/bid/raw_slovenian_DOCS_data" # <-- MODIFICATO: ora per PDF/DOCX
output_jsonl = "/home/tiziano/GLiNER_fine_tuned/model/finetuning_data/sl/order/train/data/2023_mistral_ocr_processed.jsonl"

# Estensioni e MIME types supportati
SUPPORTED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # Aggiungi altri se necessario e supportati da Mistral
}

# Recupera il token Hugging Face e la chiave API Mistral dalle variabili d'ambiente
hf_token = os.getenv("HF_TOKEN")
mistral_api_key = os.getenv("MISTRAL_API_KEY")

if not hf_token:
    print("ATTENZIONE: Il token Hugging Face (HF_TOKEN) non è stato trovato nel file .env. Alcune operazioni potrebbero fallire.")
if not mistral_api_key:
    print("ERRORE: La chiave API Mistral (MISTRAL_API_KEY) non è stata trovata nel file .env. L'OCR non funzionerà.")
    mistral_client = None
else:
    try:
        mistral_client = MistralClient(api_key=mistral_api_key)
    except Exception as e:
        print(f"Errore durante l'inizializzazione del client Mistral: {e}")
        mistral_client = None

# ============================
# 📦 FUNZIONI
# ============================

# Inizializzazione del tokenizzatore Hugging Face
try:
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", token=hf_token)
except Exception as e:
    print(f"Errore durante l'inizializzazione del tokenizzatore: {e}")
    print("Assicurati che il token HF sia valido e che il modello esista.")
    tokenizer = None

# Caricamento del modello spaCy
try:
    nlp = spacy.load(SPACY_MODEL)
except Exception as e:
    print(f"Errore durante il caricamento del modello spaCy '{SPACY_MODEL}': {e}")
    print("Assicurati che il modello sia installato (es. python -m spacy download sl_core_news_sm).")
    nlp = None

def process_text(text: str) -> str:
    """
    Funzione di pre-processing del testo con normalizzazione e pulizia.
    """
    try:
        text = text.lower()
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        text = re.sub(r"[<\[\]>\|_,:]+", " ", text)
        text = re.sub(r"\n+", " ", text)
        text = text.replace("\t", " ")
        text = text.replace("\f", " ")
        text = re.sub(r" {2,}", " ", text)
        return text.strip()
    except Exception as e:
        print(f"Errore durante il pre-processing del testo: {e}")
        return text

def smart_chunk(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    """
    Suddivide il testo in chunk basati sulla segmentazione delle frasi di spaCy
    e sulla lunghezza dei token del tokenizer Hugging Face.
    """
    if nlp is None:
        print("Errore: Modello spaCy non inizializzato. Impossibile effettuare il chunking per frasi.")
        return [text] if text else []
    if tokenizer is None:
        print("Errore: Tokenizzatore Hugging Face non inizializzato. Impossibile misurare la lunghezza dei token.")
        return [text] if text else []

    doc = nlp(text)
    chunks = []
    current_chunk_sentences = []
    current_chunk_token_length = 0

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        sent_tokens = tokenizer.encode(sent_text, add_special_tokens=False)
        sent_token_length = len(sent_tokens)

        if current_chunk_token_length + sent_token_length > max_tokens and current_chunk_sentences:
            merged_chunk_text = " ".join(current_chunk_sentences)
            chunks.append(merged_chunk_text)
            
            current_chunk_sentences = [sent_text]
            current_chunk_token_length = sent_token_length
        else:
            current_chunk_sentences.append(sent_text)
            current_chunk_token_length += sent_token_length
            
    if current_chunk_sentences:
        merged_chunk_text = " ".join(current_chunk_sentences)
        chunks.append(merged_chunk_text)

    return chunks

def is_valid_file_for_ocr(filename):
    """Verifica se il nome del file è valido in base ai criteri definiti (keyword ed estensione supportata)."""
    if FILENAME_KEYWORD not in filename:
        return False
    
    file_extension = os.path.splitext(filename)[1].lower()
    return file_extension in SUPPORTED_EXTENSIONS # <-- MODIFICATO: Controlla se l'estensione è supportata

def encode_file_to_base64(file_path: str) -> str:
    """Codifica un file in una stringa Base64."""
    with open(file_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode('utf-8')
    return encoded_string

def get_mime_type(file_path: str) -> str | None:
    """Restituisce il MIME type basandosi sull'estensione del file."""
    file_extension = os.path.splitext(file_path)[1].lower()
    return SUPPORTED_EXTENSIONS.get(file_extension) # Ottiene il MIME type dalla mappa

def extract_text_from_file_with_mistral_ocr(file_path: str) -> str | None:
    """
    Funzione per estrarre testo da un file (PDF, DOCX, PPTX) usando l'API OCR di Mistral.
    """
    if mistral_client is None:
        print("Client Mistral non inizializzato. Impossibile eseguire l'OCR.")
        return None

    mime_type = get_mime_type(file_path)
    if not mime_type:
        print(f"Errore: Tipo di file non supportato per l'OCR Mistral: {file_path}")
        return None

    try:
        # Codifica il file in Base64
        base64_file = encode_file_to_base64(file_path)
        
        # Prepara il formato del documento per l'API Mistral OCR
        document = {
            "type": "document_url",
            "document_url": f"data:{mime_type};base64,{base64_file}" # <-- Usa il MIME type dinamico
        }

        print(f"Inizio OCR per {file_path} (Tipo: {mime_type}) con Mistral AI...")
        ocr_response = mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document=document
        )

        extracted_text = ocr_response.text_content
        print(f"OCR completato per {file_path}.")
        return extracted_text

    except Exception as e:
        print(f"Errore durante l'estrazione OCR del file {file_path} con Mistral AI: {e}")
        return None

def process_directory(root_dir: str, output_path: str):
    """
    Elabora i file supportati in una directory specificata, applica l'OCR, il pre-processing,
    il chunking e salva i risultati in un file JSONL.
    """
    print(f"Inizio elaborazione della directory: {root_dir}")
    processed_files_count = 0
    with open(output_path, "w", encoding="utf-8") as outfile:
        for folder_name in os.listdir(root_dir):
            if YEAR_FILTER not in folder_name:
                continue

            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)
                if not os.path.isdir(subfolder_path):
                    continue

                for filename in os.listdir(subfolder_path):
                    if not is_valid_file_for_ocr(filename): # Controlla ora tutte le estensioni supportate
                        print(f"Saltando file non supportato o non corrispondente ai criteri: {filename}")
                        continue

                    file_path = os.path.join(subfolder_path, filename)
                    
                    # Chiamata all'OCR di Mistral per estrarre il testo dal file (PDF, DOCX, ecc.)
                    content = extract_text_from_file_with_mistral_ocr(file_path)
                    if content is None:
                        print(f"Saltando file {filename} a causa di un errore di estrazione OCR.")
                        continue

                    try:
                        processed_content = process_text(content)
                    except Exception as e:
                        print(f"Errore durante il pre-processamento del contenuto OCR di {file_path}: {e}")
                        continue

                    file_id = os.path.splitext(filename)[0]
                    smart_chunks = smart_chunk(processed_content)
                    
                    chunks_saved_for_file = 0

                    for i, chunk in enumerate(smart_chunks):
                        if chunks_saved_for_file >= 2:
                            break

                        if tokenizer:
                            num_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
                        else:
                            num_tokens = -1 

                        record = {
                            "id": file_id,
                            "chunk": i,
                            "text": chunk,
                            "num_tokens": num_tokens
                        }
                        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                        chunks_saved_for_file += 1
                    
                    processed_files_count += 1
    print(f"✅ Completato: {processed_files_count} file elaborati (tramite Mistral OCR) e salvati in {output_path}")

# ============================
# 🚀 MAIN
# ============================
if __name__ == "__main__":
    
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    process_directory(input_root, output_jsonl)