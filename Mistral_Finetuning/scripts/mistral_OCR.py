import os
import json
import re
import unicodedata
import base64 # Necessario per la codifica Base64 dei PDF
from dotenv import load_dotenv
from transformers import AutoTokenizer
import spacy
from mistralai.client import MistralClient # Importa il client Mistral
from mistralai.models.chat_completion import ChatMessage # Potrebbe essere utile per altri usi, ma non strettamente per OCR

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# ============================
# 📌 PARAMETRI MODULARI
# ============================
YEAR_FILTER = "2023"          # Nome cartella principale (es. contained in "2023")
FILENAME_KEYWORD = "ORDER"    # Sottostringa che deve apparire nel nome del file
MAX_TOKENS = 1200             # Max token per chunk
SPACY_MODEL = "sl_core_news_sm" # Modello spaCy per la segmentazione delle frasi

# Assicurati che queste directory esistano o modificale secondo le tue esigenze
input_root = "/home/tiziano/GLiNER_fine_tuned/model/finetuning_data/sl/bid/raw_slovenian_PDF_data" # Directory contenente i PDF
output_jsonl = "/home/tiziano/GLiNER_fine_tuned/model/finetuning_data/sl/order/train/data/2023_mistral_ocr_processed.jsonl"

# Recupera il token Hugging Face e la chiave API Mistral dalle variabili d'ambiente
hf_token = os.getenv("hf_token")
mistral_api_key = os.getenv("mistral_token")

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
        # Conversione minuscola
        text = text.lower()

        # Normalizzazione dei caratteri con diacritici (e.g., "Jöhn" -> "john")
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

        # Sostituisce trattini, underscore, virgole e due punti con spazi
        text = re.sub(r"[<\[\]>\|_,:]+", " ", text)

        # Pulizia input OCR da sequenze anomale
        text = re.sub(r"\n+", " ", text)
        text = text.replace("\t", " ")
        text = text.replace("\f", " ")
        text = re.sub(r" {2,}", " ", text) # Rimuove spazi multipli

        return text.strip()
    except Exception as e:
        print(f"Errore durante il pre-processing del testo: {e}")
        return text # Restituisce il testo originale in caso di errore

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
        if not sent_text: # Salta frasi vuote
            continue

        sent_tokens = tokenizer.encode(sent_text, add_special_tokens=False)
        sent_token_length = len(sent_tokens)

        # Se l'aggiunta della frase corrente supera il limite di token,
        # finalizza il chunk precedente e inizia un nuovo chunk.
        # Si assume che una singola frase non sia più lunga di MAX_TOKENS.
        if current_chunk_token_length + sent_token_length > max_tokens and current_chunk_sentences:
            # Decodifica le frasi accumulate nel chunk
            merged_chunk_text = " ".join(current_chunk_sentences)
            chunks.append(merged_chunk_text)
            
            # Inizia un nuovo chunk con la frase corrente
            current_chunk_sentences = [sent_text]
            current_chunk_token_length = sent_token_length
        else:
            # Aggiungi la frase al chunk corrente
            current_chunk_sentences.append(sent_text)
            current_chunk_token_length += sent_token_length
            
    # Aggiungi l'ultimo chunk se non è vuoto
    if current_chunk_sentences:
        merged_chunk_text = " ".join(current_chunk_sentences)
        chunks.append(merged_chunk_text)

    return chunks

def is_valid_file_for_ocr(filename):
    """Verifica se il nome del file è valido in base ai criteri definiti per i PDF."""
    return FILENAME_KEYWORD in filename and filename.lower().endswith(".pdf")

def encode_pdf_to_base64(pdf_path: str) -> str:
    """Codifica un file PDF in una stringa Base64."""
    with open(pdf_path, "rb") as pdf_file:
        encoded_string = base64.b64encode(pdf_file.read()).decode('utf-8')
    return encoded_string

def extract_text_from_pdf_with_mistral_ocr(pdf_path: str) -> str | None:
    """
    Funzione per estrarre testo da un PDF usando l'API OCR di Mistral.
    """
    if mistral_client is None:
        print("Client Mistral non inizializzato. Impossibile eseguire l'OCR.")
        return None

    try:
        # Codifica il PDF in Base64
        base64_pdf = encode_pdf_to_base64(pdf_path)
        
        # Prepara il formato del documento per l'API Mistral OCR
        document = {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{base64_pdf}"
        }

        print(f"Inizio OCR per {pdf_path} con Mistral AI...")
        ocr_response = mistral_client.ocr.process(
            model="mistral-ocr-latest", # Usa il modello OCR più recente
            document=document
        )

        # L'output dell'OCR di Mistral è in formato Markdown
        # Potrebbe essere necessario un parsing più sofisticato se la struttura è importante,
        # ma per l'estrazione del testo grezzo, di solito è sufficiente prendere il campo 'text'.
        extracted_text = ocr_response.text_content
        print(f"OCR completato per {pdf_path}.")
        return extracted_text

    except Exception as e:
        print(f"Errore durante l'estrazione OCR del PDF {pdf_path} con Mistral AI: {e}")
        # Gestione di errori specifici dell'API Mistral (es. rate limiting, file troppo grandi)
        # potresti voler aggiungere una logica di retry qui
        return None

def process_directory(root_dir: str, output_path: str):
    """
    Elabora i file PDF in una directory specificata, applica l'OCR, il pre-processing,
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
                    if not is_valid_file_for_ocr(filename):
                        continue

                    file_path = os.path.join(subfolder_path, filename)
                    
                    # Chiamata all'OCR di Mistral per estrarre il testo dal PDF
                    content = extract_text_from_pdf_with_mistral_ocr(file_path)
                    if content is None:
                        print(f"Skipping file {filename} due to Mistral OCR extraction error.")
                        continue

                    try:
                        # Applica il pre-processing al contenuto estratto dall'OCR
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
    
    # Crea le directory di output se non esistono
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    process_directory(input_root, output_jsonl)