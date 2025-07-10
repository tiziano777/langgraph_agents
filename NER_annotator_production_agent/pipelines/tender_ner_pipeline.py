import os
import traceback
import yaml
from typing import List, Dict, Any,Set
from dotenv import load_dotenv

import spacy
from transformers import AutoTokenizer
from utils.MistralLoader import MistralLoader 

from langgraph.graph import START, END, StateGraph
from states.ner_state import State

from nodes.preprocessing.ner_Preprocessing import Preprocessor
from nodes.annotators.ner_hfAnnotator import Annotator  
from nodes.evaluators.ner_tender_check import Formatter

from utils.logger_config import setup_pipeline_logger

logger = setup_pipeline_logger(log_filename_prefix='tender_ner_pipeline')

# === LOCAL MODEL SETUP (MISTRAL) ===

MISTRAL_PATH = "config/config_finetuned_mistral7B-0.3.yml"
with open(MISTRAL_PATH, "r", encoding="utf-8") as f:
    llm_config = yaml.safe_load(f)
logger.info(f"Configurazione caricata da {MISTRAL_PATH}")

hf_token=os.environ.get("hf_token")
llm_config['hf_token']=hf_token

#############################################################################

# ---- DATA OPERATIONS: CHUNKING BY TOKENS, MERGING OUTPUT ----

MAX_TOKENS=1150
nlp = None
global_tokenizer = None

try:
    nlp = spacy.load("sl_core_news_sm")
    logger.info("Modello spaCy caricato con successo per il chunking.")
except Exception as e:
    logger.error(f"Errore durante il caricamento del modello spaCy: {e}. Il chunking per frasi potrebbe non essere ottimale.")

# Carica il token HF una volta all'avvio del modulo
load_dotenv()
hf_token = os.getenv("hf_token")

try:
    if hf_token:
        global_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", token=hf_token)
        logger.info("Tokenizzatore Hugging Face (Mistral-7B-Instruct-v0.3) caricato con successo.")
    else:
        logger.warning("Token HF non trovato. Impossibile caricare il tokenizzatore per il conteggio preciso dei token.")
except Exception as e:
    logger.error(f"Errore durante l'inizializzazione del tokenizzatore Mistral-7B-Instruct-v0.3: {e}")
    logger.error("Assicurati che il token HF sia valido e che il modello esista.")

### AUX FUNCTIONS ###

def count_tokens(text: str) -> int:
    """
    Conta il numero di token in un dato testo utilizzando il tokenizzatore Mistral-7B-Instruct-v0.3
    (caricato globalmente).

    Args:
        text (str): Il testo di cui contare i token.

    Returns:
        int: Il numero di token nel testo. Restituisce -1 se il tokenizzatore non è disponibile.
    """
    if global_tokenizer is None:
        logger.error("Tokenizzatore globale non disponibile per il conteggio dei token.")
        return -1

    tokens = global_tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def smart_chunk(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    """
    Suddivide il testo in chunk basati sulla segmentazione delle frasi di spaCy
    e sulla lunghezza dei token del tokenizer fornito.
    """
    if nlp is None:
        logger.error("Errore: Modello spaCy non inizializzato. Impossibile effettuare il chunking per frasi.")
        return [text] if text else []
    if global_tokenizer is None:
        logger.error("Errore: Tokenizzatore globale non inizializzato. Impossibile misurare la lunghezza dei token.")
        return [text] if text else []

    doc = nlp(text)
    chunks = []
    current_chunk_sentences = []
    current_chunk_token_length = 0

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        sent_token_length = count_tokens(sent_text)
        if sent_token_length == -1: # Errore nel conteggio token, gestisci di conseguenza
            logger.warning(f"Errore nel conteggio token per la frase: '{sent_text[:50]}...'. Saltata o gestita diversamente.")
            continue

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

def merge_ner_refined_results(chunked_results: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Aggrega i risultati 'ner_refined' da una lista di chunked_results in un singolo dizionario.
    Il merging inizia dal dizionario del primo chunk (chunk_id = 0). Le coppie chiave:valore
    dai chunk successivi vengono aggiunte solo se la chiave non è già presente nel dizionario aggregato.

    :param chunked_results: Una lista di dizionari, ognuno contenente 'chunk_id' e 'ner_refined'.
    :return: Un singolo dizionario che rappresenta l'aggregazione di tutti i 'ner_refined'.
    """
    if not chunked_results:
        logger.info("Lista di risultati chunked_results vuota, restituendo un dizionario vuoto.")
        return {}

    # Ordina i risultati per chunk_id per assicurarsi che il chunk 0 sia il primo
    sorted_results = sorted(chunked_results, key=lambda x: x.get('chunk_id', 0))

    aggregate_result: Dict[str, str] = {}
    if sorted_results and 'ner_refined' in sorted_results[0] and isinstance(sorted_results[0]['ner_refined'], dict):
        aggregate_result = sorted_results[0]['ner_refined'].copy()
        logger.debug(f"Inizializzato l'aggregazione con il chunk 0: {aggregate_result}")
    else:
        logger.warning("Il primo chunk non contiene un dizionario 'ner_refined' valido, iniziando con un dizionario vuoto.")

    for i in range(1, len(sorted_results)):
        chunk_result = sorted_results[i]
        if 'ner_refined' in chunk_result and isinstance(chunk_result['ner_refined'], dict):
            current_ner_refined = chunk_result['ner_refined']
            for key, value in current_ner_refined.items():
                if key not in aggregate_result:
                    aggregate_result[key] = value
                    logger.debug(f"Aggiunta la chiave '{key}' dal chunk {chunk_result.get('chunk_id')} al risultato aggregato.")
                else:
                    logger.debug(f"Chiave '{key}' già presente nel risultato aggregato, saltata dal chunk {chunk_result.get('chunk_id')}.")
        else:
            logger.warning(f"Chunk {chunk_result.get('chunk_id')} non contiene un dizionario 'ner_refined' valido, saltato nel merging.")
            
    logger.info(f"Merging dei risultati 'ner_refined' completato. Risultato aggregato: {aggregate_result}")
    return aggregate_result

####################################################### 

### PIPELINE LOGIC ###

def create_pipeline(preprocessor: Preprocessor, annotator: Annotator, formatter: Formatter):
    """
    Crea e compila la pipeline LangGraph con un loop tra ner_refiner e ner_schema_validator.
    """
    workflow = StateGraph(State)

    workflow.add_node("preprocessor_node", preprocessor)
    workflow.add_node("annotator_node", annotator)
    workflow.add_node("format_node", formatter)

    workflow.add_edge(START, "preprocessor_node")
    workflow.add_edge("preprocessor_node", "annotator_node")
    workflow.add_edge("annotator_node", "format_node")
    workflow.add_edge("format_node", END)

    pipeline = workflow.compile()

    ### CREATE IMAGE FOR VISUALIZE PIPELINE ###
    # uncomment wheen decide to change the graph pipeline
    '''
    try:
        graphImage = pipeline.get_graph().draw_mermaid_png()
        # Rinomina l'immagine del grafo per riflettere l'uso del modello locale
        with open("images/tender_pipeline.png", "wb") as f:
            f.write(graphImage)
        logger.info("Salvata immagine del grafo in images/tedner_pipeline.png")
    except Exception as e:
        logger.warning(f"Errore durante la generazione del grafo: {e}")
        logger.debug(traceback.format_exc())
    '''

    return pipeline

def run_pipeline(long_text: str) -> Dict[str, str]:
    """
    Esegue la pipeline di elaborazione NER su un singolo testo lungo,
    applicando il chunking, appiattendo i risultati di ogni chunk e poi aggregandoli.

    :param long_text: Il testo lungo da elaborare.
    
    :return: Un unico dizionario che rappresenta l'aggregazione di tutte le entità dei chunk.
    """
    
    logger.info("Inizializzazione del modello LLM Mistral locale...")
    llm = MistralLoader(llm_config, hf_token) 
    logger.info("Modello LLM Mistral locale inizializzato e pronto per l'uso.")

    eligible_keys_for_flatten: Set[str] = {"TenderType","TenderNumber","TenderCode","TenderYear","TenderOrg","TenderTel","TenderFax","TenderDeadline","TenderPerson"}

    preprocessor = Preprocessor()
    annotator = Annotator(llm=llm)
    formatter = Formatter(eligible_keys_for_flatten)

    graph = create_pipeline(preprocessor, annotator, formatter)

    if not long_text:
        logger.warning("Testo in input vuoto, nessuna elaborazione.")
        return {}

    chunks = smart_chunk(long_text)
    logger.info(f"Testo diviso in {len(chunks)} chunks.")

    chunked_ner_refined_results: List[Dict[str, Any]] = []

    for chunk_id, chunk_text in enumerate(chunks):
        try:
            initial_state = State(text=chunk_text, chunk_id=str(chunk_id),error_status=None)
            
            final_state = graph.invoke(initial_state)

            error_status = final_state.get('error_status') 
            if error_status is not None:
                logger.info(f"Errore nello stato per il chunk {chunk_id} dopo l'annotazione: {error_status}")
                continue
            
            chunked_ner_refined_results.append({
                "chunk_id": chunk_id,
                "ner_refined": final_state.get('ner_refined') 
            })
            
        except Exception as e:
            logger.error(f"Errore critico durante l'invocazione della pipeline o l'appiattimento per il chunk {chunk_id}: {e}")
            logger.error(traceback.format_exc())
            # In caso di errore, aggiungiamo comunque un risultato vuoto per non bloccare il merging
            exit(0)
            chunked_ner_refined_results.append({
                "chunk_id": chunk_id,
                "ner_refined": {}
            })
            continue

    logger.info("Esecuzione pipeline completata per tutti i chunk. Avvio fase di merging.")
    # Richiama la funzione di merging per aggregare tutti i risultati dei chunk
    final_merged_result = merge_ner_refined_results(chunked_ner_refined_results)

    logger.info(f"Risultato finale aggregato: {final_merged_result}")
    return final_merged_result