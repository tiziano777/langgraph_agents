import json
import re
import spacy
import traceback
import logging
import unicodedata
from typing import List
from json_repair import repair_json

from states.linking_state import State
from utils.ErrorHandler import ErrorHandler

# Configurazione logger globale per il modulo
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def handle_exception(state, error_string, e):
    """
    Funzione ausiliaria per la gestione centralizzata delle eccezioni.
    Logga errore, traceback ed aggiorna lo stato con un messaggio d'errore coerente.
    """
    full_error = f"{error_string}: {str(e)}"
    full_trace = traceback.format_exc()
    logger.error(full_error + "\n" + full_trace)
    state.error_status = full_error
    logger.error("STATE ERROR RETURN: %s", {'state': str(state)})
    return {'error_status': full_error}

class Annotator:
    """
    Nodo LangGraph compatibile con modelli LLM API-based (es. Gemini via LangChain).
    """

    def __init__(self, llm, input_context, prompts=None):
        self.llm = llm
        self.system_prompts = prompts
        self.input_context = input_context
        self.error_handler = ErrorHandler()
        self.end_prompt = "\n 📥📥 Output JSON text: : \n"

    def __call__(self, state: State):
        print(f'\n INPUT Annotator:  {state} \n')
        return self.annotate(state)

    def annotate(self, state: State):
        ### CUSTOM LOGIC ###
        # Core della logica di annotazione: normalizzazione testo, estrazione etichette IOB, invocazione LLM
        try:
            text = self.process_text(state.chunk_text)
            state.chunk_text = text  # Aggiorna lo stato con il testo processato
        except Exception as e:
            return handle_exception(state, "Exception in process_text", e)

        try:
            entity_mentions = self.extract_iob_labels(state)
        except Exception as e:
            return handle_exception(state, "Exception in extract_iob_labels", e)

        print(f'Entity mentions extracted: {entity_mentions}')
        
        ### CUSTOM LOGIC ###
        
        try:
            linking_prompt = self.system_prompts.get('linking_prompt', "")
            # Costruzione del prompt per LLM
            full_prompt = linking_prompt + '\n "chunk_text": "' + text + ' " \n "entities":'+   str(entity_mentions) + self.end_prompt

            print(f'Full prompt for LLM: {full_prompt}')
            
            # Invocazione robusta del modello LLM
            raw_linking = self.error_handler.invoke_with_retry(llm=self.llm, prompt=full_prompt)
            log = raw_linking.usage_metadata
            total_input_tokens = log.get("input_tokens", 0)
            total_output_tokens = log.get("output_tokens", 0)

        except Exception as e:
            return handle_exception(state, "Exception in LLM call or prompt construction", e)

        try:
            # Parsing JSON dell'output del modello
            print(f'Raw linking content: {raw_linking}')
            json_linking = self.extract_json(raw_linking.content, state)
            if json_linking == {}:
                raise ValueError("Parsed JSON is empty")
        except Exception as e:
            return handle_exception(state, "Errore nel parsing JSON dell'output LLM", e)

        return {
            'chunk_text': json_linking['chunk_text'],
            'initial_labels': entity_mentions,
            'labels': json_linking['labels'],
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
        }

    def extract_iob_labels(self, state: State) -> List[dict]:
        """
        Estrae entità da file etichettato in formato IOB. Ritorna una lista di dizionari.
        """
        labels_path = state.labels_path
        entity_mentions = []

        if not labels_path:
            raise ValueError("Labels path is missing")

        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                label_data = json.load(f)

                current_entity_type = None
                current_tokens = []

                for item in label_data:
                    tag = item.get("entity")
                    token = item.get("token")

                    if not tag or not token:
                        continue

                    if tag.startswith("B-"):
                        if current_entity_type and current_tokens:
                            entity_mentions.append({current_entity_type: " ".join(current_tokens)})
                        current_entity_type = tag[2:]
                        current_tokens = [token]

                    elif tag.startswith("I-"):
                        entity_type = tag[2:]
                        if current_entity_type == entity_type:
                            current_tokens.append(token)
                        else:
                            if current_entity_type and current_tokens:
                                entity_mentions.append({current_entity_type: " ".join(current_tokens)})
                            current_entity_type = entity_type
                            current_tokens = [token]

                if current_entity_type and current_tokens:
                    entity_mentions.append({current_entity_type: " ".join(current_tokens)})

        except Exception as e:
            raise e

        return entity_mentions

    def extract_json(self, json_text: str, state: State) -> list:
        """
        Ripara e deserializza l'output JSON generato da LLM. Restituisce una lista oppure {} in caso di errore.
        """
        try:
            print("json text: ", json_text)
            repaired_text = repair_json(json_text)
            
            parsed_json = json.loads(repaired_text)

            if not isinstance(parsed_json, dict):
                raise ValueError("Parsed JSON is not a dict")

            return parsed_json

        except Exception as e:
            return handle_exception(state, "Errore nel parsing JSON", e)

    def process_text(self, text: str) -> str:
        ### CUSTOM TEXT PROCESSING WITH NORMALIZATION ###
        try:
            # Conversione minuscola
            text = text.lower()

            # Normalizzazione dei caratteri conacritici (e.g., "Jöhn" -> "john")
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

            # Sostituisce la punteggiatura (eccetto "/" , "-" e '.') con spazio
            text = re.sub(r"[^\w\s./-]", " ", text)

            ### CUSTOM LOGIC ###
            # Pulizia input OCR da sequenze anomale
            text = re.sub(r"\n+", " ", text)
            text = re.sub(r"_{2,}", " ", text)
            text = text.replace("\t", " ")
            text = text.replace("\f", " ")
            text = re.sub(r" {2,}", " ", text)

            return text.strip()

        except Exception as e:
            raise e
