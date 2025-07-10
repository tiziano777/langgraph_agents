



### TODO ###
### MISTRAL LOADER CANNOT SUPPPORT THIS version, you have to simplify IT!
### ALSO YOU HAVE TO DEFINE ADDITIONAL STATE THAT SUPPORTS REFINEMENT LOOPS




import json
import traceback
import logging

from json_repair import repair_json
from states.ner_state import State

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def handle_exception(state: State, error_string: str, e: Exception) -> dict:
    """
    Funzione ausiliaria per la gestione centralizzata delle eccezioni.
    Logga errore, traceback ed aggiorna lo stato con un messaggio d'errore coerente.
    """
    full_error = f"{error_string}: {str(e)}"
    full_trace = traceback.format_exc()
    logger.error(full_trace)
    state.error_status = str(full_error)
    logger.error("STATE ERROR RETURN: %s", {'state': str(state)})
    return {'error_status': full_error}

class AnnotatorRefiner:
    """
    Nodo LangGraph per raffinare gli span delle entità tramite un LLM Mistral locale.
    Prende in input le annotazioni e chiede al modello di correggerne i bordi.
    """

    # Modificato: llm ora è l'istanza di MistralLoader
    def __init__(self, llm, prompt: str = None):
        """
        Inizializza il nodo SpanRefiner.

        :param llm: Istanza della classe MistralLoader.
        :param prompt: Prompt di sistema da usare per il raffinamento degli span.
        """
        self.llm = llm # Ora è l'istanza di MistralLoader
        self.system_prompt = prompt # Renamed for clarity to match MistralLoader.invoke parameter
        self.end_prompt = "\n Output: \n" # Questo non è più usato per la formattazione diretta

    def refine_spans(self, state: State) -> State:
        """
        Processa le annotazioni segmentate per raffinare gli span utilizzando il modello Mistral locale.
        """
        
        # La query utente per il refiner sarà lo stato attuale delle NER o delle NER raffinate
        user_query_content = ""
        if state.schema_validation_attempts == 0:
            user_query_content = str(state.ner)
        else:
            user_query_content = str(state.ner_refined)
        
        print(user_query_content)
        
        try:
            # Chiamata a invoke() sul MistralLoader, passando system_prompt e user_query
            generated_response_text = self.llm.invoke(self.system_prompt, '\n ner:\n' + str(user_query_content) + self.end_prompt)
            state.ner_refined = self.extract_json(generated_response_text)
        except Exception as e:
            return handle_exception(state, "Exception in llm.invoke (Mistral Refiner)", e)
        
        return state

    def extract_json(self, json_text: str) -> list:
        """
        Ripara e deserializza l'output JSON generato da LLM. Restituisce una lista oppure {} in caso di errore.
        """
        
        try:
            print('json text: \n\n'+json_text)
            repaired_text = repair_json(json_text)
            
            parsed_json = json.loads(repaired_text)

            if not isinstance(parsed_json, list):
                raise ValueError("Parsed JSON is not a list AnnotatorRefiner")

            return parsed_json

        except Exception as e:
            print(traceback.format_exc())
            logger.error(f"Errore durante l'estrazione o riparazione del JSON: {e}")
            return e
      
    def __call__(self, state: State) -> State:
        """
        Entry-point per LangGraph node.
        """
        print(f"INPUT Refiner OUTPUT Annotator (Mistral): {str(state.ner)}")
        return self.refine_spans(state)

