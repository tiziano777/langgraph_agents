import json
import logging
import traceback
from utils.ErrorHandler import ErrorHandler
from states.ner_state import State
from json_repair import repair_json


# Configurazione logger globale per il modulo
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def handle_exception(state:State, error_string, e):
    """
    Funzione ausiliaria per la gestione centralizzata delle eccezioni.
    Logga errore, traceback ed aggiorna lo stato con un messaggio d'errore coerente.
    """
    full_error = f"{error_string}: {str(e)}"
    full_trace = traceback.format_exc()
    logger.error(full_error + "\n" + full_trace)
    state.error_status = str(full_error)
    logger.error("STATE ERROR RETURN: %s", {'state': str(state)})
    return {'error_status': full_error}

class Annotator():

    def __init__(self, llm,input_context ,prompt=None):
        """
        Definisci il modello e impostazioni specifiche.

        :param llm: Modello quantizzato compitalo com llama.cpp(GGUF).
        :param input_context: Dimensione massima user input.
        :param system_prompt: Prompt di sistema da utilizzare per l'annotazione.
        """
        self.llm = llm
        self.system_prompt = prompt
        self.input_context = input_context
        self.error_handler = ErrorHandler()
        self.end_prompt= "\nOutput:\n"

    def annotate(self, state: State):
        """
        Genera un'annotazione per il testo di input utilizzando il modello
        e converte il risultato in un JSON strutturato.

        :param text: Il testo da elaborare.
        :return: Un dizionario Python con l'output JSON.
        """
        
        try:
            call=ErrorHandler.invoke_with_retry(llm=self.llm, prompt=str(self.system_prompt+'\n'+state.text+ self.end_prompt))
            state.ner=self.extract_json(call.content)
            log = call.usage_metadata
            
            state.input_tokens +=log["input_tokens"]
            state.output_tokens += log["output_tokens"]

        except Exception as e:
            return handle_exception(state, "Exception in llm.invoke", e)

        return state

    def extract_json(self, json_text: str) -> list:
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
     
    def __call__(self, text: State):
        """
        Metodo principale per elaborare il testo di input.

        :param text: Il testo da elaborare.
        :return: Un dizionario Python con l'output JSON.
        """
        print('\n INPUT Annotator: \n\n', text)
        return self.annotate(text)
