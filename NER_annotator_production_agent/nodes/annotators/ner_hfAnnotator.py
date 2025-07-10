import json
import logging
import traceback
from json_repair import repair_json
from states.ner_state import State

from utils.logger_config import setup_logger, handle_exception
logger = setup_logger(__name__)

class Annotator:
    """
    Nodo LangGraph per generare annotazioni NER utilizzando un LLM Mistral locale
    tramite la classe MistralLoader semplificata.
    """

    # Modificato: llm ora è l'istanza di MistralLoader
    def __init__(self, llm):
        """
        Definisci il modello e impostazioni specifiche.

        :param llm: Istanza della classe MistralLoader.
        :param input_context: Dimensione massima user input (utilizzata per la logica del prompt).
        :param prompt: Prompt di sistema da utilizzare per l'annotazione (il TENDER_PROMPT).
        """
        self.llm = llm # MistralLoader

    def annotate(self, state: State) -> State:
        """
        Genera un'annotazione per il testo di input utilizzando il modello Mistral locale
        e converte il risultato in un JSON strutturato.

        :param state: Lo stato corrente della pipeline contenente il testo da elaborare.
        :return: Lo stato aggiornato con le annotazioni NER.
        """
        try:
            generated_response_text = self.llm.invoke(state.chunk_id,state.text)
            state.ner = self.extract_json(generated_response_text)

        except Exception as e:
            return handle_exception(state, "Exception in llm.invoke (Mistral Annotator)", e)

        return state

    def extract_json(self, json_text: str) -> list:
        """
        Ripara e deserializza l'output JSON generato da LLM. Restituisce una lista oppure {} in caso di errore.
        """
        try:
            
            #print("\n\njson text:\n\n"+json_text)
            
            repaired_text = repair_json(json_text)
            
            parsed_json = json.loads(repaired_text)

            if not isinstance(parsed_json, list):
                raise ValueError("Parsed JSON is not a list")

            return parsed_json

        except Exception as e:
            logger.error(f"Errore durante l'estrazione o riparazione del JSON: {e}")
            print( traceback.format_exc())
            return e
    
    def __call__(self, state: State) -> State:
        """
        Metodo principale per elaborare il testo di input come nodo LangGraph.
        
        :param state: Lo stato corrente della pipeline.
        :return: Lo stato aggiornato.
        """
        print(f'\n INPUT Annotator (Mistral): \n\n {state.text}')
        return self.annotate(state)
