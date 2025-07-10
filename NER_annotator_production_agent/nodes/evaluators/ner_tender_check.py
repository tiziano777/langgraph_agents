from typing import List, Dict, Any, Set
from states.ner_state import State
from utils.logger_config import setup_logger, handle_exception
logger = setup_logger(__name__)

### CUSTOM SCHEMA VALIDATOR ###

class Formatter:
    """
    Classe per appiattire una lista di dizionari in un singolo dizionario,
    scartando valori nulli o stringhe vuote e gestendo duplicati.
    """
    def __init__(self, enitity_set):

        self.seen_keys: Set[str] = set()
        self.eligible_keys: Set[str] = enitity_set or set("TenderType","TenderNumber","TenderCode","TenderYear","TenderOrg","TenderTel","TenderFax","TenderDeadline","TenderPerson")

    def _is_valid_value(self, value: Any) -> bool:
        """
        Controlla se un valore è valido (non nullo e non stringa vuota).
        """
        return value is not None and value != ""

    def flatten_list_of_dicts(self, data: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Appiattisce una lista di dizionari in un singolo dizionario,
        scartando coppie chiave-valore con valori nulli o stringhe vuote.
        Gestisce i duplicati delle chiavi inserendo solo la prima occorrenza.
        """
        if not data:
            logger.debug("Input list is empty, returning empty dictionary.")
            return {}

        flattened_dict: Dict[str, str] = {}
        self.seen_keys.clear()  # Resetta le chiavi viste per ogni nuova operazione di appiattimento

        for item_dict in data:
            if not isinstance(item_dict, dict):
                logger.warning(f"Skipping non-dictionary item: {item_dict}")
                continue

            for key, value in item_dict.items():
                if self._is_valid_value(value) and key not in self.seen_keys:
                    flattened_dict[key] = value
                    self.seen_keys.add(key)
                    self.eligible_keys.add(key) # Aggiunge la chiave alle chiavi idonee
                elif key in self.seen_keys:
                    logger.debug(f"Key '{key}' already seen, skipping to avoid duplication.")
                else:
                    logger.debug(f"Skipping key '{key}' due to invalid value: '{value}'")
        
        logger.info(f"Flattened dictionary: {flattened_dict}")
        logger.debug(f"Eligible keys after flattening: {self.eligible_keys}")
        return flattened_dict

    def __call__(self, state: State) -> State:
        """
        Metodo principale per elaborare lo stato LangGraph.
        Prende state.ner, lo appiattisce e salva l'output in state.ner_refined.
        """
        logger.info("Formatter node called.")
        logger.debug(f"Input state.ner: {state.ner}")

        try:
            if not hasattr(state, 'ner') or not isinstance(state.ner, list):
                error_msg = "state.ner is missing or not a list, cannot format."
                logger.error(error_msg)
                state.error_status = error_msg
                return state # Ritorna lo stato con l'errore senza eccezione

            refined_ner = self.flatten_list_of_dicts(state.ner)
            state.ner_refined = refined_ner
            logger.info(f"Formatted output saved to state.ner_refined: {state.ner_refined}")
        except Exception as e:
            return handle_exception(state, "Exception in Formatter during flattening", e)

        return state