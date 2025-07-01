import re
import logging
import traceback
import unicodedata
from states.ner_state import State

#nlp = spacy.load("sl_core_news_lg")

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

class Preprocessor():
    """
    Initial System Gateway for text preprocessing.
    """
    def __init__(self):
        pass
    
    '''def chunk_text(self, text: str, min_sentence_length: int = 1000):
        """
        Se il testo è corto (<n_max) , lo restituisce così com'è.
        Se è lungo, lo divide in frasi usando spaCy,
        escludendo frasi troppo corte.

        :param text: Stringa di input.
        :param min_sentence_length: Lunghezza minima delle frasi da includere.
        :return: Lista di stringhe (frasi) se il testo è lungo, altrimenti il testo originale.
        """
        if len(text) < self.max_sentence_length:
            return [text]  # Testo corto, restituisci direttamente
        
        # Tokenizza e suddivide in frasi
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= min_sentence_length]
        
        return sentences
    '''
    
    def process_text(self, text: str) -> str:
        ### CUSTOM TEXT PROCESSING WITH NORMALIZATION ###
        try:
            # Conversione minuscola
            text = text.lower()

            # Normalizzazione dei caratteri con diacritici (e.g., "Jöhn" -> "john")
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

            # Sostituisce trattini, underscore, virgole e due punti con spazi
            text = re.sub(r"[-_,:]+", " ", text)

            ### CUSTOM LOGIC ###
            # Pulizia input OCR da sequenze anomale
            text = re.sub(r"\n+", " ", text)
            #text = re.sub(r"_{2,}", " ", text)  # Ridondante ora, ma mantieni per sicurezza
            text = text.replace("\t", " ")
            text = text.replace("\f", " ")
            text = re.sub(r" {2,}", " ", text)
            
            return text.strip()
        
        except Exception as e:
            raise e
     
    def __call__(self, state: State):
        """
        Metodo principale per elaborare il testo di input.

        :param text: Il testo da elaborare.
        :return: Un dizionario Python con l'output JSON.
        """
        print('\nINPUT Preprocessor:\n', state.text)
        
        try:
            state.text=self.process_text(state.text) 
        except Exception as e:
            return handle_exception(state, "Exception in process_text", e) 
        
        return state
