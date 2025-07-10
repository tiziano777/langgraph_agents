import json, re, spacy
import logging
import traceback
import unicodedata

from states.ner_state import State
from json_repair import repair_json

nlp = spacy.load("sl_core_news_lg")

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

class Annotator():
    """
    Wrapper per un modello LLM compilato con llama.cpp, utilizzabile come nodo di elaborazione in LangGraph.
    """
    def __init__(self, llm, max_sentence_length: int = 4000,system_prompt: str = None):
        """
        Inizializza il modello llama.cpp.

        :param llm: Modello quantizzato compitalo com llama.cpp(GGUF).
        :param max_sentence length: Dimensione massima user input.
        :param system_prompt: Prompt di sistema da utilizzare per l'annotazione.
        """
        self.system_prompt = system_prompt
        self.output_prompt= "\nOutput:\n"
        self.end_instruction_token='[/INST]'
        self.max_sentence_length = max_sentence_length
        self.llm = llm

    def annotate(self, state: State):
        """
        Genera un'annotazione per il testo di input utilizzando il modello
        e converte il risultato in un JSON strutturato.

        :param text: Il testo da elaborare.
        :return: Un dizionario Python con l'output JSON.
        """
        
        ### CUSTOM TEXT PROCESSING AND INVOCATION LOGIC ###
        
        text=self.process_text(state.text)
        ner=[]

        ner.append(self.extract_json(self.llm.invoke(self.system_prompt+'\n'+text+ self.output_prompt+ self.end_instruction_token)))

        ### END CUSTOM LOGIC ###
            
        return {'text': text,'ner': ner}

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
            return handle_exception(json_text, "Errore nel parsing JSON", e)
     
    def chunk_text(self, text: str, min_sentence_length: int = 1000):
        """
        Se il testo è corto (<n_max) , lo restituisce così com'è.
        Se è lungo, lo divide in frasi usando spaCy per la lingua slovena,
        escludendo frasi troppo corte.

        :param text: Stringa di input in sloveno.
        :param min_sentence_length: Lunghezza minima delle frasi da includere.
        :return: Lista di stringhe (frasi) se il testo è lungo, altrimenti il testo originale.
        """
        if len(text) < self.max_sentence_length:
            return [text]  # Testo corto, restituisci direttamente
        
        # Tokenizza e suddivide in frasi
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= min_sentence_length]
        
        return sentences
     
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
        print('INPUT Annotator: ', state.text)
        return self.annotate(state)
