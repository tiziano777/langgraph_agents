import json, re, spacy
from states.ner_state import State
from json_repair import repair_json

nlp = spacy.load("sl_core_news_lg")


class Annotator():
    """
    Wrapper per un modello LLM compilato con llama.cpp, utilizzabile come nodo di elaborazione in LangGraph.
    """
    def __init__(self, llm, max_sentence_length: int = 855,system_prompt: str = None):
        """
        Inizializza il modello llama.cpp.

        :param llm: Modello quantizzato compitalo com llama.cpp(GGUF).
        :param max_sentence length: Dimensione massima user input.
        :param system_prompt: Prompt di sistema da utilizzare per l'annotazione.
        """
        self.system_prompt = system_prompt
        self.output_prompt= "\nOutput JSON Syntax:\n"
        self.max_sentence_length = max_sentence_length
        self.llm = llm

    def annotate(self, state: State):
        """
        Genera un'annotazione per il testo di input utilizzando il modello
        e converte il risultato in un JSON strutturato.

        :param text: Il testo da elaborare.
        :return: Un dizionario Python con l'output JSON.
        """
        text=state.text
        sentences=self.process_text(text)
        ners=[]
        texts=[]
        for s in sentences:
            texts.append(s)
            ners.append(self.extract_json(self.llm.invoke(self.system_prompt+s+ self.output_prompt)))

        return {'segmented_text': texts,'segmented_ner': ners}

    def extract_json(self, json_text: str) -> dict:
        """
        Estrae e corregge l'output JSON generato dal modello utilizzando la libreria `json_repair`:
        """
        try:     
            json_text = repair_json(json_text)
            json_match = re.search(r"\{.*?\}", json_text, re.DOTALL)
            if json_match is None:
                print(f"Nessun JSON trovato nel testo:\n\n{json_text}")
                return {}
            repaired_json = repair_json(json_match.group(0))
            parsed_json = json.loads(repaired_json)    
            
            return parsed_json

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Errore nel parsing JSON:\n\n[ text: \n\n {json_text} \n\n error: {e} ]")
            return {}
     
    def process_text(self, text: str, min_sentence_length: int = 19):
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
     
    def __call__(self, text: State):
        """
        Metodo principale per elaborare il testo di input.

        :param text: Il testo da elaborare.
        :return: Un dizionario Python con l'output JSON.
        """
        #print('INPUT Annotator: ', text)
        return self.annotate(text)
