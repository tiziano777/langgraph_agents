import re
from rapidfuzz import fuzz, process
from states.ner_state import State

class NerCorrection():
    '''
    Correttore di entità NER.
    Dato un testo e delle entità NER, corregge le entità in base al testo.
    Utilizza la libreria `rapidfuzz` per confrontare le entità con le sottostringhe del testo.
    La correzione avviene in due fasi:
    Legge le entità NER e le confronta con il testo.
    Se non trova corrispondenze con il testo, elimina la ner.
    Se trova una corrispondenza con un punteggio di similarità superiore a `similarity_threshold`, la sostituisce con il testo.
    Cosi possiamo assicurare che le entità NER siano sempre presenti nel testo.
    '''
    def __init__(self, similarity_threshold=79):
        self.similarity_threshold = similarity_threshold
    
    def extract_ngram_candidates(self, text, entity_length):
        """
        Estrae sottostringhe di lunghezza `entity_length` dal testo.
        """
        words = text.split()
        ngrams = [' '.join(words[i:i + entity_length]) for i in range(len(words) - entity_length + 1)]
        return ngrams

    '''def clean_entity(self,entity):
        """
        Rimuove la punteggiatura solo all'inizio e alla fine dell'entità.
        """
        # Rimuove la punteggiatura solo all'inizio della stringa
        entity = str(entity)
        entity = re.sub(r'^[^\w\d]+', '', entity)
        
        # Rimuove la punteggiatura solo alla fine della stringa
        entity = re.sub(r'[^\w\d]+$', '', entity)
    
        return entity
'''
    
    def clean_ner(self, text,ner):
        """
        Corregge le entità in `ner` in base al testo `text`, eliminando duplicati e ripulendo i caratteri speciali.
        """
        text_words = set(str(text).split())  # Parole nel testo
        corrected_ner = {}
        
        for dict_elem in ner: 
            best_match=''
            for category, entity in dict_elem.items():  # Itera su ogni categoria e le sue entità
            
                unique_entities = set()
                entity_words = set(str(entity).split())  # Parole dell'entità
                
                # Controllo se tutte le parole dell'entità sono presenti nel testo
                if entity_words.issubset(text_words):
                    unique_entities.add(entity)
                    
                else:
                    entity_length = len(str(entity).split())  # Conta le parole dell'entità
                    candidates = self.extract_ngram_candidates(text, entity_length)
                    
                    #caso in cui una entita sia piu grande di un testo chunked
                    if len(str(entity))>len(str(text)) or candidates==[]:
                        continue
                    
                    # Trova la migliore corrispondenza nel testo
                    best_match, score, _ = process.extractOne(str(entity), candidates, scorer=fuzz.ratio)
                    if best_match and score >= self.similarity_threshold:
                        # Normalizza la corrispondenza finale
                        best_match = best_match # Pulisce la corrispondenza finale
                        unique_entities.add(best_match)  # Sostituisci con il match migliore
            
            if unique_entities:
                corrected_ner[category] = list(unique_entities)[0]
                
        return corrected_ner
    
    def __call__(self, state:State):
        print('\n OUTPUT ANNOTATOR & INPUT NerCorrector: \n', state)
        return {'corrected_ner': self.clean_ner(state.text,state.ner)}
