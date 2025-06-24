import re
from rapidfuzz import fuzz, process
from states.linking_state import State

class OutputCorrection():
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
    
    '''
    def extract_ngram_candidates(self, text, entity_length):
        """
        Estrae sottostringhe di lunghezza `entity_length` dal testo.
        """
        words = text.split()
        ngrams = [' '.join(words[i:i + entity_length]) for i in range(len(words) - entity_length + 1)]
        return ngrams
    '''

    def clean_entity(self,entity):
        """
        Rimuove la punteggiatura solo all'inizio e alla fine dell'entità.
        """
        # Rimuove la punteggiatura solo all'inizio della stringa
        entity = str(entity)
        entity = re.sub(r'^[^\w\d]+', '', entity)
        
        # Rimuove la punteggiatura solo alla fine della stringa
        entity = re.sub(r'[^\w\d]+$', '', entity)
    
        return entity

    def clean_ner(self, text, ner):
        """
        Corregge le entità in `ner` in base al testo `text`, eliminando duplicati e ripulendo i caratteri speciali.
        """
        text_words = set(str(text).split())  # Parole nel testo
        final_labels = []
        
        for label in ner: # for each category key
            category, entity = next(iter(label.items()))
            unique_entities = set()
            entity_words = set(str(entity).split())  # Parole dello span dell' entità
                
            # Controllo se tutte le parole dell'entità sono presenti nel testo
            if entity_words.issubset(text_words):
                unique_entities.add(self.clean_entity(entity))
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
                    #best_match = self.clean_entity(best_match)  # Pulisce la corrispondenza finale
                    
                    final_labels.append({str(category):str(best_match)})

        print(final_labels)
        exit(0)
        return final_labels
    
    def __call__(self, state:State):
        print('\nOUTPUT ANNOTATOR & INPUT NerCorrector: \n' + str(state.chunk_text)+" \n "+str(state.labels))
        
        text= state.chunk_text.lower()
        ner= state.labels.lower()

        corrected_ner_item=self.clean_ner(text,ner)

        return {'labels': corrected_ner_item}
