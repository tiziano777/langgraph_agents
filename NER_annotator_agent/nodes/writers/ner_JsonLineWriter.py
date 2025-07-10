import json
# Assicurati che 'State' sia importato correttamente dal tuo modulo
from states.ner_state import State 

class StreamWriter:
    '''
    StreamWriter is a class that writes the output of the NER model to a file in a specific format.
    The output is a JSONL object with two keys: 'text' and 'ner'.
    The 'text' key contains the text of the sentence, and the 'ner' key contains a list of dictionaries,
    each with the format { "entity_type": "entity_value"}.
    This version includes deduplication of entity objects in 'ner' and 'ner_refined' lists.
    '''

    def __init__(self, output_file):
        self.file = output_file

    def _deduplicate_entity_list(self, entity_list: list[dict]) -> list[dict]:
        """
        Rimuove gli oggetti duplicati da una lista di dizionari di entità.
        Un duplicato è considerato un dizionario con la stessa coppia chiave-valore.
        Ad esempio, {"TenderOrg": "abc"} e {"TenderOrg": "abc"} sono duplicati.
        Mantiene l'ordine originale il più possibile per le entità uniche.
        """
        seen = set()
        deduplicated_list = []
        for entity_dict in entity_list:
            # Converti il dizionario in un formato hashable (tupla di coppie (chiave, valore))
            # Ordina gli elementi per garantire che l'ordine delle chiavi non influenzi l'hashing
            hashable_item = tuple(sorted(entity_dict.items()))
            if hashable_item not in seen:
                seen.add(hashable_item)
                deduplicated_list.append(entity_dict)
        return deduplicated_list

    def _write_to_file(self, state: State):
        """
        Write the cleaned data to the file in JSONL format.
        Deduplicates 'ner' and 'ner_refined' lists before writing.
        """
        try:
            data_to_write = []  # Accumulate cleaned data
            
            # Deduplica le liste 'ner' e 'ner_refined'
            deduplicated_ner = []
            if state.ner:
                deduplicated_ner = self._deduplicate_entity_list(state.ner)
            
            deduplicated_ner_refined = []
            if state.ner_refined:
                deduplicated_ner_refined = self._deduplicate_entity_list(state.ner_refined)

            if deduplicated_ner or deduplicated_ner_refined: # Scrivi solo se ci sono entità da salvare
                data_to_write.append({
                    'id': state.id,
                    'chunk_id': state.chunk_id,
                    'input_tokens': state.input_tokens,
                    #'refiner_input_tokens': state.refine_input_tokens,
                    'total_output_tokens': state.output_tokens + state.refine_output_tokens,
                    'text': state.text,
                    'ner': deduplicated_ner,        
                    #'ner_refine': deduplicated_ner_refined 
                })
                
            # Atomic write of the JSONL data
            with open(self.file, "a", encoding="utf-8") as f:
                for item in data_to_write:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")

            return {'error_status': None}

        except Exception as e: # Cattura l'eccezione specifica per un debug migliore
            print(f'Cannot write data to file for text: {state.text}. Error: {e}')
            state.error_status = f'Cannot write data to file for text: {state.text}. Error: {e}'
            return state

    def __call__(self, state: State) -> State:
        print('OUTPUT NerSpanFormat & INPUT Writer: ', state.ner_refined)
        
        # Esegui la scrittura e gestisci lo stato di errore
        result = self._write_to_file(state)
        state.error_status = result.get('error_status') # Aggiorna lo stato di errore

        if state.error_status is not None:
            print(f"Error writing to file: {state.error_status}")
            return state
        else:
            # Store data on Database (se questa logica è esterna, assicurati che sia gestita altrove)
            # Se 'Store data on Database' è un placeholder per un'azione futura, va bene.
            # Altrimenti, se è una parte mancante, dovresti implementarla qui.
            return state

