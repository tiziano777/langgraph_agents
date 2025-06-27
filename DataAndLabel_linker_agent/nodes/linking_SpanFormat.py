
import re
from states.linking_state import State

class SpanFormat:
    '''
    Class to convert NER segments into span format.
    The input is a list of segments with their corresponding NER tags,
    and the output is a list of spans with their start and end positions.
    Each span is represented as a dictionary with the following keys:
    - 'type': the NER category
    - 'text': the entity text
    - 'start': the start position of the entity in the segment
    - 'end': the end position of the entity in the segment
    The class takes a State object as input and modifies its span_ner attribute.
    The input state should have the following attributes:
    - segmented_text: a list of text segments
    - segmented_ner: a list of dictionaries, where each dictionary contains NER categories as keys
    and lists of entities as values.
    '''
    def __init__(self):
        pass
    
    def __call__(self, state: State) -> State:
        print(f'\nOUTPUT Output Correction & INPUT Linking SpanFormat:  {state} \n')
        
        if state.error_status:
            return state
        
        text=state.chunk_text
        span_ner = []
            
        for label in state.labels:
            category, entity = next(iter(label.items()))
            # Trova tutte le occorrenze di `entity` nel testo senza loop infinito
            if entity and len(entity)>1:
                for match in re.finditer(re.escape(entity), text):
                    span_ner.append({
                            "type": category,
                            "text": entity,
                            "start": match.start(),
                            "end": match.end()
                        })
        
        
        return {"span_ner": span_ner}