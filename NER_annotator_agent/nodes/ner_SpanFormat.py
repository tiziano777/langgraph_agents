
import re
from states.ner_state import State

class NerSpanFormat:
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
        #print('OUTPUT NerCorrection & INPUT NerSpanFormat: ', state)
        state.span_ner = []

        for segment_text, segment_ner in zip(state.segmented_text, state.segmented_ner):
            span_ner = []
            
            for category, entity_list in segment_ner.items():
                for entity in entity_list:
                    # Trova tutte le occorrenze di `entity` nel testo senza loop infinito
                    if entity and len(entity)>1:
                        for match in re.finditer(re.escape(entity), segment_text):
                            span_ner.append({
                                "type": category,
                                "text": entity,
                                "start": match.start(),
                                "end": match.end()
                            })
                    
            state.span_ner.append(span_ner)
        
        return state