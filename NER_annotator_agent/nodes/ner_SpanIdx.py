
import re
from states.ner_state import State

class NerSpanFormat:
    '''
    Class to convert NER into span format.
    The input is a text with corresponding NER tags,
    and the output is a list of spans with their start and end positions.
    Each span is represented as a dictionary with the following keys:
    - 'type': the NER category
    - 'text': the entity text
    - 'start': the start position of the entity in the segment
    - 'end': the end position of the entity in the segment
    The class takes a State object as input and modifies its span_ner attribute.
    The input state should have the following attributes:
    - text: a reprocessed text
    - ner: a list of dictionaries, where each dictionary contains NER categories as keys
    and  entity text as values.
    '''
    def __init__(self):
        pass
    
    def __call__(self, state: State) -> State:
        print('OUTPUT NerCorrection & INPUT NerSpanFormat: ', state)
        state.span_ner= []
        for category, entity in state.corrected_ner.items():
            for match in re.finditer(re.escape(entity), state.text):
                state.span_ner.append({
                    "type": category,
                    "text": entity,
                    "start": match.start(),
                    "end": match.end()
                })
        
        return state