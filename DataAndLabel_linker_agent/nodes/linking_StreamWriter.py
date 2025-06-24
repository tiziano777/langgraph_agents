import json
from states.linking_state import State

class StreamWriter:
    '''
    StreamWriter is a class that writes the output of the NER model to a file in a specific format.
    The output is a JSONL object with two keys: 'text' and 'span'.
    The 'text' key contains the text of the sentence, and the 'span' key contains a list of dictionaries,
    each with at least the keys: 'start', 'end', and 'text', representing the span of an entity.
    '''

    def __init__(self, output_file):
        self.file = output_file

    def _deduplicate_spans(self, spans):
        """
        Remove duplicates in a list of NER spans based on (start, end, text).
        """
        seen = set()
        unique_spans = []
        for span in spans:
            key = (span.get('start'), span.get('end'), span.get('text'))
            if key not in seen:
                seen.add(key)
                unique_spans.append(span)
        return unique_spans

    def __call__(self, state: State) -> State:
        print('OUTPUT NerSpanFormat & INPUT Writer: ', state)
        spans= state.span_ner
        text=state.chunk_text
        id= state.id
        chunk=state.chunk_id
        
        try:
            data_to_write = []  # Accumulate cleaned data
            if spans: # Controlla se 'ner' non è un dizionario vuoto
                deduplicated_spans = self._deduplicate_spans(spans)
                data_to_write.append({'id':id, 'chunk_id': chunk, 'text': text, 'span': deduplicated_spans})

                # Atomic write of the JSONL data
                with open(self.file, "a", encoding="utf-8") as f:
                    for item in data_to_write:
                        json.dump(item, f, ensure_ascii=False)
                        f.write("\n")

                return {'error_status': None}

        except:
            print(f'cannot write on file this data: {text}')
            state.error_status=f'cannot write on file this data: {text}'
            return state


