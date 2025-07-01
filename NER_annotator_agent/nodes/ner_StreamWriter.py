import json
from states.ner_state import State

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

    def _write_to_file(self, state: State):
        """
        Write the cleaned data to the file in JSONL format.
        """
        try:
            data_to_write = []  # Accumulate cleaned data
            if state.span_ner: # Controlla se 'ner' non è un dizionario vuoto
                    deduplicated_spans = self._deduplicate_spans(state.span_ner)
                    data_to_write.append({'id':state.id, 'chunk_id':state.chunk_id,'text': state.text , 'span': deduplicated_spans})

            # Atomic write of the JSONL data
            with open(self.file, "a", encoding="utf-8") as f:
                for item in data_to_write:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")

            return {'error_status': None}

        except:
            print(f'cannot write on file this data: {state.text}')
            state.error_status=f'cannot write on file this data: {state.text}'
            return state

    def _write_to_db(self, state: State):
        """
        Placeholder for writing to a database.
        Currently, this method does nothing but can be implemented later.
        """
        # Implement database writing logic here if needed
        pass

    def __call__(self, state: State) -> State:
        print('OUTPUT NerSpanFormat & INPUT Writer: ', state)
        state.error_status = self._write_to_file(state)
        if state.error_status is not None:
            print(f"Error writing to file: {state.error_status}")
            return state
        else:
            # Store data on Database
            return state


