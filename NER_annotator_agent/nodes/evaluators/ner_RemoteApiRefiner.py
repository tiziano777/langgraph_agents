import json
import traceback


from json_repair import repair_json
from states.ner_state import State
from utils.CostLogger import CostLogger # Anche se non usiamo i token, la classe potrebbe essere usata altrove
from utils.ErrorHandler import ErrorHandler

class AnnotatorRefiner:
    """
    Nodo LangGraph per raffinare gli span delle entità tramite un LLM.
    Prende in input le annotazioni e chiede al modello di correggerne i bordi.
    """

    def __init__(self, llm, prompt=None):
        """
        Inizializza il nodo SpanRefiner.

        :param llm: Modello LangChain-compatible (es. ChatGoogleGenerativeAI).
        :param prompts: Prompt di sistema da usare per il raffinamento degli span.
        """
        self.llm = llm
        self.prompt = prompt
        self.logger = CostLogger() # Manteniamo per coerenza con il codice originale, anche se non logghiamo i token qui
        self.error_handler = ErrorHandler()
        self.end_prompt = "\n Output: \n"

    def refine_spans(self, state: State) -> State:
        """
        Processa le annotazioni segmentate per raffinare gli span.
        """
        
        if state.schema_validation_attempts==0:
            prompt= self.prompt + '\n' + str(state.ner) + '\n' + self.end_prompt 
        else:
            prompt= self.prompt + '\n' + str(state.ner_refined) + '\n' + self.end_prompt 
        
        try:
            # Utilizziamo rest_invoke_with_retry per la chiamata all'LLM aziendale
            response_content = self.error_handler.rest_invoke_with_retry(llm=self.llm, prompt=prompt)
            state.ner_refined = self.extract_json(response_content)
            
        except Exception as e:
            print(f"Errore durante il raffinamento degli span: {e}")
            traceback.print_exc()
        
        # log = response.usage_metadata
        # state.input_tokens += log["input_tokens"]
        # state.output_tokens += log["output_tokens"]

        return state

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
            return e
      

    def __call__(self, state: State) -> State:
        """
        Entry-point per LangGraph node.
        """
        print("INPUT Refiner OUTPUR Annotator: " + str(state.ner))

        return self.refine_spans(state)
