import json
import re
import spacy
import traceback

from states.disinfo_state import State
from transformers import AutoTokenizer
from typing import List
from json_repair import repair_json

# Caricamento dei modelli linguistici per l'analisi del testo
en_nlp = spacy.load("en_core_web_sm")
it_nlp = spacy.load("it_core_news_sm")

class Annotator:
    """
    Wrapper per un modello LLM compilato con llama.cpp, utilizzabile come nodo di elaborazione in LangGraph.
    """

    def __init__(self, llm, input_context, prompts=None, context_padding_length=512):
        """
        Inizializza il modello API.

        :param llm: Modello quantizzato compilato con llama.cpp (GGUF).
        :param input_context: Numero massimo di token per il contesto di input.
        :param prompts: Prompt di sistema da utilizzare per l'annotazione.
        :param budget_for_user_input: percentuale massima contesto delle frasi in input.
        """
        self.llm = llm
        self.system_prompts = prompts
        self.input_context = input_context
        self.end_prompt = "\n Output JSON Syntax: \n"
        self.context_padding_length = context_padding_length

        # Tokenizer di Gemma 3
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it", trust_remote_code=True)

    def annotate(self, state: State):
        text = state.text
        language = state.language
        title = state.title or ""

        sentences = self.process_text(language, text)
        signals = []  # Qui accumuleremo solo i segnali validi
        texts = []    # E qui i testi corrispondenti

        total_input_tokens = 0
        total_output_tokens = 0
        clickbait_score = 0

        ####### START CUSTOM LOGIC #######
        
        # Clickbait detection
        clickbait_prompt = self.system_prompts.get('clickbait_prompt', "")
        try:
            full_cb_prompt = clickbait_prompt + title
            raw_clickbait = self.error_handler.invoke_with_retry(llm=self.llm, prompt=full_cb_prompt)

            
            log = raw_clickbait.usage_metadata
            total_input_tokens += log["input_tokens"]
            total_output_tokens += log["output_tokens"]

            json_clickbait = self.extract_json(raw_clickbait.content)
            clickbait_score = int(json_clickbait.get('clickbait', 0))
            
        except Exception as e:
            print(f"Errore nel clickbait parsing: {e}")
            traceback.print_exc()
        
        # Segment-level annotation
        annotation_prompts = self.system_prompts.get('election_optimized_strategy')
        gate_prompt = self.system_prompts.get('gate_prompt', "")

        for s in sentences:
            cumulative_annotations = []
            
            if s.strip():
                # ⛔️ Step 1: GATE STEP To avoid unusefull calls (Skip if score < 3)
                try:
                    gate_response = self.error_handler.invoke_with_retry(llm=self.llm, prompt= str(gate_prompt + s + self.end_prompt))
                    log = gate_response.usage_metadata
                    total_input_tokens += log["input_tokens"]
                    total_output_tokens += log["output_tokens"]

                    json_gate = self.extract_json(gate_response.content)
                    gate_score = float(json_gate.get('score', 0))
                except Exception as e:
                    print(f"Errore nel gate scoring: {e}")
                    continue  # Skip segment if error in gate or malformed output

                if gate_score < 3:
                    print(f"gate score too low {gate_score}, skipping segment \n")
                    print(f" \n Segmento skipped: {s[:100]}... \n\n")
                    continue  # Skip segment if gate score is below threshold

                # ✅ Step 2: Run segment-level annotations (only if gate passed)
                for prompt_name, prompt_template in annotation_prompts.items():
                    try:
                        full_prompt = prompt_template + s + self.end_prompt
                        response =  self.error_handler.invoke_with_retry(llm=self.llm, prompt=full_prompt)
                        
                        log = response.usage_metadata
                        total_input_tokens += log["input_tokens"]
                        total_output_tokens += log["output_tokens"]

                        parsed = self.extract_json(response.content)
                        cumulative_annotations.append(parsed)

                    except Exception as e:
                        print(f"Errore nella annotazione con prompt {prompt_name}: {e}")
                        traceback.print_exc()
                        # Anche in caso di errore, aggiungiamo una lista vuota per la categoria per non bloccare
                        cumulative_annotations.append({prompt_name: []}) 

                # Appiattisci le annotazioni accumulate per il segmento corrente
                cumulative_annotations_flat = self.flatten_dict_list(cumulative_annotations)

                # Verifica se cumulative_annotations_flat contiene solo array vuoti Considera un segmento "vuoto" se tutte le liste di entità sono vuote
                # Esempio: {"PERSON": [], "ORG": []}
                is_empty_annotation = True
                for key, value in cumulative_annotations_flat.items():
                    if isinstance(value, list) and len(value) > 0:
                        #print(f"Trovata annotazione non vuota len(value) > 0 per la chiave {key}: {value}")
                        #print(value) # Span+intensity detected!
                        is_empty_annotation = False
                        break # Trovato almeno un'entità, quindi non è vuoto

                if is_empty_annotation:
                    print(f"Annotazioni vuote per il segmento dopo il gate, saltando il segmento: {s[:100]}...")
                    continue # Salta l'aggiunta di questo segmento e delle sue annotazioni

                # Se non è vuoto, allora lo aggiungiamo
                texts.append(s)
                signals.append(cumulative_annotations_flat)

        self.logger(total_input_tokens,total_output_tokens)
        
        return {
            'clickbait': clickbait_score,
            'segmented_text': texts,
            'segmented_signals': signals,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
        } 
        
    def extract_json(self, text: str) -> dict:
        """
        Estrae e corregge l'output JSON generato dal modello utilizzando la libreria `json_repair`:
        """
        try:
            repaired_json = repair_json(json_text)
            json_match = re.search(r"\{.*?\}", text, re.DOTALL)
            if not json_match:
                repaired_json = repair_json(json_text)
                raise ValueError("Nessun JSON valido trovato nel testo: ",text)

            json_text = json_match.group(0)

            # Correggi il JSON utilizzando la libreria json_repair
            repaired_json = repair_json(json_text)

            if repaired_json is None:
                raise ValueError("Impossibile riparare il JSON.")

            parsed_json = json.loads(repaired_json)
            return parsed_json

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Errore nel parsing JSON:\n\n[ text: \n\n {text} \n\n error: {e} ]")
            return {}

    def process_text(self, lang: str, text: str) -> List[str]:
        """
        Segmenta il testo in chunk semantici concatenando frasi fino a rientrare nel token budget.

        :param lang: Lingua del testo ('English' o 'Italian').
        :param text: Testo da elaborare.
        :return: Lista di chunk semantici.
        """
        # Carica il sentencer per la lingua specificata
        nlp = it_nlp if lang == 'Italian' else en_nlp
        doc = nlp(text)

        # Segmenta il testo in frasi
        sentences = [sent.text.strip() for sent in doc.sents]

        # Calcola il massimo numero di token dei prompt di sistema
        prompt_token_lengths = [
            len(self.tokenizer.tokenize(p)) for p in self.system_prompts.get('annotation_prompts', {}).values()
        ]
        
        print("Prompt token lengths: ", prompt_token_lengths)
        max_prompt_tokens = max(prompt_token_lengths) if prompt_token_lengths else 0
        token_budget = int(self.input_context - max_prompt_tokens -self.context_padding_length)

        print("Max Token prompt budget: ", max_prompt_tokens)
        print("Token budget: ", token_budget)

        # Concatenazione di frasi fino al raggiungimento del token budget
        chunks = []
        current_chunk = []
        current_token_count = 0

        for sentence in sentences:
            sentence_token_count = len(self.tokenizer.tokenize(sentence))

            # Se l'aggiunta della frase non supera il token budget, aggiungi la frase al chunk corrente
            if current_token_count + sentence_token_count <= token_budget:
                current_chunk.append(sentence)
                current_token_count += sentence_token_count
            else:
                # Se supera il token budget, salva il chunk corrente e inizia un nuovo chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_token_count = sentence_token_count

        # Aggiungi l'ultimo chunk se non è vuoto
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Stampa i chunk risultanti
        
        for i, chunk in enumerate(chunks):
            print(f"CHUNK {i+1} (lunghezza {len(self.tokenizer.tokenize(chunk))})")
        
        
        return chunks

    def __call__(self, text: State):
        """
        Metodo principale per elaborare il testo di input.

        :param text: Il testo da elaborare.
        :return: Un dizionario Python con l'output JSON.
        """
        print('INPUT Annotator: ', text)
        return self.annotate(text)
