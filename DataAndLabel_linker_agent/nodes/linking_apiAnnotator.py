import json
import re
import spacy
import traceback
from typing import List

from json_repair import repair_json

from states.linking_state import State
from utils.CostLogger import CostLogger 
from utils.ErrorHandler import ErrorHandler


# Caricamento modelli SpaCy
#en_nlp = spacy.load("en_core_web_sm")
#it_nlp = spacy.load("it_core_news_sm")
#sl_nlp = spacy.load("sl_core_news_sm")  

class Annotator:
    """
    Nodo LangGraph compatibile con modelli LLM API-based (es. Gemini via LangChain).
    """

    def __init__(self, llm, input_context, prompts=None):
        """
        :param llm: Modello LangChain-compatible (es. ChatGoogleGenerativeAI).
        :param input_context: Numero massimo di token di contesto.
        :param prompts: Prompt di sistema da usare per l'annotazione.
        """
        self.llm = llm
        self.system_prompts = prompts
        self.input_context = input_context
        self.logger = CostLogger()  # Istanza per logging dei token e costi
        self.error_handler = ErrorHandler()
        self.end_prompt = "\n OUTPUT: \n"

    def annotate(self, state: State):
        text = state.chunk_text
        labels_path= state.labels_path

        text = self.process_text(text)
        
        total_input_tokens = 0
        total_output_tokens = 0

        ####### START CUSTOM LOGIC #######
        
        # Carica le etichette dal file JSON, convert form BIO to spans
        # Parsing IOB -> lista di oggetti con entità complete
        entity_mentions = []
        if labels_path:
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    label_data = json.load(f)

                    current_entity_type = None
                    current_tokens = []

                    for item in label_data:
                        tag = item.get("entity")
                        token = item.get("token")

                        if not tag or not token:
                            continue

                        if tag.startswith("B-"):
                            # Chiude la precedente entità se presente
                            if current_entity_type and current_tokens:
                                entity_mentions.append({
                                    current_entity_type: " ".join(current_tokens)
                                })
                            # Inizia nuova entità
                            current_entity_type = tag[2:]  # rimuove "B-"
                            current_tokens = [token]

                        elif tag.startswith("I-"):
                            entity_type = tag[2:]
                            if current_entity_type == entity_type:
                                current_tokens.append(token)
                            else:
                                # errore di struttura BIO: chiude la corrente, apre nuova
                                if current_entity_type and current_tokens:
                                    entity_mentions.append({
                                        current_entity_type: " ".join(current_tokens)
                                    })
                                current_entity_type = entity_type
                                current_tokens = [token]

                        else:
                            # token senza prefisso: ignora o logga errore
                            continue

                    # Chiude eventuale ultima entità aperta
                    if current_entity_type and current_tokens:
                        entity_mentions.append({
                            current_entity_type: " ".join(current_tokens)
                        })
            except Exception as e:
                print(f"Errore nella lettura del file label: {e}")
                traceback.print_exc()
                return {'error_status': f'Error reading labels file: {str(e)}'}
        else:
            return {'error_status': 'Labels path is missing'}

        #print(f"Entity tokens: {entity_mentions}")
       
        
        # prompt for the linking task
        linking_prompt = self.system_prompts.get('linking_prompt', "")
        try:
            full_prompt = linking_prompt + text +' \n Input LABELS \n ' + str(entity_mentions) + self.end_prompt
            raw_linking = self.error_handler.invoke_with_retry(llm=self.llm, prompt=full_prompt)
            
            log = raw_linking.usage_metadata
            total_input_tokens += log["input_tokens"]
            total_output_tokens += log["output_tokens"]
            
            #EXTRACT LIST JSON OUTPUT
            json_linking = self.extract_json(raw_linking.content)
            
        except Exception as e:
            print(f"Errore nel linking parsing: {e}")
            traceback.print_exc()
        
        '''
        print('text:\n\n'+str(text))
        print("initial_labels:\n\n"+str(entity_mentions))
        print("llm_labels:\n\n"+str(json_linking))
        '''
        
        return {
            'text': text,  # Ritorna il testo processato
            'initial_labels':entity_mentions,  # Ritorna le etichette iniziali estratte dal file
            'labels': json_linking,  # Ritorna le etichette eventualmente corrette dal LLM
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
        } 
    
    def extract_json(self, json_text: str) -> list[dict]: # Il tipo di ritorno è List[Dict]
        """
        Estrae una lista completa di oggetti JSON validi da una stringa di output.
        Usa json_repair per tolleranza agli errori.
        """
        try:
            # Ripara l'intera stringa prima di tentare di caricarla come JSON.
            # Questo è cruciale perché l'output completo dovrebbe essere una lista JSON.
            repaired_text = repair_json(json_text)

            # Il tuo output atteso è una lista di JSON.
            # Assicurati che l'intera stringa riparata sia un JSON array valido.
            # Non cercare solo il primo oggetto {}. Se l'output è una lista,
            # json.loads la gestirà direttamente.
            parsed_json = json.loads(repaired_text)

            # Verifica se il risultato è effettivamente una lista (come atteso)
            if not isinstance(parsed_json, list):
                print(f"L'output JSON riparato non è una lista: {parsed_json}")
                return [] # Restituisci una lista vuota o gestisci l'errore diversamente

            return parsed_json

        except json.JSONDecodeError as e:
            print(f"Errore di decodifica JSON dopo la riparazione:\n→ Testo originale:\n{json_text}\n→ Testo riparato:\n{repaired_text}\n→ Errore: {e}")
            return [] # Restituisci una lista vuota in caso di errore di parsing
        except Exception as e:
            print(f"Errore generico nel parsing JSON:\n→ Testo:\n{json_text}\n→ Errore: {e}")
            return []

    def process_text(self, text: str) -> List[str]:
        """
        rimuove alcune parti del testo che non sono utili
        """

        # 1. \n -> " "
        text = re.sub(r"\n+", " ", text)
        # 2. __+ -> " "
        text = re.sub(r"_{2,}", " ", text)
        # 3. \t -> " "
        text = text.replace("\t", " ")
        # 4. \f -> " "
        text = text.replace("\f", " ")
        # 5. Normalizzazione degli spazi
        text = re.sub(r" {2,}", " ", text)
        
        return text

    def __call__(self, state: State):
        """
        Entry-point per LangGraph node.
        """
        #print('INPUT Annotator: ', state)

        return self.annotate(state)