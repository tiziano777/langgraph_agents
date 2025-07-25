# scripts/training_callbacks.py

import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import json
from json_repair import repair_json # Importa repair_json

class GenerationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_dataset_for_inference, 
                 num_examples=3, log_steps_interval=10, max_new_tokens=350):
        
        self.model = model
        self.tokenizer = tokenizer
        
        # eval_dataset_for_inference deve essere un sottoinsieme del processed_eval_dataset
        # che è stato mappato con format_ner_example_for_training,
        # quindi la colonna rilevante sarà "text" (input utente + istruzioni + output atteso).
        # Per la generazione, useremo solo la parte di input.
        
        # Se vuoi usare gli esempi dal test_dataset (quello con input_for_inference e expected_output)
        # dovresti passare processed_test_dataset qui.
        # Per il logging intermedio durante il training, useremo gli esempi del set di validazione.
        # Questi sono formattati come input + ground truth.
        
        # Prendiamo un sottoinsieme degli esempi di validazione per non sovraccaricare il log
        # Estraiamo solo il numero desiderato di esempi.
        selected_examples = eval_dataset_for_inference.select(range(min(num_examples, len(eval_dataset_for_inference))))
        
        # Ora dobbiamo estrarre il prompt che il modello vedrà DURANTE L'INFERENZA
        # E la ground truth con cui confrontarlo.
        # Questo riflette la logica di `format_ner_example_for_inference` ma applicata agli esempi di eval.
        
        self.prompts_for_generation = []
        self.ground_truths = []
        
        # Qui dobbiamo replicare la logica di estrazione del prompt per inferenza
        # dalla stringa completa 'text' del dataset di training/eval.
        # La colonna 'text' in eval_dataset_for_inference (processed_eval_dataset)
        # è del formato: <s>[INST] USER_PROMPT [/INST] ASSISTANT_RESPONSE</s>
        
        end_instruction_token_str = "[/INST]" # Il delimitatore chiave
        
        for example in selected_examples:
            full_sequence = example["text"] # Questa è la stringa completa del training/eval
            
            inst_end_pos = full_sequence.rfind(end_instruction_token_str)
            
            if inst_end_pos != -1:
                # L'input per l'inferenza è tutto fino a [/INST] incluso, più lo spazio per la generazione
                inference_input = full_sequence[:inst_end_pos + len(end_instruction_token_str)].strip()
                # Aggiungi lo spazio che tokenizer.apply_chat_template con add_generation_prompt=True aggiungerebbe
                # Mistral v0.3 si aspetta uno spazio dopo [/INST]
                inference_input += self.tokenizer.bos_token # Aggiungi BOS per il template Mistral. E' già nell'input prompt!
                                                            # NO, il tokenizer.apply_chat_template lo aggiunge se tokenize=True.
                                                            # Se tokenize=False, devi assicurarti che sia già nella stringa.
                                                            # I tuoi prompts di training/eval iniziano con <s>[INST], che è corretto.
                                                            # Per l'inferenza si vuole <s>[INST]...[/INST]
                                                            # Quindi il `full_sequence` è `<s>[INST] ... [/INST] ... </s>`
                                                            # L'input per l'inferenza è `<s>[INST] ... [/INST]`
                # tokenizer.apply_chat_template con add_generation_prompt=True aggiunge uno spazio dopo [/INST]
                # Se il tuo prompt non ha lo spazio, aggiungilo qui.
                # Per coerenza con format_ner_example_for_inference, ri-applichiamo il template
                # per creare il prompt di inferenza da un esempio (anche se di training/eval)
                
                # Questa parte è un po' delicata. Se processed_eval_dataset ha "text": "<s>[INST] USER [/INST] ASSISTANT </s>"
                # e vogliamo generare da "<s>[INST] USER [/INST]", dobbiamo ricrearlo.
                # L'ideale è che la callback riceva lo stesso formato del test set (input_for_inference, expected_output).
                # MA, dato che la stai integrando nel trainer, e il trainer lavora con eval_dataset,
                # e eval_dataset ha la colonna "text" completa...
                # Dobbiamo estrarre la parte `<s>[INST] ... [/INST]` e la parte `ASSISTANT_RESPONSE`.
                
                # Trova la fine dell'istruzione e l'inizio della risposta dell'assistente
                assistant_response_start_pos = inst_end_pos + len(end_instruction_token_str)
                
                # Il prompt effettivo per la generazione (senza risposta dell'assistente)
                # È la parte "<s>[INST] ... [/INST]"
                prompt_text_for_gen = full_sequence[:assistant_response_start_pos].strip()

                # La ground truth è la parte dopo "[/INST]" e prima di "</s>"
                gt_text = full_sequence[assistant_response_start_pos:].strip()
                if gt_text.endswith(self.tokenizer.eos_token):
                    gt_text = gt_text[:-len(self.tokenizer.eos_token)].strip()

                self.prompts_for_generation.append(prompt_text_for_gen)
                self.ground_truths.append(gt_text)
            else:
                print(f"Warning: Could not find '{end_instruction_token_str}' in example for generation callback: {full_sequence[:100]}...")
                self.prompts_for_generation.append("") # Fallback
                self.ground_truths.append("") # Fallback

        self.num_examples = len(self.prompts_for_generation) # Aggiorna num_examples effettivo
        self.log_steps_interval = log_steps_interval
        self.printed_steps = set() 

        self.max_new_tokens = max_new_tokens

        # Assicurati che pad_token_id sia impostato per la generazione
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _generate_output(self, prompt_text):
        inputs = self.tokenizer(
            prompt_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.tokenizer.model_max_length 
        ).to(self.model.device) 

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            use_cache=True, # Utilizza il caching delle key/value per inferenza più veloce
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Decodifica l'output, saltando i token di input
        # outputs contiene [BOS, INST, ..., /INST, GEN_TOKENS, EOS]
        # inputs.input_ids.shape[1] è la lunghezza della parte [BOS, INST, ..., /INST]
        generated_output_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_output_ids, skip_special_tokens=True).strip()
        
        # Il modello potrebbe generare il prefisso "Output:\n" che fa parte del prompt template
        # Rimuovilo per ottenere il JSON puro
        output_prefix = "Output:\n" # Questo è parte del tuo prompt template negli esempi
        if generated_text.startswith(output_prefix):
            generated_text = generated_text[len(output_prefix):].strip()
            
        # Potrebbe esserci un token </s> alla fine se skip_special_tokens=False fosse stato usato
        # Ma con True, dovrebbe già essere rimosso. Facciamo un controllo extra per sicurezza.
        if generated_text.endswith(self.tokenizer.eos_token): # Solitamente "</s>"
             generated_text = generated_text[:-len(self.tokenizer.eos_token)].strip()

        return generated_text

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Controlla se è il momento di loggare e se non abbiamo già stampato per questo step
        if state.global_step > 0 and state.global_step % self.log_steps_interval == 0 and state.global_step not in self.printed_steps:
            print(f"\n--- Generation Examples at Step {state.global_step} ---")
            self.model.eval() # Imposta il modello in modalità valutazione
            with torch.no_grad():
                for i in range(self.num_examples):
                    prompt = self.prompts_for_generation[i]
                    ground_truth = self.ground_truths[i]
                    
                    generated_output_raw = self._generate_output(prompt)
                    
                    print(f"\n--- Example {i+1} ---")
                    print(f"Prompt (inference input):\n{prompt}")
                    print(f"Ground Truth Output:\n{ground_truth}")
                    print(f"Generated Output (raw):\n{generated_output_raw}")
                    
                    # Usa repair_json per la validazione e il confronto
                    try:
                        gt_parsed_json = json.loads(ground_truth)
                        gen_parsed_json = repair_json(generated_output_raw) # repair_json restituisce stringa
                        gen_parsed_json = json.loads(gen_parsed_json) # Poi parsala in oggetto Python

                        print(f"JSON Output Valid: {True}")
                        
                        # Confronto base: converti a stringa compatta per confronto esatto
                        # Questo può essere troppo restrittivo, meglio una metrica Jaccard o F1 per JSON
                        # Ma per un feedback visivo immediato va bene.
                        if json.dumps(gt_parsed_json, sort_keys=True) == json.dumps(gen_parsed_json, sort_keys=True):
                            print("JSON Match: EXACT")
                        else:
                            print("JSON Match: DIFFERENCE (consider visual inspection)")
                            # Puoi anche stampare le differenze con `difflib` se vuoi
                            # import difflib
                            # diff = list(difflib.unified_diff(
                            #     json.dumps(gt_parsed_json, indent=2).splitlines(keepends=True),
                            #     json.dumps(gen_parsed_json, indent=2).splitlines(keepends=True),
                            #     fromfile='ground_truth', tofile='generated'
                            # ))
                            # print("JSON Diff:")
                            # for line in diff:
                            #     print(line.strip())

                    except json.JSONDecodeError as e:
                        print(f"JSON Output Valid: {False} - Error: {e}")
                        print(f"Raw generated text causing error: {generated_output_raw[:200]}...")
                    except Exception as e: # Per catturare errori da repair_json
                        print(f"Error during JSON repair/parse: {e}")
                        print(f"Raw generated text causing error: {generated_output_raw[:200]}...")
            self.model.train() # Torna il modello in modalità training
            self.printed_steps.add(state.global_step)