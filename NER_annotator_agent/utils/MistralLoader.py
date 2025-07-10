import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
# from langchain_huggingface import HuggingFacePipeline # Non la useremo direttamente in invoke
# from langchain_core.messages import HumanMessage # Non la useremo direttamente in invoke
import yaml # Aggiunto per caricare il config
import os # Aggiunto per os.environ.get

logger = logging.getLogger(__name__)

class MistralLoader:
    """
    Classe per caricare un LLM Mistral (non quantizzato) usando Hugging Face Transformers.
    Questa classe incapsula il modello e la logica di formattazione del prompt,
    permettendo di invocare direttamente il modello con prompt e query.
    """

    def __init__(self, llm_config: dict, hf_token: str = None): # hf_token aggiunto come parametro
        """
        Inizializza il caricatore del modello Mistral.

        Args:
            llm_config (dict): Dizionario di configurazione contenente i parametri per
                                il modello (es. model_name, generation_params).
            hf_token (str): Token di autenticazione per Hugging Face Hub.
        """
        self.model_name = llm_config.get("model_name", "mistralai/Mistral-7B-Instruct-v0.3")
        self.llm_config = llm_config
        self.hf_token = hf_token # Salviamo il token

        logger.info(f"Caricamento del modello Mistral originale (non quantizzato): {self.model_name}")

        torch_dtype_str = llm_config.get("torch_dtype", "bfloat16")
        try:
            self.torch_dtype = getattr(torch, torch_dtype_str)
        except AttributeError:
            logger.warning(f"Tipo di dato torch '{torch_dtype_str}' non valido. Usando torch.bfloat16 come default.")
            self.torch_dtype = torch.bfloat16

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                token=self.hf_token # Usa self.hf_token
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token) # Usa self.hf_token
            logger.info("Modello e tokenizer caricati con successo (non quantizzato).")
        except Exception as e:
            logger.error(f"Errore durante il caricamento del modello o tokenizer: {e}")
            raise

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        # Carica i token di istruzione e il contenuto del prompt dal config
        self.START_INSTRUCTION_TOKEN = self.llm_config.get("START_INSTRUCTION_TOKEN", "<s>[INST]")
        self.END_INSTRUCTION_TOKEN = self.llm_config.get("END_INSTRUCTION_TOKEN", "[/INST]")
        self.PROMPT_TEMPLATE_CONTENT = self.llm_config.get("TENDER_PROMPT", "") # Assicurati che il nome sia corretto

        self.generation_params = {
            "max_new_tokens": int(self.llm_config.get("max_output_tokens", 800)),
            "temperature": float(self.llm_config.get("temperature", 0.1)), # Ho messo 0.1 di default qui per consistenza con il tuo config
            "do_sample": bool(self.llm_config.get("do_sample", False)), # Ho messo False di default qui per consistenza con il tuo config
            "top_p": float(self.llm_config.get("top_p", 0.5)),
            "top_k": int(self.llm_config.get("top_k", 10)),
            "repetition_penalty": float(self.llm_config.get("repetition_penalty", 1.11)),
            "return_full_text": False,  # Cruciale: restituisce solo il testo generato
            "pad_token_id": self.tokenizer.pad_token_id
        }
        # Aggiungi 'eos_token_id' per fermare la generazione al token di fine sequenza (o al [/INST])
        # Mistral-instruct usa </s> per la fine della sequenza di risposta.
        # Spesso, se il modello è fine-tunato per generare solo JSON, si può aggiungere ']' o '}' come stop token.
        # Per ora, lasciamo che sia '</s>' o max_new_tokens a fermare la generazione.
        self.generation_params["eos_token_id"] = self.tokenizer.eos_token_id


        try:
            self.hf_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                **self.generation_params
            )
            logger.info("Pipeline di Hugging Face creata con successo.")
        except Exception as e:
            logger.error(f"Errore durante la creazione della pipeline di Hugging Face: {e}")
            raise

        # La LangChain HuggingFacePipeline non verrà usata direttamente per l'invocazione
        # nel modo specifico che abbiamo definito per questo caso,
        # ma l'import può rimanere se è usata altrove nel progetto.
        # self._llm_instance = HuggingFacePipeline(pipeline=self.hf_pipeline)
        # logger.info("LLM di LangChain (HuggingFacePipeline) creato e pronto.")

    def invoke(self, chunk_id: str, user_input_text: str) -> str:
        """
        Invoca il modello Mistral con la query utente, formattando il prompt
        secondo lo schema di training per l'inferenza.

        Args:
            chunk_id (str): Il chunk_id dell'esempio.
            user_input_text (str): Il testo effettivo da cui estrarre le entità.

        Returns:
            str: Il testo JSON generato dal modello.
        """
        # Costruiamo il prompt completo esattamente come lo desideri per l'inferenza
        # Questo prompt sarà passato direttamente alla pipeline di Hugging Face.
        # Include i token di istruzione, il contenuto del prompt, l'input e "Output:"
        
        # NOTA: l'f-string aggiunge automaticamente i newline se il multiline string del YAML li contiene.
        # Assicurati che TENDER_PROMPT_CONTENT nel YAML non abbia newlines extra all'inizio/fine.
        
        # Questo è il prompt esatto che il modello riceverà per l'inferenza.
        # Il modello dovrebbe generare solo la parte JSON + </s>
        inference_prompt = (
            f"{self.START_INSTRUCTION_TOKEN}\n"  # e.g., <s>[INST]
            f"{self.PROMPT_TEMPLATE_CONTENT}\n"  # Your detailed prompt content
            f"{self.END_INSTRUCTION_TOKEN}\n"    # e.g., [/INST]
            f"Input:\nchunk_id: {chunk_id}\n{user_input_text}\n"
            f"Output:\n"                         # Cruciale: questo fa parte del prompt di input
        )
        
        #print(f"Prompt di inferenza generato:\n---\n{inference_prompt}\n---")

        outputs = self.hf_pipeline(inference_prompt)

        # L'output è una lista di dizionari, prendiamo il testo generato dalla prima (e unica) risposta.
        # outputs[0]["generated_text"] conterrà il JSON + </s>
        generated_text = outputs[0]["generated_text"]

        # Rimuoviamo il </s> se presente, dato che l'output desiderato è solo il JSON pulito.
        if generated_text.endswith("</s>"):
            generated_text = generated_text[:-len("</s>")].strip()
        
        # Opzionale: pulire anche eventuali spazi bianchi extra all'inizio o alla fine del JSON
        generated_text = generated_text.strip()
        
        return generated_text