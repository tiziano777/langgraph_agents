import json
import os
import logging
import traceback


from utils.MistralLoader import MistralLoader 
from langgraph.graph import START, END, StateGraph

from states.ner_state import State

from nodes.preprocessing.ner_Preprocessing import Preprocessor
from nodes.annotators.ner_hfAnnotator import Annotator  
from nodes.writers.ner_JsonLineWriter import StreamWriter

# === Setup Logging ===
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
# Rinomina il file di log per riflettere l'uso del modello Mistral locale
log_filename = os.path.join(log_dir, f"local_mistral_ner_pipeline.log") 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
   
####################################################### 
    
def create_pipeline(preprocessor: Preprocessor, annotator: Annotator, writer: StreamWriter):
    """
    Crea e compila la pipeline LangGraph con un loop tra ner_refiner e ner_schema_validator.
    """
    workflow = StateGraph(State)

    workflow.add_node("preprocessor_node", preprocessor)
    workflow.add_node("annotator_node", annotator)
    workflow.add_node("writer_node", writer)

    workflow.add_edge(START, "preprocessor_node")
    workflow.add_edge("preprocessor_node", "annotator_node")
    workflow.add_edge("annotator_node", "writer_node")
    workflow.add_edge("writer_node", END)

    pipeline = workflow.compile()

    try:
        graphImage = pipeline.get_graph().draw_mermaid_png()
        # Rinomina l'immagine del grafo per riflettere l'uso del modello locale
        with open("images/local_mistral_llm_ner_refine_pipeline.png", "wb") as f:
            f.write(graphImage)
        logger.info("Salvata immagine del grafo in images/local_mistral_llm_ner_refine_pipeline.png")
    except Exception as e:
        logger.warning(f"Errore durante la generazione del grafo: {e}")
        logger.debug(traceback.format_exc())

    return pipeline

def run_pipeline(input_path, output_path, checkpoint_path, llm_config, hf_token):
    """
    Esegue la pipeline di elaborazione NER con il modello Mistral locale.
    """
    
    # Inizializza il caricatore del modello Mistral e ottieni l'istanza LLM
    logger.info("Inizializzazione del modello LLM Mistral locale...")
    llm = MistralLoader(llm_config,hf_token) # Ottieni l'istanza dell'LLM di LangChain pronta per invoke
    logger.info("Modello LLM Mistral locale inizializzato e pronto per l'uso.")

    preprocessor = Preprocessor()
    # input_context è un parametro specifico del tuo Annotator, non direttamente del LLM di HF.
    # Lo manteniamo per coerenza, ma il controllo della dimensione del contesto dell'LLM
    # è gestito da max_new_tokens e dalla max_length del tokenizer interno alla pipeline HF.
    annotator = Annotator(llm=llm)
    writer = StreamWriter(output_file=output_path)
    
    graph = create_pipeline(preprocessor, annotator, writer)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
    # Carica il file dati
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f]
        logger.info(f"Dataset caricato: {len(dataset)} elementi")
    except Exception as e:
        logger.error(f"Errore durante il caricamento del dataset: {e}")
        logger.debug(traceback.format_exc())
        return
    logger.info(f"Caricato dataset con {len(dataset)} voci da {input_path}")

    if not os.path.exists(output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            pass

    if not os.path.exists(checkpoint_path):
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": 0}, f, ensure_ascii=False, indent=4)
    
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f).get("checkpoint", 0)
    logger.info(f"Ripresa dal checkpoint: {checkpoint}")

    for entry in range(checkpoint, len(dataset)):
        text = dataset[entry].get("text", "")
        id= dataset[entry].get("id", "")
        chunk_id= dataset[entry].get("chunk_id", "")
        
        if not text:
            logger.warning(f"Testo vuoto alla riga {entry}, saltato.")
            continue

        try:
            state = graph.invoke({'id': id, 'chunk_id': chunk_id, 'text': text})

            if state['error_status'] is not None:
                logger.warning(f"Errore nello stato a checkpoint {entry}: {state['error_status']}")
                
        except Exception as e:
            logger.error(f"Errore durante l'invocazione della pipeline al checkpoint {entry}: {e}")
            logger.error(traceback.format_exc())
            break

        checkpoint += 1
        logger.info(f"Salvato: {checkpoint}/{len(dataset)}")

        try:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"checkpoint": checkpoint}, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Errore nel salvataggio del checkpoint {checkpoint}: {e}")
            logger.debug(traceback.format_exc())
            break
        
    logger.info("Esecuzione completata.")
