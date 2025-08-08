import json
import os
import logging
import traceback

from utils.CostAnalyze import CostAnalyze
from utils.CostLogger import CostLogger

from utils.llm_simple_client import LLMClient
from langgraph.graph import START, END, StateGraph

from states.ner_state import State

from nodes.preprocessing.ner_Preprocessing import Preprocessor
from nodes.retriever.Retriever import Retriever
from nodes.writers.ner_JsonLineRagWriter import StreamWriter

# === Setup Logging ===
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"api_ner_pipeline.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
    
def create_pipeline(preprocessor: Preprocessor, retriever: Retriever, writer: StreamWriter):

    workflow = StateGraph(State)
    workflow.add_node("preprocessor_node", preprocessor)
    workflow.add_node("retriever_node", retriever)
    workflow.add_node("writer_node", writer)

    workflow.add_edge(START, "preprocessor_node")
    workflow.add_edge("preprocessor_node", "retriever_node")
    workflow.add_edge("retriever_node","writer_node")
    workflow.add_edge("writer_node", END)

    pipeline = workflow.compile()

    try:
        graphImage = pipeline.get_graph().draw_mermaid_png()
        with open("images/remote_rag_llm_ner_pipeline.png", "wb") as f:
            f.write(graphImage)
        logger.info("Salvata immagine del grafo in graph.png")
    except Exception as e:
        logger.warning(f"Errore durante la generazione del grafo: {e}")
        logger.debug(traceback.format_exc())

    return pipeline

def run_pipeline(input_path, output_path, checkpoint_path, llm_config):
    
    # Inizializza il modello LLM
    llm = LLMClient(llm_config)

    preprocessor= Preprocessor()
    retriever = Retriever()
    writer = StreamWriter(output_file=output_path)
    cost_analyzer = CostAnalyze()
    cost_logger = CostLogger()
    
    graph = create_pipeline(preprocessor, retriever, writer)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
    ### LOAD DATA ###
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f]
        logger.info(f"Dataset caricato: {len(dataset)} elementi")
    except Exception as e:
        logger.error(f"Errore durante il caricamento del dataset: {e}")
        logger.debug(traceback.format_exc())
        return
    logger.info(f"Caricato dataset con {len(dataset)} voci da {input_path}")

    ### OUTPUT E CHECKPOINT ###
    if not os.path.exists(output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            pass
    if not os.path.exists(checkpoint_path):
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": 0}, f, ensure_ascii=False, indent=4)
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f).get("checkpoint", 0)
    logger.info(f"Ripresa dal checkpoint: {checkpoint}")
    
    
    ### RUN PIPELINE FOR EACH ENTRY ###
    for entry in range(checkpoint, len(dataset)):
        text = dataset[entry].get("text", "")
        chunk_id = dataset[entry].get("chunk_id", "")
        id = dataset[entry].get("id", "")
        ner = dataset[entry].get("ner", "")
            
        if not text:
            logger.warning(f"Testo vuoto alla riga {entry}, saltato.")
            continue

        try:
            state = graph.invoke({'text': text, 'id': id, 'chunk_id': chunk_id, 'ner': ner})

            if state['error_status'] is not None:
                logger.warning(f"Errore nello stato a checkpoint {entry}: {state['error_status']}")
                
            cost=cost_logger(input_tokens=state['input_tokens'], output_tokens=state['output_tokens'])
            logger.info(cost)
                
        except Exception as e:
            logger.info(traceback.format_exc())
            logger.error(f"Errore durante l'invocazione della pipeline al checkpoint {entry}: {e}")
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
