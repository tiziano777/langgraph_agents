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
from nodes.annotators.ner_RemoteApiAnnotator import Annotator
from nodes.evaluators.ner_RemoteApiRefiner import AnnotatorRefiner
from nodes.writers.ner_JsonLineWriter import StreamWriter

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
    
def create_pipeline(preprocessor: Preprocessor,annotator: Annotator, ner_span_refiner: AnnotatorRefiner, writer: StreamWriter):

    workflow = StateGraph(State)
    workflow.add_node("preprocessor_node", preprocessor)
    workflow.add_node("annotator_node", annotator)
    workflow.add_node("ner_refiner_node", ner_span_refiner)
    workflow.add_node("writer_node", writer)

    workflow.add_edge(START, "preprocessor_node")
    workflow.add_edge("preprocessor_node", "annotator_node")
    workflow.add_edge("annotator_node", "ner_refiner_node")
    workflow.add_edge("ner_refiner_node", "writer_node")
    workflow.add_edge("writer_node", END)

    pipeline = workflow.compile()

    try:
        graphImage = pipeline.get_graph().draw_mermaid_png()
        with open("images/remote_api_llm_ner_refine_pipeline.png", "wb") as f:
            f.write(graphImage)
        logger.info("Salvata immagine del grafo in graph.png")
    except Exception as e:
        logger.warning(f"Errore durante la generazione del grafo: {e}")
        logger.debug(traceback.format_exc())

    return pipeline

def run_pipeline(input_path, output_path, checkpoint_path, base_prompt, refinement_prompt, llm_config):
    
    # Inizializza il modello LLM
    llm = LLMClient(llm_config)

    preprocessor= Preprocessor()
    annotator = Annotator(llm=llm, prompt=base_prompt, input_context=4096)
    refiner = AnnotatorRefiner(llm=llm, prompt=refinement_prompt)
    writer = StreamWriter(output_file=output_path)
    cost_analyzer = CostAnalyze()
    cost_logger = CostLogger()
    
    graph = create_pipeline(preprocessor, annotator, refiner, writer)

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
            
        if not text:
            logger.warning(f"Testo vuoto alla riga {entry}, saltato.")
            continue

        try:
            state = graph.invoke({'text': text})

            if state['error_status'] is not None:
                logger.warning(f"Errore nello stato a checkpoint {entry}: {state['error_status']}")
                
            cost=cost_logger(input_tokens=state['input_tokens'], output_tokens=state['output_tokens'])
            logger.info(cost)
                
        except Exception as e:
            logger.error(f"Errore durante l'invocazione della pipeline al checkpoint {entry}: {e}")
            logger.debug(traceback.format_exc())
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
        
    try:
        cost_analyzer.daily_cost(threshold=llm_config['daily_cost_threshold'])
    except RuntimeError as e:
        logger.error(f"Errore nel controllo costi: {e}")
        logger.debug(traceback.format_exc())
        
    logger.info("Esecuzione completata.")
