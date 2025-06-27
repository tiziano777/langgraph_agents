import json
import os
import traceback
import logging
from datetime import datetime

from utils.CostAnalyze import CostAnalyze
from utils.CostLogger import CostLogger

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from states.linking_state import State

from nodes.linking_apiAnnotator import Annotator
from nodes.linking_OutputCorrection import OutputCorrection
from nodes.linking_SpanFormat import SpanFormat
from nodes.linking_StreamWriter import StreamWriter

# === Setup Logging ===
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def create_pipeline(annotator, output_correction, span_format, writer):
    ### NER annotation pipeline ###
    
    workflow = StateGraph(State)
    workflow.add_node("annotator_node", annotator)
    workflow.add_node("correction_node", output_correction)
    workflow.add_node("span_node", span_format)
    workflow.add_node("writer_node", writer)
    
    workflow.add_edge(START, "annotator_node")
    workflow.add_edge("annotator_node", "correction_node")
    workflow.add_edge("correction_node", "span_node")
    workflow.add_edge("span_node", "writer_node")
    workflow.add_edge("writer_node", END)
    
    # Compila il flusso
    pipeline = workflow.compile()

    # Salva immagine grafo (commentato)
    '''
    graphImage = pipeline.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(graphImage)
    '''
 
    return pipeline


def run_pipeline(data_path, label_path, output_path, checkpoint_path, lang, llm_config, prompts):
    logger.info("Avvio della pipeline")
    
    # Inizializza il modello LLM
    llm = ChatGoogleGenerativeAI(
        model=llm_config["model_name"],
        google_api_key=llm_config["api_key"],
        temperature=llm_config["temperature"],
        max_output_tokens=llm_config["max_output_tokens"],
        top_p=llm_config["top_p"],
        top_k=llm_config.get("top_k", None),
    )
    logger.info(f"LLM inizializzato: {llm_config['model_name']}")

    # Inizializza i nodi del grafo con il modello LLM e i prompt
    annotator = Annotator(llm=llm, prompts=prompts, input_context=llm_config['n_ctx'])
    correction = OutputCorrection(similarity_threshold=79)
    span_format = SpanFormat()
    writer = StreamWriter(output_file=output_path)
    cost_analyzer = CostAnalyze()
    cost_logger = CostLogger()

    # Crea il grafo del workflow
    graph = create_pipeline(annotator, correction, span_format, writer)
    logger.info("Pipeline compilata correttamente")

    # Carica il file dati
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f]
        logger.info(f"Dataset caricato: {len(dataset)} elementi")
    except Exception as e:
        logger.error(f"Errore durante il caricamento del dataset: {e}")
        logger.debug(traceback.format_exc())
        return

    # Crea il file di checkpoint se non esiste
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f).get("checkpoint", 0)
    else:
        checkpoint = 0
    logger.info(f"Ripresa dal checkpoint: {checkpoint}")

    for idx in range(checkpoint, len(dataset)):
        item = dataset[idx]

        if item.get('id') is None or item.get('id') == "" or item.get('text') is None or item.get('text') == "":
            logger.warning(f"Record incompleto a index {idx}, saltato.")
            continue

        try:
            ### CUSTOM LOGIC ###
            state = graph.invoke({
                'id': item['id'],
                'chunk_id': item.get('chunk', 0),
                'chunk_text': item['text'],
                'labels_path': label_path + item['id'] + '.json',
                'language': lang
            })

            if state.get('error_status') is not None:
                logger.warning(f"Errore durante l'elaborazione al checkpoint {idx}: {state['error_status']} | {item['id']}")
                continue
            
            cost=cost_logger(input_tokens=state['input_tokens'], output_tokens=state['output_tokens'])
            logger.info(cost)
            
        except Exception as e:
            logger.error(f"Errore durante il processing del record {idx}: {e}")
            logger.debug(traceback.format_exc())
            break

        logger.info(f"Salvato: {idx + 1}/{len(dataset)}")

        try:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"checkpoint": idx + 1}, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Errore nel salvataggio del checkpoint: {e}")
            logger.debug(traceback.format_exc())
            break

    try:
        cost_analyzer.daily_cost(threshold=llm_config['daily_cost_threshold'])
    except RuntimeError as e:
        logger.error(f"Errore nel controllo costi: {e}")
        logger.debug(traceback.format_exc())


    logger.info("Esecuzione pipeline completata")
