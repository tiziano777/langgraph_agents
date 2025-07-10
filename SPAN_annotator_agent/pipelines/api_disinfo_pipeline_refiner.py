import json
import os
import traceback
import logging
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from states.disinfo_state import State 
from nodes.disinfo_apiAnnotator import Annotator
from nodes.disinfo_OutputCorrection import OutputCorrection
from nodes.disinfo_SpanFormat import SpanFormat
from nodes.disinfo_StreamWriter import StreamWriter
from nodes.disinfo_SpanRefine import SpanRefiner 
from utils.CostAnalyze import CostAnalyze

# === Setup Logging ===
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"disinfo_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def route_after_correction(state: State) -> str:
    if not state.refined_once:
        logger.info("Routing: refined_once = False → raffinamento")
        return "refine"
    else:
        logger.info("Routing: refined_once = True → formatting finale")
        return "continue_to_span_format"


def create_pipeline(annotator, output_correction, span_refiner, span_format, writer):
    workflow = StateGraph(State)

    workflow.add_node("annotator_node", annotator)
    workflow.add_node("correction_node", output_correction)
    workflow.add_node("span_refiner_node", span_refiner)
    workflow.add_node("span_node", span_format)
    workflow.add_node("writer_node", writer)

    workflow.add_edge(START, "annotator_node")
    workflow.add_edge("annotator_node", "correction_node")
    workflow.add_conditional_edges(
        "correction_node", route_after_correction,
        {
            "refine": "span_refiner_node",
            "continue_to_span_format": "span_node",
        }
    )
    workflow.add_edge("span_refiner_node", "correction_node")
    workflow.add_edge("span_node", "writer_node")
    workflow.add_edge("writer_node", END)

    pipeline = workflow.compile()
    logger.info("Pipeline compilata correttamente.")
    return pipeline


def run_pipeline(input_path, output_path, checkpoint_path, llm_config, prompts):
    try:
        # === Inizializza LLM ===
        llm = ChatGoogleGenerativeAI(
            model=llm_config["model_name"],
            google_api_key=llm_config["api_key"],
            temperature=llm_config["temperature"],
            max_output_tokens=llm_config["max_output_tokens"],
            top_p=llm_config["top_p"],
            top_k=llm_config.get("top_k", None),
        )
        logger.info(f"LLM inizializzato con modello: {llm_config['model_name']}")

        annotator = Annotator(llm=llm, prompts=prompts, input_context=llm_config['n_ctx'])
        correction = OutputCorrection(similarity_threshold=79)
        span_refiner = SpanRefiner(llm=llm, prompts=prompts) 
        span_format = SpanFormat()
        writer = StreamWriter(output_file=output_path)
        cost_analyzer = CostAnalyze()

        graph = create_pipeline(annotator, correction, span_refiner, span_format, writer)

        # === Carica input ===
        with open(input_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f]
        logger.info(f"Caricati {len(dataset)} esempi da {input_path}")

        # === Checkpoint ===
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f).get("checkpoint", 0)
        else:
            checkpoint = 0
        logger.info(f"Checkpoint iniziale: {checkpoint}")

        for idx in range(checkpoint, len(dataset)):
            item = dataset[idx]

            if item.get('language') not in ['Italian', 'English']:
                logger.warning(f"Salto esempio {idx} con lingua non supportata: {item.get('language')}")
                continue

            initial_state = {
                'text': item['text'].replace("\n", " "),
                'url': item.get('url', ''),
                'title': item.get('title', ''),
                'language': item['language'],
                'refined_once': False
            }

            try:
                logger.info(f"Inizio elaborazione item {idx}")
                state = graph.invoke(initial_state)

                if state.get('error_status') is not None:
                    logger.warning(f"Errore nello stato a checkpoint {idx}: {state['error_status']}")
                    break

            except Exception as e:
                logger.error(f"Errore durante il processing di item {idx}: {e}")
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

        logger.info("Pipeline completata.")

    except Exception as e:
        logger.critical(f"Errore critico nella pipeline: {e}")
        print(traceback.format_exc())
