import json
import os
import yaml
import logging
import traceback
from datetime import datetime

from langchain_community.llms import LlamaCpp
from langgraph.graph import START, END, StateGraph

from states.ner_state import State
from nodes.ner_cppAnnotator import Annotator
from nodes.ner_OutputCorrection import NerCorrection
from nodes.ner_SpanFormat import NerSpanFormat
from nodes.ner_StreamWriter import StreamWriter

# === Setup Logging ===
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"ner_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# === Load LLM Config ===
GEMMA_PATH = "/home/tiziano/AutoAnnotator/src/config/config_gemma3.yml"
with open(GEMMA_PATH, "r", encoding="utf-8") as f:
    llm_config = yaml.safe_load(f)
logger.info(f"Configurazione caricata da {GEMMA_PATH}")


def create_pipeline(annotator: Annotator, ner_correction: NerCorrection,
                    ner_span_format: NerSpanFormat, writer: StreamWriter):

    workflow = StateGraph(State)
    workflow.add_node("annotator_node", annotator)
    workflow.add_node("ner_correction_node", ner_correction)
    workflow.add_node("ner_span_node", ner_span_format)
    workflow.add_node("writer_node", writer)

    workflow.add_edge(START, "annotator_node")
    workflow.add_edge("annotator_node", "ner_correction_node")
    workflow.add_edge("ner_correction_node", "ner_span_node")
    workflow.add_edge("ner_span_node", "writer_node")
    workflow.add_edge("writer_node", END)

    pipeline = workflow.compile()

    try:
        graphImage = pipeline.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(graphImage)
        logger.info("Salvata immagine del grafo in graph.png")
    except Exception as e:
        logger.warning(f"Errore durante la generazione del grafo: {e}")
        logger.debug(traceback.format_exc())

    return pipeline


def run_pipeline(input_path, output_path, checkpoint_path, prompt):
    try:
        model_path = os.path.join(llm_config["model_directory"], llm_config["model_name"])
        user_input_limit = llm_config["user_input_limit"]

        logger.info(f"Inizializzazione LLM da: {model_path}")
        logger.info(f"Prompt di sistema: {prompt}")

        llm = LlamaCpp(
            model_path=str(model_path),
            grammar_path=llm_config["grammar_path"],
            n_gpu_layers=llm_config["n_gpu_layers"],
            n_ctx=llm_config["n_ctx"],
            n_batch=llm_config["n_batch"],
            verbose=llm_config["verbose"],
            repeat_penalty=llm_config["repeat_penalty"],
            temperature=llm_config["temperature"],
            top_k=llm_config["top_k"],
            top_p=llm_config["top_p"],
            streaming=llm_config["streaming"]
        )

        annotator = Annotator(llm=llm, system_prompt=prompt, max_sentence_length=user_input_limit)
        ner_correction = NerCorrection(similarity_threshold=79)
        span_format = NerSpanFormat()
        writer = StreamWriter(output_file=output_path)

        graph = create_pipeline(annotator, ner_correction, span_format, writer)

        # === Load Input Data ===
        with open(input_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        logger.info(f"Caricato dataset con {len(dataset)} voci da {input_path}")

        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f).get("checkpoint", 0)
        logger.info(f"Ripresa dal checkpoint: {checkpoint}")

        for entry in range(checkpoint, len(dataset)):
            text = dataset[entry].get("slovenian_text", "")
            if not text:
                logger.warning(f"Testo vuoto alla riga {entry}, saltato.")
                continue

            try:
                state = graph.invoke({'text': text})

                if state.get('error_status') is not None:
                    logger.warning(f"Errore nello stato a checkpoint {entry}: {state['error_status']}")

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

        logger.info("Esecuzione completata.")

    except Exception as e:
        logger.critical(f"Errore critico durante la run della pipeline: {e}")
        logger.debug(traceback.format_exc())
