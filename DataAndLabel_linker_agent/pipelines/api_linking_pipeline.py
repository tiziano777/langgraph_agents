import json
import os
import traceback
from utils.CostAnalyze import CostAnalyze

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from states.linking_state import State

from nodes.linking.linking_apiAnnotator import Annotator
from nodes.linking.linking_OutputCorrection import OutputCorrection
from nodes.linking.linking_SpanFormat import SpanFormat
from nodes.linking.linking_StreamWriter import StreamWriter




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
    return workflow.compile()


def run_pipeline(data_path, label_path, output_path, checkpoint_path, lang, llm_config, prompts):
    
    # Inizializza il modello LLM
    llm = ChatGoogleGenerativeAI(
        model=llm_config["model_name"],
        google_api_key=llm_config["api_key"],
        temperature=llm_config["temperature"],
        max_output_tokens=llm_config["max_output_tokens"],
        top_p=llm_config["top_p"],
        top_k=llm_config.get("top_k", None),
    )
    
    # Inizializza i nodi del grafo con il modello LLM e i prompt
    annotator = Annotator(llm=llm, prompts=prompts, input_context=llm_config['n_ctx'])
    correction = OutputCorrection(similarity_threshold=79)
    span_format = SpanFormat()
    writer = StreamWriter(output_file=output_path)
    cost_analyzer = CostAnalyze()
    
    # Crea il grafo del workflow
    graph = create_pipeline(annotator, correction, span_format, writer)

    # Carica il file dati
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    # Crea il file di checkpoint se non esiste
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f).get("checkpoint", 0)
    else:
        checkpoint = 0

    for idx in range(checkpoint, len(dataset)):
        item = dataset[idx]
        
        
        ####### START CUSTOM LOGIC #######
        
        if item.get('id') is None or item.get('id')=="" or item.get('text') is None or item.get('text')=="":
            continue
        
        try:
            state = graph.invoke({
                'id': item['id'],
                'chunk_id': item.get('chunk_id', 0),  
                
                'chunk_text': item['text'].lower(),
                'labels_path': label_path + item['id'] + '.json',
                
                'language': lang
            })
            if state.get('error_status') is not None:
                print(f'Errore al checkpoint {idx}: {state["error_status"]}')
                break
        except Exception as e:
            print(f"Errore durante il processing: {e}")
            traceback.print_exc()  # <-- stack trace completo
            break
        
        
        ####### END CUSTOM LOGIC #######
        
        print(f"Salvato: {idx + 1}/{len(dataset)}")

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": idx + 1}, f, ensure_ascii=False, indent=4)

        try:
            cost_analyzer.daily_cost(threshold=llm_config['daily_cost_threshold'])
        except RuntimeError as e:
            print(f"Errore nel controllo costi: {e}")
            traceback.print_exc()  # <--  stack trace
            break
