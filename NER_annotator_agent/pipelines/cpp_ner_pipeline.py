import json
import os
import yaml
from langchain_community.llms import LlamaCpp
from langgraph.graph import START, END, StateGraph
from states.ner_state import State
from nodes.ner_cppAnnotator import Annotator
from nodes.ner_OutputCorrection import NerCorrection
from nodes.ner_SpanFormat import NerSpanFormat
from nodes.ner_StreamWriter import StreamWriter


# Crea le istanze dei tuoi oggetti
GEMMA_PATH = "/home/tiziano/AutoAnnotator/src/config/config_gemma3.yml"
with open(GEMMA_PATH, "r", encoding="utf-8") as f:
    llm_config = yaml.safe_load(f)

def create_pipeline(annotator: Annotator, ner_correction: NerCorrection,ner_span_format: NerSpanFormat, writer: StreamWriter):

    # Crea il grafo del flusso
    workflow = StateGraph(State)
    
    # Aggiungi nodi
    start_node = START
    annotator_node_name = "annotator_node"
    ner_correction_node_name = "ner_correction_node"
    ner_span_format_name= "ner_span_node"
    writer_node_name = "writer_node"
    end_node = END

    # Aggiungi i nodi alla pipeline
    workflow.add_node(annotator_node_name, annotator)
    workflow.add_node(ner_correction_node_name, ner_correction)
    workflow.add_node(ner_span_format_name, ner_span_format)
    workflow.add_node(writer_node_name, writer)

    # Collega i nodi
    workflow.add_edge(start_node, annotator_node_name)
    workflow.add_edge(annotator_node_name, ner_correction_node_name)
    workflow.add_edge(ner_correction_node_name,ner_span_format_name)
    workflow.add_edge(ner_span_format_name,writer_node_name)
    workflow.add_edge(writer_node_name, end_node)

    # Compila il flusso
    pipeline = workflow.compile()

    # Salva immagine grafo
    graphImage=pipeline.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(graphImage)
 
    return pipeline

# Esegui la pipeline su una lista di testi
def run_pipeline(input_path,output_path,checkpoint_path, prompt):
    # Estrai i parametri dalla configurazione
    model_path = os.path.join(llm_config["model_directory"], llm_config["model_name"])
    user_input_limit = llm_config["user_input_limit"]
    
    print(model_path)
    print(prompt)
    
    # Inizializza LLM da file di configurazione
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
    
    system_prompt = prompt
    
    annotator = Annotator(llm=llm, system_prompt=system_prompt, max_sentence_length=user_input_limit)
    ner_correction = NerCorrection(similarity_threshold=79)
    span_format = NerSpanFormat()
    writer = StreamWriter(output_file=output_path)

    # Crea la pipeline LangGraph
    graph = create_pipeline(annotator, ner_correction,span_format, writer)
    
    
    #INPUT PHASE:
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f) 

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        check = json.load(f)  
    check=check['checkpoint']
    
    for entry in range(check,len(dataset)):
        text = dataset[entry].get("slovenian_text", "")
        if text: 
            state=graph.invoke({'text':text})
            if state.get('error_status') is not None:
                print(f'checkpoint: {check}')
                print(f"state: {state}")
    
        
        check+=1
        print(f'saved {check}/{len(dataset)}')
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": check}, f, ensure_ascii=False, indent=4)
        