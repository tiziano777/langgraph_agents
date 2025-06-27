from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class State(BaseModel):
    
    ### CUSTOM LOGIC ###
    
    #first stage, input text, related id, and chunk id, and reference to its labels in memory 
    id: Optional[str] = Field(default=None, description="ID del file di input")
    chunk_id: Optional[Any] = Field(default=None, description="ID del chunk di input")
    chunk_text: Optional[str] = Field(default=None, description="Testo di input")
    labels_path: Optional[str] = Field(default=None, description="Percorso del file delle etichette dell'input text")
    
    # second stage, extracted labels from the labels_path file, LLM adapts labels to OCR text (or viceversa)
    initial_labels: Optional[List[Dict[str, str]]] = Field(default=None, description="Etichette iniziali estratte dal file delle etichette")
    labels: Optional[List[Dict[str, str]]] = Field(default=None, description="Etichette eventually corrected by the LLM")
    
    # third stage, ensure labels span with exact match form input text
    span_ner: Optional[List[Dict[str, Any]]] = Field(default=None, description="Span NER con posizioni nel testo")
    
    error_status: Optional[str] = Field(default=None, description="Stato dell'errore")
    language: Optional[str] = Field(default=None, description="Lingua del testo")
    
    input_tokens: Optional[int] = Field(default=None, description="Numero di token in input")
    output_tokens: Optional[int] = Field(default=None, description="Numero di token in output")