from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class State(BaseModel):
    
    text: Optional[str] = Field(default=None, description="Testo di input")
    id: Optional[str] = Field(default=None, description="id originale opzionale")
    chunk_id: Optional[str] = Field(default='0', description="chunk di riferimento")
    
    one_shot_example: Optional[str] = Field(default=None, description="RAG shot example, utile per allineare prompt e input con esempi coerenti")
    
    ner: Optional[List[Dict[str, Any]]] = Field(default=[], description="segmentazione NER per ogni elemento di testo")
    
    ner_refined: Optional[List[Dict[str, Any]]] = Field(default=[], description="refined NER per ogni elemento di testo")
       
    error_status: Optional[str] = Field(default=None, description="Stato dell'errore")
    
    input_tokens: Optional[int] = Field(default=0, description="Numero di token di input")
    output_tokens: Optional[int] = Field(default=0, description="Numero di token di input e output")
    
    refine_input_tokens: Optional[int] = Field(default=0, description="Numero di token di input per refiner")
    refine_output_tokens: Optional[int] = Field(default=0, description="Numero di token di input e output per refiner")