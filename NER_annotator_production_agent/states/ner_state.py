from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class State(BaseModel):
    
    text: Optional[str] = Field(default=None, description="Testo di input")
    chunk_id: Optional[str] = Field(default='0', description="chunk di riferimento")
    
    ner: Optional[List[Dict[str, Any]]] = Field(default=[], description="segmentazione NER per ogni elemento di testo")
    ner_refined: Optional[List[Dict[str, Any]]] = Field(default=[], description="refined NER per ogni elemento di testo")
       
    error_status: Optional[str] = Field(default=None, description="Stato dell'errore")
