from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class State(BaseModel):
    text: Optional[str] = Field(default=None, description="Testo di input")
    chunk_id:Optional[int] = Field(default=None, description="ID del chunk di testo")
    id: Optional[str] = Field(default=None, description="ID del testo")

    ner: Optional[List[Dict[str, str]]] = Field(default=[], description="segmentazione NER per ogni elemento di testo")
    corrected_ner: Optional[Dict[str, str]]= Field(default={}, description="segmentazione NER eventualmente corretta e allineata al testo per ogni entità")
    span_ner: Optional[List[Dict[str, Any]]] = Field(default=[], description="Span NER con posizioni")
    
    error_status: Optional[str] = Field(default=None, description="Stato dell'errore")
    
    input_tokens: Optional[int] = Field(default=0, description="Numero di token di input")
    output_tokens: Optional[int] = Field(default=0, description="Numero di token di input e output")