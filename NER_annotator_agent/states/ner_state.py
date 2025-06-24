from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class State(BaseModel):
    text: Optional[str] = Field(default=None, description="Testo di input")
    
    segmented_text: Optional[List[str]] = Field(default=None, description="Testo di input segmentato")
    segmented_ner: Optional[List[Dict[str, List[str]]]] = Field(default=None, description="segmentazione NER per ogni elemento di testo")
    
    span_ner: Optional[List[Dict[str, Any]]] = Field(default=None, description="Span NER con posizioni")
    
    error_status: Optional[str] = Field(default=None, description="Stato dell'errore")