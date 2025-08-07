import json
import os
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer, util


class Retriever:
    """
    Classe per recuperare il documento più simile da un pool di file pre-clusterizzati.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Inizializza il Retriever e carica il modello di embedding.
        
        Args:
            model_name (str): Nome del modello di Sentence Transformer da utilizzare.
        """
        self.model = SentenceTransformer(model_name)
        # Dizionario per memorizzare i percorsi dei file
        self.file_pool = {
            "BID": "/home/tiziano/langgraph_agents/NER_annotator_agent/data/output/bid_rag_db.json",
            "TENDER_0": "/home/tiziano/langgraph_agents/NER_annotator_agent/data/output/TENDER0_rag_db.json",
            "TENDER_1": "/home/tiziano/langgraph_agents/NER_annotator_agent/data/output/TENDER1_rag_db.json",
            "ORDER": "/home/tiziano/langgraph_agents/NER_annotator_agent/data/output/ORDER_rag_db.json"
        }
        # Cache per gli embedding pre-calcolati per evitare letture multiple
        self._document_cache = {}

    def _document_routing(self, state: Dict[str, Any]) -> str:
        """
        Seleziona il file JSON corretto in base all'ID e al chunk_id nello stato.
        
        Args:
            state (Dict[str, Any]): Lo stato passato dal LangGraph.
            
        Returns:
            str: Il percorso del file corretto.
        
        Raises:
            ValueError: Se l'ID o il chunk_id non sono validi.
        """
        doc_id = state.get("id", "").upper()
        chunk_id = state.get("chunk_id")
        
        if "TENDER" in doc_id:
            if chunk_id == "0":
                return self.file_pool["TENDER_0"]
            elif chunk_id == "1":
                return self.file_pool["TENDER_1"]
            else:
                raise ValueError("Chunk_id non valido per TENDER.")
        elif "BID" in doc_id:
            return self.file_pool["BID"]
        elif "ORDER" in doc_id:
            return self.file_pool["ORDER"]
        else:
            raise ValueError("ID documento non valido per il routing.")

    def _load_documents(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Carica i documenti da un file JSON e li mette in cache.
        
        Args:
            file_path (str): Il percorso del file JSON.
            
        Returns:
            List[Dict[str, Any]]: Una lista di documenti.
        """
        if file_path in self._document_cache:
            return self._document_cache[file_path]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            self._document_cache[file_path] = documents
            return documents
        except FileNotFoundError:
            raise FileNotFoundError(f"Il file specificato non è stato trovato: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Impossibile decodificare il file JSON: {file_path}")

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Metodo principale per recuperare l'esempio 1-shot.
        
        Args:
            state (Dict[str, Any]): Lo stato passato dal LangGraph. Deve contenere 'text', 'id' e 'chunk_id'.
            
        Returns:
            Dict[str, Any]: Lo stato aggiornato con l'esempio 1-shot.
        """
        input_text = state["text"]
        
        # 1. Routing del documento
        file_path = self._document_routing(state)
        
        # 2. Caricamento dei documenti
        documents = self._load_documents(file_path)
        
        # 3. Embedding dell'input
        input_embedding = self.model.encode(input_text, convert_to_tensor=False)
        
        # 4. Calcolo della similarità e ricerca del più simile
        best_match = None
        max_similarity = -1
        
        for doc in documents:
            doc_embedding = doc["embedding"]
            similarity = util.cos_sim(input_embedding, doc_embedding).item()
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = doc

        if best_match:
            # 5. Creazione dell'esempio 1-shot
            one_shot_example = f"<EXAMPLES>\nExample 1:\input:\n{best_match["text"]}\nOutput\n{str(best_match["ner"])}</EXAMPLES>"
            
            state["one_shot_example"] = one_shot_example
            print(f"Esempio 1-shot recuperato con similarità: {max_similarity:.4f}")
        else:
            state["one_shot_example"] = None
            print("Nessun esempio trovato.")
            
        return state