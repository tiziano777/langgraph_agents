import requests

class LLMClient:
    """
    Minimal REST client per endpoint LLM (Mistral, etc.).
    La logica di retry è delegata a ErrorHandler.
    """
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url
        self.headers = {
            "Content-Type": "application/json"
        }

    def invoke(self, query_text: str) -> dict:
        """
        Invia una richiesta all'endpoint LLM.

        Args:
            query_text (str): Il prompt da inviare.

        Returns:
            dict: Oggetto JSON restituito dal server, deve contenere almeno:
                  { "success": bool, "response": str }

        Raises:
            requests.exceptions.RequestException: In caso di errore HTTP.
        """
        payload = {"query": query_text}

        response = requests.post(
            url=self.endpoint_url,
            headers=self.headers,
            json=payload,
            timeout=30,  # opzionale: timeout client-side
        )

        response.raise_for_status()
        return response.json()