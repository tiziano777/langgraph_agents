import traceback
import time
import requests

class ErrorHandler:

    def __init__(self):
        pass

    def extract_retry_delay_from_error(self, e) -> float | None:
        """
        Estrae il retry delay da un'eccezione ResourceExhausted di Gemini.
        """
        try:
            if hasattr(e, 'code') and e.code == 429 and hasattr(e, 'details'):
                retry_info = e.details[2].retry_delay
                return float(retry_info.seconds)
        except Exception as ex:
            print(f"[extract_retry_delay_from_error] Errore: {ex}")
        return None

    def gemini_invoke_with_retry(self, llm, prompt, max_retries=5, retry_count=0):
        """
        Retry per Gemini con gestione del codice 429 e retry_delay.
        """
        try:
            return llm.invoke(prompt)
        
        except Exception as e:
            if hasattr(e, "code") and e.code == 429:
                delay = self.extract_retry_delay_from_error(e)
                if delay:
                    print(f"[gemini_invoke_with_retry] Rate limit: attendo {delay:.2f}s (retry #{retry_count + 1})")
                    time.sleep(delay)
                    if retry_count < max_retries:
                        return self.gemini_invoke_with_retry(llm, prompt, max_retries, retry_count + 1)
                    raise RuntimeError("Max retries exceeded (Gemini).")
                else:
                    raise RuntimeError("Retry delay non disponibile in errore 429 (Gemini).")
            else:
                traceback.print_exc()
                raise RuntimeError(f"Errore Gemini: {e}")

    def rest_invoke_with_retry(self, llm, prompt, max_retries=5, retry_count=0):
        """
        Retry per endpoint REST che restituisce { 'response': ..., 'success': true/false }.
        """
        try:
            result = llm.invoke(prompt)

            if isinstance(result, dict) and result.get("success") is True:
                return result["response"]
            elif isinstance(result, dict) and result.get("success") is False:
                print(f"[rest_invoke_with_retry] Tentativo fallito (retry #{retry_count + 1})")
                if retry_count < max_retries:
                    time.sleep(1)  # backoff statico, puoi renderlo esponenziale
                    return self.rest_invoke_with_retry(llm, prompt, max_retries, retry_count + 1)
                raise RuntimeError("Max retries exceeded (REST).")
            else:
                raise RuntimeError(f"Formato risposta non riconosciuto: {result}")

        except requests.exceptions.RequestException as e:
            traceback.print_exc()
            raise RuntimeError(f"Errore HTTP REST: {e}")
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Errore generico REST: {e}")
