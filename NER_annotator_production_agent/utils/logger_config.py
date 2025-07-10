import logging
import os
import logging
import traceback
from typing import Dict, Any
from states.ner_state import State 

def setup_pipeline_logger(log_dir: str = "log", log_filename_prefix: str = "ner_pipeline") -> logging.Logger:
    """
    Sets up the main logger for the pipeline, configuring both file and stream handlers.

    Args:
        log_dir (str): The directory where log files will be stored.
        log_filename_prefix (str): The prefix for the log file name.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{log_filename_prefix}.log")

    # Get the root logger or a specific logger (it's often good practice to name your logger)
    # Using __name__ for the main logger is common, or a specific name like 'pipeline_logger'
    logger = logging.getLogger(__name__) # Or logging.getLogger('my_pipeline_logger')
    logger.setLevel(logging.INFO)

    # Prevent adding duplicate handlers if the function is called multiple times
    # This is crucial in scenarios where the setup might be re-triggered.
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream handler (for console output)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)
    
    logger.info("Pipeline logger initialized.")
    return logger

def setup_logger(name: str) -> logging.Logger:
    """
    Configura e restituisce un logger per il modulo specificato.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def handle_exception(state: State, error_string: str, e: Exception) -> Dict[str, Any]:
    """
    Funzione ausiliaria per la gestione centralizzata delle eccezioni.
    Logga errore, traceback ed aggiorna lo stato con un messaggio d'errore coerente.
    """
    full_error = f"{error_string}: {str(e)}"
    full_trace = traceback.format_exc()
    # Stampa anche a console per visibilità immediata durante lo sviluppo
    print(full_error + "\n" + full_trace) 
    
    # Usa un logger specifico per questa funzione, o quello del modulo chiamante se passato
    logger = logging.getLogger(__name__) # Qui usa il logger di questo modulo per gli errori
    logger.error(full_error + "\n" + full_trace)
    
    state.error_status = str(full_error)
    logger.error("STATE ERROR RETURN: %s", {'state': str(state)})
    return {'error_status': full_error}