import logging

def setup_logger():
    """Configura el logger."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger
