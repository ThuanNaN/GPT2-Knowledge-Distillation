from .pretrain_loader import load_model_from_pretrain, load_tokenizer
import logging



def get_logger(name = None):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if name is not None:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(__name__)
    return logger