import os
import logging
import json
from pathlib import Path

# Configuração de Logs
def setup_logger(name: str, log_file: str = 'app.log', level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    return logger

# Carregar Configurações (Mock para simplicidade, poderia ser via YAML/Env)
CONFIG = {
    "img_size": (224, 224),
    "batch_size": 32,
    "max_text_len": 100,
    "vocab_size": 10000,
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_classes": 2, # 0: Autêntico, 1: Enganoso
    "device": "cuda" if os.environ.get("USE_CUDA") == "1" else "cpu"
}

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

