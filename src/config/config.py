import yaml
from pathlib import Path

def _load_config():
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

CONFIG = _load_config()
CHUNK_SIZE = CONFIG["chunker"]["chunk_size"]
OVERLAP = CONFIG["chunker"]["overlap"]
SEPARATORS = CONFIG["chunker"]["separator"]

MODEL_NAME = CONFIG["embedding"]["model_name"]
BATCH_SIZE = CONFIG["embedding"]["batch_size"]
DEVICE = CONFIG["embedding"]["device"]