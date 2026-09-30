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

# Ingestion options. Defaults keep older configs working.
_INGESTION = CONFIG.get("ingestion", {}) or {}
_OCR = _INGESTION.get("ocr", {}) or {}
_TABLES = _INGESTION.get("tables", {}) or {}
_CLEANING = _INGESTION.get("cleaning", {}) or {}

OCR_ENABLED = _OCR.get("enabled", True)
OCR_LANGUAGE = _OCR.get("language", "eng")
OCR_DPI = _OCR.get("dpi", 300)
OCR_MIN_TEXT_CHARS = _OCR.get("min_text_chars", 40)
EXTRACT_TABLES = _TABLES.get("extract", True)
CLEANING_ENABLED = _CLEANING.get("enabled", True)
FIX_HYPHENATION = _CLEANING.get("fix_hyphenation", True)
STRIP_PAGE_NUMBERS = _CLEANING.get("strip_page_numbers", True)
REMOVE_FURNITURE = _CLEANING.get("remove_furniture", True)


def ingestion_options() -> dict:
    """Ingestion settings in the keyword form `load_document` expects."""
    return {
        "ocr": OCR_ENABLED,
        "ocr_language": OCR_LANGUAGE,
        "ocr_dpi": OCR_DPI,
        "min_text_chars": OCR_MIN_TEXT_CHARS,
        "extract_tables": EXTRACT_TABLES,
        "clean": CLEANING_ENABLED,
        "fix_hyphenation": FIX_HYPHENATION,
        "strip_page_numbers": STRIP_PAGE_NUMBERS,
        "remove_furniture": REMOVE_FURNITURE,
    }