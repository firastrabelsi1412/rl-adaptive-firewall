import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT    = os.environ.get("RL_FIREWALL_DATA", "./data")
RAW_DIR      = os.path.join(DATA_ROOT, "raw")
PROC_DIR     = os.path.join(DATA_ROOT, "processed")
MODELS_DIR   = os.path.join(DATA_ROOT, "models")
