import os


USER = os.getenv("USER")
EXPERIMENT = "L_SHIE"

TOML_PATH = f"/home/{USER}/repos/rat-mapping-paper/configs/{EXPERIMENT}.toml"
DATA_PATH = f"/home/{USER}/data/hbmep-processed/rat/{EXPERIMENT}/data.csv"
MEP_MATRIX_PATH = f"/home/{USER}/data/hbmep-processed/rat/{EXPERIMENT}/mat.npy"

INFERENCE_FILE = "inference.pkl"
MODEL_FILE = "model.pkl"

BUILD_DIR = f"/home/{USER}/repos/rat-mapping-paper/reports/log-hierarchical/{EXPERIMENT}"

POSITIONS_MAP = {
    "-C6LC": "-C",
    "C6LC-": "C-",
    "C6LC-C6LX": "C-X",
    "C6LX-C6LC": "X-C",
}
CHARGES_MAP = {
    "50-0-50-100": "Biphasic",
    "20-0-80-25": "Pseudo-Mono"
}
