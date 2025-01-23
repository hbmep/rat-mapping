import os


USER = os.getenv("USER")
EXPERIMENT = "L_CIRC"

TOML_PATH = f"/home/{USER}/repos/rat-mapping-paper/configs/{EXPERIMENT}.toml"
DATA_PATH = f"/home/{USER}/data/hbmep-processed/rat/{EXPERIMENT}/data.csv"
MEP_MATRIX_PATH = f"/home/{USER}/data/hbmep-processed/rat/{EXPERIMENT}/mat.npy"

INFERENCE_FILE = "inference.pkl"

BUILD_DIR = f"/home/{USER}/repos/rat-mapping-paper/reports/log-hierarchical/{EXPERIMENT}"
