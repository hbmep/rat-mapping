import os


USER = os.getenv("USER")
EXPERIMENT = "L_CIRC"

TOML_PATH = f"/home/{USER}/repos/rat-mapping-paper/configs/{EXPERIMENT}.toml"
DATA_PATH = f"/home/{USER}/data/hbmep-processed/rat/{EXPERIMENT}/data.csv"

INFERENCE_FILE = "inference.pkl"

BUILD_DIR = f"/home/{USER}/repos/rat-mapping-paper/reports/non-hierarchical/{EXPERIMENT}"
