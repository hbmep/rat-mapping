import os


USER = os.getenv("USER")

TOML_PATH = f"/home/{USER}/repos/rat-mapping-paper/configs/L_SHIE.toml"
DATA_PATH = f"/home/{USER}/data/hbmep-processed/L_SHIE/data.csv"

INFERENCE_FILE = "inference.pkl"

BUILD_DIR = f"/home/{USER}/repos/rat-mapping-paper/reports/L_SHIE"
