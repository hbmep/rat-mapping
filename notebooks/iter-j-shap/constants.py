import os


USER = os.getenv("USER")

TOML_PATH = f"/home/{USER}/repos/rat-mapping-paper/configs/J_SHAP.toml"
DATA_PATH = f"/home/{USER}/data/hbmep-processed/rat/J_SHAP/data.csv"

INFERENCE_FILE = "inference.pkl"

BUILD_DIR = f"/home/{USER}/repos/rat-mapping-paper/reports/iter-j-shap"
