import os


USER = "vishu"
PAPER_DIR = f"/home/{USER}/repos/rat-mapping-paper"
DATA_DIR = f"/home/{USER}/data/hbmep-processed/rat/"

TOML_PATH = os.path.join(PAPER_DIR, "configs", "J_RCML.toml")
DATA_PATH = os.path.join(DATA_DIR, "J_RCML", "data.csv")

BUILD_DIR = os.path.join(PAPER_DIR, "reports", "j-rcml")

INFERENCE_FILE = "inference.pkl"