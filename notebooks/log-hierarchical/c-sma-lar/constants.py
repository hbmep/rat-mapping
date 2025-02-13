import os


USER = os.getenv("USER")
EXPERIMENT = "C_SMA_LAR"

TOML_PATH = f"/home/{USER}/repos/rat-mapping-paper/configs/{EXPERIMENT}.toml"
DATA_PATH = f"/home/{USER}/data/hbmep-processed/rat/{EXPERIMENT}/data.csv"
MEP_MATRIX_PATH = f"/home/{USER}/data/hbmep-processed/rat/{EXPERIMENT}/mat.npy"

INFERENCE_FILE = "inference.pkl"
MODEL_FILE = "model.pkl"

BUILD_DIR = f"/home/{USER}/repos/rat-mapping-paper/reports/log-hierarchical/{EXPERIMENT}"

NO_GROUND_SMALL = [
    ('M-LL', 'C5-C5', 'S'),
    ('M-L', 'C5-C5', 'S'),
    ('M-LM1', 'C5-C5', 'S'),
    ('M-LM2', 'C5-C5', 'S'),

    ('M-LL', 'C6-C6', 'S'),
    ('M-L', 'C6-C6', 'S'),
    ('M-LM1', 'C6-C6', 'S'),
    ('M-LM2', 'C6-C6', 'S'),

    ('-M', '-C5', 'S'),
    ('-M', '-C6', 'S')
]
NO_GROUND_BIG = [
    ('M-LL', 'C5-C5', 'B'),
    ('M-L', 'C5-C5', 'B'),
    ('M-LM', 'C5-C5', 'B'),

    ('M-LL', 'C6-C6', 'B'),
    ('M-L', 'C6-C6', 'B'),
    ('M-LM', 'C6-C6', 'B'),

    ('-M', '-C5', 'B'),
    ('-M', '-C6', 'B')
]
WITH_GROUND_BIG = [
    ('-M', '-C5', 'B'),
    ('-LL', '-C5', 'B'),
    ('-L', '-C5', 'B'),
    ('-LM', '-C5', 'B'),

    ('-M', '-C6', 'B'),
    ('-LL', '-C6', 'B'),
    ('-L', '-C6', 'B'),
    ('-LM', '-C6', 'B'),
]
WITH_GROUND_SMALL = [
	('-M', '-C5', 'S'),
	('-LL', '-C5', 'S'),
    ('-L', '-C5', 'S'),
	('-LM1', '-C5', 'S'),
	('-LM2', '-C5', 'S'),

	('-M', '-C6', 'S'),
	('-LL', '-C6', 'S'),
	('-L', '-C6', 'S'),
	('-LM1', '-C6', 'S'),
	('-LM2', '-C6', 'S'),
]

