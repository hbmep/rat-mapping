import os
import logging

import pandas as pd
import numpy as np
import hbmep as mep
from hbmep.util import setup_logging

assert mep.__version__ == "0.7.0"
logger = logging.getLogger(__name__)

# Point this to directory where rat dataset is present
# after cloning this repository
DATA_DIR = "/home/vishu/data/rat-dataset"

# Point this to directory where the output PDFs will be saved
OUTPUT_DIR = "/home/vishu/reports/rat-mapping/plot_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main(experiment):
    intensity = "pulse_amplitude"
    response = ["LADM", "LBiceps", "LDeltoid", "LECR", "LFCR", "LTriceps"]
    features = ["participant", "compound_position"]
    match experiment:
        case "L_CIRC":
            pass
        case "L_SHIE":
            features.append("compound_charge_params")
        case "C_SMA_LAR":
            features.append("compound_size")
        case _:
            raise ValueError
    
    data_path = os.path.join(DATA_DIR, f"{experiment}.csv")
    df = pd.read_csv(data_path)
 
    # # Plot only a subset of the data
    # idx = (df[features[0]].isin(['amap01']))
    # df = df[idx].reset_index(drop=True).copy()

    output_path = os.path.join(OUTPUT_DIR, f"{experiment}.pdf")
    mep.plot(
        df=df,
        intensity=intensity,
        features=features,
        response=response,
        output_path=output_path,
    )
    logger.info(f"Saved to {output_path}")
    return


if __name__ == "__main__":
    setup_logging(OUTPUT_DIR)
    experiments = ["L_CIRC", "L_SHIE", "C_SMA_LAR"]
    for experiment in experiments:
        main(experiment)
