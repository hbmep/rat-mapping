import os
import logging

import pandas as pd
import numpy as np

import hbmep as mep
from hbmep.model import BaseModel
from hbmep.util import timing, setup_logging

from paper.model import HB
from paper.util import run, log_transform_intensity, load_csmalar_data
from constants import (
    BUILD_DIR,
    TOML_PATH,
    DATA_PATH,
    NO_GROUND,
    GROUND,
    NO_GROUND_SMALL,
    NO_GROUND_BIG,
    GROUND_BIG,
    GROUND_SMALL,
    INBETWEEN_BIG,
    INBETWEEN_SMALL
)


@timing
def main(model):
    # Load data
    src = "/home/vishu/data/hbmep-processed/rat/C_SMA_LAR"
    data_path = os.path.join(src, "data.csv")
    mat_path = os.path.join(src, "mat.npy")
    data = pd.read_csv(data_path)
    mat = np.load(mat_path)
    df, mat = load_csmalar_data(data, mat=mat)
    
    subset = (
        NO_GROUND
        + GROUND
        + NO_GROUND_SMALL
        + NO_GROUND_BIG
        + GROUND_SMALL
        + GROUND_BIG
        + INBETWEEN_BIG
        + INBETWEEN_SMALL
    )
    subset = list(set(subset))
    cols = ["lat", "segment", "compound_size"]

    a = set(subset)
    b = set(df[cols].apply(tuple, axis=1).tolist())
    for u in a:
        if u not in b:
            print(u)

    # assert set(subset) <= set(df[cols].apply(tuple, axis=1).tolist())
    idx = df[cols].apply(tuple, axis=1).isin(subset)
    df = df[idx].reset_index(drop=True).copy()
    mat = mat[idx]
    # output_path = os.path.join(model.build_dir, "unfiltered.pdf")
    # model.plot(df, output_path=output_path)

    hue = np.full((df.shape[0], model.num_response), False)
    hue_columns = [f"hue_{response}" for response in model.response]
    idx = (
        (df.subdir == "/mnt/hdd1/acute_mapping/proc/physio/amap04/2023-03-24_C_SMA2_000")
        & (df.time >= 937.26834688)
    )
    hue[idx, :] = True
    df[hue_columns] = hue
    # output_path = os.path.join(model.build_dir, "flagged.pdf")
    # model.plot(df, output_path=output_path, hue=hue_columns)

    idx = hue.any(axis=1)
    print(f"filter: {idx.sum()}")
    df = df[~idx].reset_index(drop=True).copy()
    mat = mat[~idx]
    # output_path = os.path.join(model.build_dir, "filtered.pdf")
    # model.plot(df, output_path=output_path)

    output_path = "/home/vishu/data/rat-dataset/C_SMA_LAR_inbetween.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    output_path = "/home/vishu/data/rat-dataset/C_SMA_LAR_inbetween.npy"
    np.save(output_path, mat)
    return


if __name__ == "__main__":
    model = BaseModel(toml_path=TOML_PATH)
    model.features = ["participant", "segment", "lat", "compound_size"]
    model.build_dir = os.path.join(BUILD_DIR, "filter")
    main(model)
