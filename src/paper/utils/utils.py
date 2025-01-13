import os
import glob
import tomllib
import logging
from tqdm import tqdm
from pathlib import Path

import mat73
import numpy as np
import pandas as pd

from hbmep.utils import timing

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(dir, fname, level=logging.INFO):
    fname = f"{fname.split('.')[0]}.log"
    dest = os.path.join(
        dir, fname
    )
    logging.basicConfig(
        format=FORMAT,
        level=level,
        handlers=[
            logging.FileHandler(dest, mode="w"),
            logging.StreamHandler()
        ],
        force=True
    )
    logger.info(f"Logging to {dest}")
    return


@timing
def load_rat_data(
    dir: Path,
    subdir_pattern: list[str] = ["*L_CIRC*"],
    subjects: list[int] = range(1, 9)
):
    df = None

    for p in tqdm(subjects):
        subject = f"amap{p:02}"

        for pattern in subdir_pattern:
            PREFIX = f"{dir}/{subject}/{pattern}"

            subdirs = glob.glob(PREFIX)
            subdirs = sorted(subdirs)

            for subdir in subdirs:

                fpath = glob.glob(f"{subdir}/*auc_table.csv")[0]
                temp_df = pd.read_csv(fpath)

                fpath = glob.glob(f"{subdir}/*ep_matrix.mat")[0]
                data_dict = mat73.loadmat(fpath)

                temp_mat = data_dict["ep_sliced"]

                fpath = glob.glob(f"{subdir}/*cfg_proc.toml")[0]
                with open(fpath, "rb") as f:
                    cfg_proc = tomllib.load(f)

                fpath = glob.glob(f"{subdir}/*cfg_data.toml")[0]
                with open(fpath, "rb") as f:
                    cfg_data = tomllib.load(f)

                temp_df["participant"] = subject
                temp_df["subdir_pattern"] = pattern

                # Rename columns to actual muscle names
                muscles = cfg_data["muscles_emg"]
                muscles_map = {
                    f"auc_{i + 1}": m for i, m in enumerate(muscles)
                }
                temp_df = temp_df.rename(columns=muscles_map).copy()

                # Reorder MEP matrix
                temp_mat = temp_mat[..., np.argsort(muscles)]

                if df is None:
                    df = temp_df.copy()
                    mat = temp_mat

                    time = data_dict["t_sliced"]
                    auc_window = cfg_proc["auc"]["t_slice_minmax"]
                    muscles_sorted = sorted(cfg_data["muscles_emg"])

                    assert len(set(muscles_sorted)) == len(muscles_sorted)
                    continue

                assert (data_dict["t_sliced"] == time).all()
                assert cfg_proc["auc"]["t_slice_minmax"] == auc_window
                assert set(cfg_data["muscles_emg"]) == set(muscles_sorted)

                df = pd.concat([df, temp_df], ignore_index=True).copy()
                mat = np.vstack((mat, temp_mat))

    # Rename df response columns to auc_i
    muscles_map = {
        m: f"auc_{i + 1}" for i, m in enumerate(muscles_sorted)
    }
    df = df.rename(columns=muscles_map).copy()
    df.reset_index(drop=True, inplace=True)

    muscles_map = {
        v: u for u, v in muscles_map.items()
    }

    df = df.rename(columns=muscles_map).copy()
    return df, mat, time, auc_window, muscles_map
