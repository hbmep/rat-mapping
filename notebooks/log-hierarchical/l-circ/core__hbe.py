import os
import sys
import pickle
import logging

import pandas as pd
import numpy as np
from hbmep.config import Config
from hbmep.utils import timing

from paper.utils import setup_logging
from models import (
    HBe,
)
from constants import (
    TOML_PATH,
    DATA_PATH,
    BUILD_DIR,
    INFERENCE_FILE
)

logger = logging.getLogger(__name__)
VERTICES = [
    '-C6LE', '-C6LN', '-C6LNE', '-C6LNW',
    '-C6LS', '-C6LSE', '-C6LSW', '-C6LW'
]
DIAM = [
    'C6LE-C6LW', 'C6LNE-C6LSW', 'C6LS-C6LN', 'C6LSE-C6LNW'
]
RADII = [
    'C6LE-C6LC', 'C6LN-C6LC', 'C6LNE-C6LC', 'C6LNW-C6LC',
    'C6LS-C6LC', 'C6LSE-C6LC', 'C6LSW-C6LC', 'C6LW-C6LC',
]


@timing
def main(M, run_id):
    assert run_id in {"diam", "radii", "vertices", "all"}
    src = DATA_PATH
    data = pd.read_csv(src)

    config = Config(toml_path=TOML_PATH)
    config.BASE = 1
    model = M(config=config)
    model.build_dir = os.path.join(
        BUILD_DIR,
        model.NAME,
        run_id
    )

    # Logging
    os.makedirs(model.build_dir, exist_ok=True)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__)
    )

    ind = data[model.intensity] > 0
    df = data[ind].reset_index(drop=True).copy()

    # subset = ['amap01', 'amap02']
    # ind = df[model.features[0]].isin(subset)
    # df = df[ind].reset_index(drop=True).copy()

    subset = ['-C6LC']
    if run_id == "diam": subset += DIAM
    if run_id == "radii": subset += RADII
    if run_id == "vertices": subset += VERTICES
    if run_id == "all": subset += DIAM + RADII + VERTICES
    ind = df[model.features[1]].isin(subset)
    df = df[ind].reset_index(drop=True).copy()

    # Run inference
    df, encoder_dict = model.load(df=df)
    df[model.intensity] = np.log(df[model.intensity])
    logger.info(f"df.shape {df.shape}")
    logger.info(
        f"Running {model.NAME} with {run_id} - {subset} ..."
    )
    _, posterior_samples = model.run(df=df, **model.run_kwargs)

    # Predictions and recruitment curves
    prediction_df = model.make_prediction_dataset(df=df)
    posterior_predictive = model.predict(
        df=prediction_df, posterior_samples=posterior_samples
    )
    model.render_recruitment_curves(
        df=df,
        encoder_dict=encoder_dict,
        posterior_samples=posterior_samples,
        prediction_df=prediction_df,
        posterior_predictive=posterior_predictive
    )
    model.render_predictive_check(
        df=df,
        encoder_dict=encoder_dict,
        prediction_df=prediction_df,
        posterior_predictive=posterior_predictive
    )

    summary_df = model.summary(posterior_samples)
    logger.info(f"Summary:\n{summary_df.to_string()}")
    dest = os.path.join(model.build_dir, "summary.csv")
    summary_df.to_csv(dest)
    logger.info(f"Saved summary to {dest}")

    # Save
    src = os.path.join(model.build_dir, INFERENCE_FILE)
    with open(src, "wb") as f:
        pickle.dump((df, encoder_dict, model, posterior_samples,), f)

    logger.info(
        f"Finished running {model.NAME} with {run_id} - {subset}"
    )
    logger.info(f"Saved results to {model.build_dir}")
    return


if __name__ == "__main__":
    run_id = sys.argv[1:][0]
    M = HBe
    main(M=M, run_id=run_id)
