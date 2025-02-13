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
    HierarchicalBayesianModel,
)
from constants import (
    TOML_PATH,
    DATA_PATH,
    BUILD_DIR,
    INFERENCE_FILE,
    MODEL_FILE,
)

logger = logging.getLogger(__name__)


@timing
def main(M, response_ind):
    config = Config(toml_path=TOML_PATH)
    config.BASE = 1
    if response_ind != -1:
        config.RESPONSE = [config.RESPONSE[response_ind]]

    model = M(config=config)
    model.NAME = model.NAME + f"__{model.subname}"
    model.build_dir = os.path.join(
        BUILD_DIR,
        model.NAME,
        f"response_{response_ind}"
    )

    # Logging
    os.makedirs(model.build_dir, exist_ok=True)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__)
    )

    src = DATA_PATH
    data = pd.read_csv(src)
    ind = data[model.intensity] > 0
    df = data[ind].reset_index(drop=True).copy()
    df[model.intensity] = np.log(df[model.intensity])
    df, encoder_dict = model.load(df=df)

    # Run inference
    ind = df[model.features[0]] < 2
    df = df[ind].reset_index(drop=True).copy()
    ind = df[model.features[1]] < 1
    df = df[ind].reset_index(drop=True).copy()
    ind = df[model.features[2]] < 1
    df = df[ind].reset_index(drop=True).copy()

    logger.info(f"df.shape {df.shape}")
    logger.info(f"Running {M.NAME} with response {response_ind} - {model.response}")
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
        pickle.dump((df, encoder_dict, mcmc, posterior_samples,), f)

    src = os.path.join(model.build_dir, MODEL_FILE)
    with open(src, "wb") as f:
        pickle.dump((model,), f)

    logger.info(f"Finished running {model.NAME}")
    logger.info(f"Saved results to {model.build_dir}")
    return


if __name__ == "__main__":
    response_ind, = list(map(int, sys.argv[1:]))
    # response_ind = -1
    M = HB
    main(M=M, response_ind=response_ind)

