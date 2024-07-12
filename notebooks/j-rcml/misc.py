import os
import logging

import pandas as pd
import numpy as np
import arviz as az

from hbmep.config import Config
from hbmep.model.utils import Site as site

from paper.utils import setup_logging
from models import Misc
from constants import (
    TOML_PATH,
    DATA_PATH,
    BUILD_DIR
)

logger = logging.getLogger(__name__)

FEATURES = [["participant", "compound_position"]]
RESPONSE = ["LECR"]
COMBINATIONS = [
    ("amap04", "C7L-C8L"),
    ("amap05", "C6L-C7L"),
    ("amap07", "C7L-C8L"),
    # ("amap08", "C5L-C6L"),
]

BUILD_DIR = os.path.join(BUILD_DIR, "misc")


def main():
    df = pd.read_csv(DATA_PATH)
    ind = df[FEATURES[0]].apply(tuple, axis=1).isin(COMBINATIONS)
    df = df[ind].reset_index(drop=True).copy()
    logger.info(f"df: {df.shape}")

    config = Config(toml_path=TOML_PATH)
    config.FEATURES = FEATURES
    config.RESPONSE = RESPONSE
    config.BUILD_DIR = BUILD_DIR

    model = Misc(config=config)

    df, encoder_dict = model.load(df=df)
    # model.plot(df=df, encoder_dict=encoder_dict)

    # Run inference
    mcmc, posterior_samples_ = model.run_inference(df=df)

    # # Save posterior
    # dest = os.path.join(model.build_dir, INFERENCE_FILE)
    # with open(dest, "wb") as f:
    #     pickle.dump((model, mcmc, posterior_samples_,), f)
    # logger.info(f"Saved inference data to {dest}")

    # Predictions and recruitment curves
    posterior_samples = posterior_samples_.copy()
    # posterior_samples[site.outlier_prob] = 0 * posterior_samples[site.outlier_prob]
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

    # Model evaluation
    inference_data = az.from_numpyro(mcmc)
    logger.info("Evaluating model ...")
    logger.info("LOO ...")
    score = az.loo(inference_data, var_name=site.obs)
    logger.info(score)
    logger.info("WAIC ...")
    score = az.waic(inference_data, var_name=site.obs)
    logger.info(score)
    vars_to_exclude = [site.mu, site.alpha, site.beta, site.obs]
    vars_to_exclude += [site.q, site.bg_scale]
    vars_to_exclude = ["~" + var for var in vars_to_exclude]
    logger.info(az.summary(inference_data, var_names=vars_to_exclude).to_string())

    return


if __name__ == "__main__":
    os.makedirs(BUILD_DIR, exist_ok=True)
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )
    main()
