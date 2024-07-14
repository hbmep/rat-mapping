import os
import pickle
import logging

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

from hbmep.config import Config
from hbmep.model.utils import Site as site

from paper.utils import setup_logging
from models import HierarchicalBayesianModel
from constants import (
    TOML_PATH,
    DATA_PATH,
    BUILD_DIR,
    INFERENCE_FILE
)

logger = logging.getLogger(__name__)

def main(experiment):
    build_dir = os.path.join(BUILD_DIR, experiment)
    os.makedirs(build_dir, exist_ok=True)
    setup_logging(
        dir=build_dir,
        fname=os.path.basename(__file__)
    )

    df = pd.read_csv(DATA_PATH)
    logger.info("Processing data ...")

    match experiment:
        case "horizontal":
            logger.info(f"Processing {experiment}...")
            # Remove 80-0-20-400 charge
            ind = df.compound_charge_params != "80-0-20-400"
            df = df[ind].reset_index(drop=True).copy()
            # Horizontal
            ind = (
                df.compound_position
                .apply(lambda x: x.split("-"))
                .apply(lambda x: (
                    # Remove ground contacts
                    ("" not in x) and
                    # Horizontal
                    (x[0][:-1] == x[1][:-1])
                ))
            )
            df = df[ind].reset_index(drop=True).copy()
            df["segment"] = df.compound_position.apply(lambda x: x.split("-")[0][:-1])
            df["direction"] = df.compound_position.apply(lambda x: "-".join([u[-1] for u in x.split("-")]))
            df["direction__charge"] = df.direction + "__" + df.compound_charge_params
            df.direction__charge = df.direction__charge.replace({
                'L-M__50-0-50-100': '01__L-M__50-0-50-100',
                'M-L__50-0-50-100': '02__M-L__50-0-50-100',
                'L-M__20-0-80-25': '03__L-M__20-0-80-25',
                'M-L__20-0-80-25': '04__M-L__20-0-80-25',
                'L-M__50-0-50-0': '05__L-M__50-0-50-0',
                'M-L__50-0-50-0': '06__M-L__50-0-50-0'
            })
            FEATURES = [["participant", "segment"], "direction__charge"]
        case "vertical":
            pass
        case _:
            raise ValueError(f"Invalid experiment: {experiment}")

    logger.info(f"Processed df shape: {df.shape}")
    config = Config(toml_path=TOML_PATH)
    config.FEATURES = FEATURES
    config.BUILD_DIR = build_dir

    model = HierarchicalBayesianModel(config=config)
    df, encoder_dict = model.load(df=df)
    # model.plot(df=df, encoder_dict=encoder_dict)

    # Run inference
    mcmc, posterior_samples_ = model.run_inference(df=df)

    # Save posterior
    dest = os.path.join(model.build_dir, INFERENCE_FILE)
    with open(dest, "wb") as f:
        pickle.dump((model, mcmc, posterior_samples_,), f)
    logger.info(f"Saved inference data to {dest}")

    dest = os.path.join(model.build_dir, "encoder_dict.pkl")
    with open(dest, "wb") as f:
        pickle.dump((df, encoder_dict), f)
    logger.info(f"Saved encoder dict to {dest}")

    # Predictions and recruitment curves
    posterior_samples = posterior_samples_.copy()
    if site.outlier_prob in posterior_samples:
        posterior_samples[site.outlier_prob] = 0 * posterior_samples[site.outlier_prob]

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
    experiment = "horizontal"
    main(experiment)
