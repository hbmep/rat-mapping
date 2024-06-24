import os
import pickle
import logging

import numpy as np
import pandas as pd
import arviz as az

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

BUILD_DIR = os.path.join(BUILD_DIR, "midline_lateral")
FEATURES = [["participant", "segment"], "laterality"]


def _process_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logger.info(f"Original data shape: {df.shape}")

    features = ["participant", "compound_position"]

    compound_positions = df[features[1]].unique().tolist()
    logger.info(f"compound_positions: {compound_positions}")

    filtered_compound_positions = []
    for cpos in compound_positions:
        l,r = cpos.split('-')
        if not l:
            continue
        if l[-1] == "L" and r[-1] == "L":
            filtered_compound_positions.append(cpos)
            continue
        if l[-1] == "M" and r[-1] == "M":
            filtered_compound_positions.append(cpos)
            continue

    logger.info(f"filtered_compound_positions: {filtered_compound_positions}")

    ind = df[features[1]].isin(filtered_compound_positions)
    df = df[ind].reset_index(drop=True).copy()
    logger.info(f"Filtered data shape: {df.shape}")

    assert (df.channel1_laterality == df.channel2_laterality).all()

    df["segment"] = df.channel1_segment + "-" + df.channel2_segment
    df["laterality"] = df.channel1_laterality

    combinations = (
        df[[features[0], "segment", "laterality"]]
        .apply(tuple, axis=1)
        .unique()
        .tolist()
    )

    filter_combinations = []
    for c in combinations:
        subject, segment, laterality = c
        if (subject, segment, "L") in combinations and (subject, segment, "M") in combinations:
            filter_combinations.append((subject, segment))

    filter_combinations = set(filter_combinations)
    filter_combinations = list(filter_combinations)

    ind = df[[features[0], "segment"]].apply(tuple, axis=1).isin(filter_combinations)
    df = df[ind].reset_index(drop=True).copy()
    return df


def main():
    df = pd.read_csv(DATA_PATH)
    logger.info("Processing data ...")
    df = _process_data(df=df)
    logger.info(f"Processed df shape: {df.shape}")

    config = Config(TOML_PATH)
    config.FEATURES = FEATURES
    config.BUILD_DIR = BUILD_DIR

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

    # Predictions and recruitment curves
    posterior_samples = posterior_samples_.copy()
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
    os.makedirs(BUILD_DIR, exist_ok=True)
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )
    main()
