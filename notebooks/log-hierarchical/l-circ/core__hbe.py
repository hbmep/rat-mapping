import os
import sys
import pickle
import logging

import pandas as pd
import numpy as np
from hbmep.config import Config
from hbmep.utils import timing

from paper.utils import setup_logging
from models import HBe
from constants import (
    TOML_PATH,
    DATA_PATH,
    BUILD_DIR,
    INFERENCE_FILE,
    MODEL_FILE,
    MAP,
    VERTICES,
    DIAM,
    RADII
)

logger = logging.getLogger(__name__)


@timing
def main(model, run_id):
    assert run_id in {"diam", "radii", "vertices", "all"}
    src = DATA_PATH
    data = pd.read_csv(src)
    model.build_dir = os.path.join(
        BUILD_DIR,
        model.NAME,
        model.subname,
        run_id,
    )

    # Logging
    os.makedirs(model.build_dir, exist_ok=True)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__)
    )

    ind = data[model.intensity] > 0
    df = data[ind].reset_index(drop=True).copy()
    df[model.features[1]] = df[model.features[1]].replace(MAP)

    # subset = ['amap01', 'amap02']
    # ind = df[model.features[0]].isin(subset)
    # df = df[ind].reset_index(drop=True).copy()

    subset = []
    if run_id == "diam": subset += DIAM; reference = "-C"; subset += [reference]
    if run_id == "radii": subset += RADII; reference = "-C"; subset += [reference]
    if run_id == "vertices": subset += VERTICES; reference = "S-N"; subset += [reference]
    if run_id == "all": subset += DIAM + RADII + VERTICES; reference = "-C";
    assert set(subset) <= set(df[model.features[1]].values.tolist())
    ind = df[model.features[1]].isin(subset)
    df = df[ind].reset_index(drop=True).copy()
    df[model.features[1]] = df[model.features[1]].replace({reference: " " + reference})

    # Run inference
    df, encoder_dict = model.load(df=df)
    df[model.intensity] = np.log(df[model.intensity])
    logger.info(f"df.shape {df.shape}")
    logger.info(
        f"Running {model.NAME} with {run_id} - {subset} and reference {reference} ..."
    )
    mcmc, posterior_samples = model.run(df=df, **model.run_kwargs)

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
    # model.render_predictive_check(
    #     df=df,
    #     encoder_dict=encoder_dict,
    #     prediction_df=prediction_df,
    #     posterior_predictive=posterior_predictive
    # )

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

    logger.info(f"Finished running {model.NAME} with {run_id} - {subset}")
    logger.info(f"Saved results to {model.build_dir}")
    return


if __name__ == "__main__":
    run_id = sys.argv[1]
    config = Config(toml_path=TOML_PATH)
    config.BASE = 1

    M = HBe
    model = M(config=config)
    model._model = model.mvn_reference
    # model._model = model.mvn_hb_reference
    model.NAME = model._model.__name__
    main(model=model, run_id=run_id)

