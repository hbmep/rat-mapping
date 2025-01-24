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
    INFERENCE_FILE,
    POSITIONS_MAP,
    CHARGES_MAP,
)

logger = logging.getLogger(__name__)
REFERENCE = [(' -C', 'Biphasic')]
WITH_GROUND = [
    ('-C', 'Pseudo-Mono'),
    ('C-', 'Pseudo-Mono'),
    ('C-', 'Biphasic'),
]
NO_GROUND = [
    ('C-X', 'Pseudo-Mono'),
    ('C-X', 'Biphasic'),
    ('X-C', 'Pseudo-Mono'),
    ('X-C', 'Biphasic')
]


@timing
def main(M, run_id):
    assert run_id in {"no-ground", "with-ground", "all"}
    src = DATA_PATH
    data = pd.read_csv(src)

    config = Config(toml_path=TOML_PATH)
    config.FEATURES = config.FEATURES[:1] + [config.FEATURES[1:]]
    config.BASE = 1
    model = M(config=config)
    model.build_dir = os.path.join(
        BUILD_DIR,
        model.NAME,
        run_id
    )
    data[model._features[1][0]] = data[model._features[1][0]].replace(POSITIONS_MAP)
    data[model._features[1][1]] = data[model._features[1][1]].replace(CHARGES_MAP)

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

    ind = df[model._features[1]].apply(tuple, axis=1).isin([('-C', 'Biphasic')])
    assert df.loc[ind, model._features[1][0]].nunique() == 1
    df.loc[ind, model._features[1][0]] = " -C"

    subset = REFERENCE
    if run_id == "no-ground": subset += NO_GROUND
    if run_id == "with-ground": subset += WITH_GROUND
    if run_id == "all": subset += NO_GROUND + WITH_GROUND
    ind = df[model._features[1]].apply(tuple, axis=1).isin(subset)
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
    run_id = sys.argv[1]
    M = HBe
    main(M=M, run_id=run_id)
