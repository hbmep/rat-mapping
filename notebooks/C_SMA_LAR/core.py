import os
import gc
import pickle
import logging

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from hbmep.config import Config
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from paper.utils import setup_logging
from models import (
    NonHierarchicalBayesianModel,
)
from constants import (
    TOML_PATH,
    DATA_PATH,
    BUILD_DIR,
    INFERENCE_FILE
)

logger = logging.getLogger(__name__)


@timing
def main(build_dir):
    src = DATA_PATH
    data = pd.read_csv(src)


    def run_inference(
        subject,
        position,
        response,
        M
    ):
        # Required for build directory
        subject_dir = f"sub__{subject}"
        position_dir = f"pos__{position}"
        response_dir = f"resp__{response}"

        # Build model
        config = Config(toml_path=TOML_PATH)
        config.BUILD_DIR = os.path.join(
            build_dir,
            subject_dir,
            position_dir,
            response_dir,
            M.NAME
        )
        config.RESPONSE = [response]
        model = M(config=config)

        # Set up logging
        model._make_dir(model.build_dir)
        setup_logging(
            dir=model.build_dir,
            fname="logs"
        )

        # Load data
        ind = (
            (data[model.features[0]] == subject) &
            (data[model.features[1]] == position)
        )
        df = data[ind].reset_index(drop=True).copy()

        ind = df[model.response[0]] > 0
        df = df[ind].reset_index(drop=True).copy()

        # Run inference
        df, encoder_dict = model.load(df=df)
        mcmc, posterior_samples = model.run_inference(df=df)

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

        # Save posterior
        dest = os.path.join(model.build_dir, INFERENCE_FILE)
        with open(dest, "wb") as f:
            pickle.dump((model, mcmc, posterior_samples,), f)
        logger.info(f"Saved inference data to {dest}")

        config, df, prediction_df, encoder_dict, _, = None, None, None, None, None
        model, mcmc, posterior_samples, posterior_predictive = None, None, None, None
        del config, df, prediction_df, encoder_dict, _
        del model, mcmc, posterior_samples, posterior_predictive
        gc.collect()

        return


    config = Config(toml_path=TOML_PATH)
    combinations = (
        data[config.FEATURES]
        .apply(tuple, axis=1)
        .unique()
    )

    M = NonHierarchicalBayesianModel

    n_jobs = -1
    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(run_inference)(
                c[0], c[1], response, M
            )
            for c in combinations
            for response in config.RESPONSE
        )


if __name__ == "__main__":
    main(BUILD_DIR)
