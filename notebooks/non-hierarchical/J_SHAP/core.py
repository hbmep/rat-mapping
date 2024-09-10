import os
import pickle
import logging

import pandas as pd

from hbmep.config import Config
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
def main(M):
    src = DATA_PATH
    data = pd.read_csv(src)

    config = Config(toml_path=TOML_PATH)
    config.BUILD_DIR = os.path.join(
        BUILD_DIR,
        M.NAME
    )
    model = M(config=config)

    # Logging
    os.makedirs(model.build_dir, exist_ok=True)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__)
    )

    # Run inference
    df, encoder_dict = model.load(df=data)
    _, posterior_samples = model.run_inference(df=df)

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

    # Save
    src = os.path.join(model.build_dir, INFERENCE_FILE)
    with open(src, "wb") as f:
        pickle.dump((df, encoder_dict, model, posterior_samples,), f)


if __name__ == "__main__":
    M = NonHierarchicalBayesianModel
    main(M=M)
