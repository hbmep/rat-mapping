import os
import sys
import logging

import pandas as pd
import numpy as np
from hbmep.util import timing, setup_logging

from paper.model import HB
from paper.util import load_lat, run
from constants import BUILD_DIR, TOML_PATH

logger = logging.getLogger(__name__)


@timing
def main(model):
    run_id = model.run_id
    df = load_lat(**model.variables, run_id=run_id)

    if model.test_run:
        model.build_dir = os.path.join(model.build_dir, "test_run")
        os.makedirs(model.build_dir, exist_ok=True)
        subset = ["amap01", "amap02"]
        idx = df[model.features[0]].isin(subset)
        df = df[idx].reset_index(drop=True).copy()
        model.response = model.response[:3]
        model.mcmc_params = {
            "num_chains": 4,
            "thinning": 1,
            "num_warmup": 400,
            "num_samples": 400,
        }

    run(df, model, extra_fields=["num_steps"])
    return


if __name__ == "__main__":
    model = HB(toml_path=TOML_PATH)
    model.features = ["participant", "segment", "lat"]
    model.use_mixture = True
    model.test_run = False

    # model._model = model.hb_rl_masked
    model._model = model.hb_mvn_rl_masked
    # model._model = model.robust_hb_mvn_rl_masked

    # model.run_id = "lat-small-ground"
    # model.run_id = "lat-big-ground"
    model.run_id = "lat-small-inbetween"
    # model.run_id = "lat-big-inbetween"

    response_id = None
    # response_id = int(sys.argv[1:][0])
    response_id = 1
    if response_id is not None:
        model.response = model.response[response_id: response_id + 1]

    model.mcmc_params = {
        "num_chains": 4,
        "thinning": 4,
        "num_warmup": 4000,
        "num_samples": 4000,
    }
    model.nuts_params = {
        "max_tree_depth": (15, 15),
        "target_accept_prob": .95,
    }

    model.build_dir = os.path.join(
        BUILD_DIR, "hb", model.name, model.run_id, model._model.__name__
    )

    if response_id is not None:
        assert model.num_response == 1
        model.build_dir = os.path.join(model.build_dir, model.response[0])

    setup_logging(model.build_dir)
    main(model)
