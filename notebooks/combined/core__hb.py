import os
import sys
import logging

import pandas as pd

from hbmep.util import setup_logging, timing

from paper.model import HB
from paper.util import run
from paper.constants import DATA, REPORTS

logger = logging.getLogger(__name__)

BUILD_DIR = os.path.join(REPORTS, "rat-mapping", "combined")
CONFIG = {
    "variables": {
        "intensity": "pulse_amplitude",
        "features": ["participant", "combination_cdf"],
        "response": ["LADM", "LBiceps", "LDeltoid", "LECR", "LFCR", "LTriceps"]
    }
}


@timing
def main(model):
    run_id = model.run_id
    assert run_id in {
        "L_CIRC___L_SHIE___C_SMA_LAR",
    }
    src = os.path.join(DATA, "rat", f"{run_id}.csv")
    df = pd.read_csv(src)

    if model.test_run:
        model.build_dir = os.path.join(model.build_dir, "test_run")
        os.makedirs(model.build_dir, exist_ok=True)
        subset = ["amap01", "amap02"]
        idx = df[model.features[0]].isin(subset)
        df = df[idx].reset_index(drop=True).copy()
        subset = [
            "SE-NW___L_CIRC",
            "NE-SW___L_CIRC",
            "S-N___L_CIRC",
            "E-W___L_CIRC"
        ]
        idx = df[model.features[1]].isin(subset)
        df = df[idx].reset_index(drop=True).copy()

        model.mcmc_params = {
            "num_chains": 4,
            "thinning": 1,
            "num_warmup": 400,
            "num_samples": 400,
        }

    logger.info(f"*** run id: {run_id} ***")
    logger.info(f"*** model.response: {model.response} ***")
    logger.info(f"*** model: {model._model.__name__} ***")
    run(df, model, extra_fields=["num_steps"])
    return


if __name__ == "__main__":
    model = HB(config=CONFIG)
    model.use_mixture = True
    model.test_run = True
    model.run_id = "L_CIRC___L_SHIE___C_SMA_LAR"
    model._model = model.hb_rl_masked

    response_id = None
    response_id = int(sys.argv[1:][0])
    # response_id = 1
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
        BUILD_DIR,
        model.run_id,
        model.name,
        model._model.__name__,
    )

    if response_id is not None:
        assert model.num_response == 1
        model.build_dir = os.path.join(model.build_dir, model.response[0])
    
    setup_logging(model.build_dir)
    main(model)
