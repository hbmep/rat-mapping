import os
import logging

import hbmep as mep

from paper.model import HB
from paper.util import run, load_shie
from constants import BUILD_DIR, TOML_PATH

logger = logging.getLogger(__name__)

USE_MIXTURE = True
# USE_MIXTURE = not USE_MIXTURE

TEST_RUN = True
TEST_RUN = not TEST_RUN

RESPONSE = ["LADM", "LBiceps", "LDeltoid", "LECR", "LFCR", "LTriceps"]


@mep.timing
def run_model(model: mep.BaseModel):
    run_id = model.run_id
    df = load_shie(**model.variables, run_id=run_id)

    if model.test_run:
        subset = ["amap01", "amap02"]
        idx = df[model.features[0]].isin(subset)
        df = df[idx].reset_index(drop=True).copy()
        model.response = model.response[:3]
        model.mcmc_params["num_warmup"] = 400
        model.mcmc_params["num_samples"] = 400
        model.build_dir = os.path.join(model.build_dir, "test_run")
        os.makedirs(model.build_dir, exist_ok=True)

    log_transform = False
    if "log2" in model._model.__name__:
        log_transform = True

    mep.enable_logging(model.build_dir)
    logger.info(f"*** run id: {run_id} ***")
    logger.info(f"*** model: {model._model.__name__} ***")
    run(model, df, log_transform=log_transform, extra_fields=["num_steps"])
    return


def main(run_id, response=None):
    model = HB(toml_path=TOML_PATH)
    model.use_mixture = USE_MIXTURE
    model.test_run = TEST_RUN
    model.run_id = run_id

    # model._model = model.log2_hb_mvn
    # model._model = model.log2_hb_mvn_gfix

    model._model = model.log2_hb_mvn_mixed

    if response is not None:
        model.response = [response]

    model.mcmc_params = {
        "num_chains": 4,
        "thinning": 1,
        "num_warmup": 4000,
        "num_samples": 4000,
    }
    model.nuts_params = {
        "max_tree_depth": (15, 15),
        "target_accept_prob": .95,
    }

    model.build_dir = os.path.join(
        BUILD_DIR,
        "hb",
        model.name,
        model._model.__name__,
        model.run_id,
    )

    if response is not None:
        assert len(model.response) == 1
        model.build_dir = os.path.join(model.build_dir, f"res_{model.response[0]}")

    run_model(model)
    return


if __name__ == "__main__":
    run_id = "all"
    main(run_id)
