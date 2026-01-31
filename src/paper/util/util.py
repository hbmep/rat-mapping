import os
import pickle
import logging 

import hbmep as mep
import numpy as np
from numpyro.infer import MCMC
import pandas as pd

logger = logging.getLogger(__name__)


def get_subname(model: mep.BaseModel):
    return (
        f'{model.mcmc_params["num_warmup"]}w'
        f'_{model.mcmc_params["num_samples"]}s'
        f'_{model.mcmc_params["num_chains"]}c'
        f'_{model.mcmc_params["thinning"]}t'
        f'_{model.nuts_params["max_tree_depth"][0]}d'
        f'_{model.nuts_params["target_accept_prob"] * 100:.0f}a'
        f'_{"t" if model.use_mixture else "f"}m'
    )


def log_transform_fn(df: pd.DataFrame, intensity: str, **kw):
    data = df.copy()
    intensities = sorted(data[intensity].unique().tolist())
    min_intensity = intensities[0]
    assert min_intensity >= 0
    if min_intensity > 0:
        ...
    else:
        replace_zero_with = 1.
        assert replace_zero_with < intensities[1]
        logger.info(f"Replacing {min_intensity} with {replace_zero_with}")
        data[intensity] = data[intensity].replace(
            {min_intensity: replace_zero_with}
        )
        intensities = sorted(data[intensity].unique().tolist())[:5]
        logger.info(f"New minimum intensities: {intensities}")
    data[intensity] = np.log2(data[intensity])
    return data


def run(
    model: mep.BaseModel,
    df: pd.DataFrame,
    encoder: dict | None = None,
    log_transform: bool = True,
    **kw
):
    df = df.copy()
    if encoder is None:
        df, encoder = model.load(df)
    if log_transform:
        df = log_transform_fn(df, **model.variables)
    # model.plot(df, encoder=encoder); return
    logger.info(f"df.shape {df.shape}")
    mcmc, posterior = model.run(df=df, **kw)

    # Save
    output_path = os.path.join(model.build_dir, "inf.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((df, encoder, posterior,), f)
    logger.info(f"Saved to {output_path}")

    output_path = os.path.join(model.build_dir, "model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((model,), f)
    logger.info(f"Saved to {output_path}")

    output_path = os.path.join(model.build_dir, "model_dict.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((model.__dict__,), f)
    logger.info(f"Saved to {output_path}")

    if mcmc is not None:
        output_path = os.path.join(model.build_dir, "mcmc.pkl")
        with open(output_path, "wb") as f:
            pickle.dump((mcmc,), f)
        logger.info(f"Saved to {output_path}")

    predict(model, df, encoder, posterior, mcmc)
    return


def predict(
    model: mep.BaseModel,
    df: pd.DataFrame,
    encoder: dict,
    posterior: dict,
    mcmc: MCMC,
):
    # Predictions
    prediction_df = model.make_prediction_dataset(df=df)
    if mep.site.outlier_prob in posterior.keys():
        posterior[mep.site.outlier_prob] = 0 * posterior[mep.site.outlier_prob]
    predictive = model.predict(prediction_df, posterior=posterior)
    model.plot_curves(
        df=df,
        encoder=encoder,
        prediction_df=prediction_df,
        predictive=predictive,
        posterior=posterior,
    )

    if mep.site.outlier_prob in posterior.keys():
        posterior.pop(mep.site.outlier_prob)
    summary_df = model.summary(posterior)
    logger.info(f"Summary:\n{summary_df.to_string()}")
    dest = os.path.join(model.build_dir, "summary.csv")
    summary_df.to_csv(dest)
    logger.info(f"Saved summary to {dest}")
    logger.info(f"Finished running {model.name}")
    try:
        divergences = mcmc.get_extra_fields()["diverging"].sum().item()
        logger.info(f"No. of divergences {divergences}")
        num_steps = mcmc.get_extra_fields()["num_steps"]
        tree_depth = np.floor(np.log2(num_steps)).astype(int)
        logger.info(f"Tree depth statistics:")
        logger.info(f"Min: {tree_depth.min()}")
        logger.info(f"Max: {tree_depth.max()}")
        logger.info(f"Mean: {tree_depth.mean()}")
    except: pass
    logger.info(f"Saved results to {model.build_dir}")
    return


def load_model(
    model_dir,
    inference_file="inf.pkl",
    model_file="model.pkl",
    mcmc_file="mcmc.pkl",
) -> list[mep.BaseModel, pd.DataFrame, dict, dict, MCMC]:
    src = os.path.join(model_dir, inference_file)
    with open(src, "rb") as f:
        df, encoder, posterior, = pickle.load(f)

    src = os.path.join(model_dir, model_file)
    with open(src, "rb") as f:
        model, = pickle.load(f)

    mcmc = None
    try:
        src = os.path.join(model_dir, mcmc_file)
        with open(src, "rb") as f:
            mcmc, = pickle.load(f)
    except Exception as e:
        logger.info(e)
    else:
        logger.info(f"Found {model_file}")

    if mcmc is None:
        try:
            logger.info(f"Attempting to read model from model_dict.pkl")
            src = os.path.join(model_dir, "model_dict.pkl")
            with open(src, "rb") as f:
                mcmc, _ = pickle.load(f)
        except Exception as e:
            logger.info(e)
        else:
            logger.info("Found model_dict.pkl")

    return model, df, encoder, posterior, mcmc
