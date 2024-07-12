import logging

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer import autoguide

from hbmep.config import Config
from hbmep.nn import functional as F
from hbmep.model import GammaModel
from hbmep.model.utils import Site as site
from hbmep.utils import timing

logger = logging.getLogger(__name__)


class HierarchicalBayesianModel(GammaModel):
    NAME = "hierarchical_bayesian_model"

    def __init__(self, config: Config):
        super(HierarchicalBayesianModel, self).__init__(config=config)

    def _model(self, intensity, features, response_obs=None, subsample_size=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        n_fixed = 1
        n_delta = n_features[1] - 1

        # Fixed
        a_fixed_loc = numpyro.sample(
            "a_fixed_loc", dist.TruncatedNormal(150., 100., low=0)
        )
        a_fixed_scale = numpyro.sample(
            "a_fixed_scale", dist.HalfNormal(100.)
        )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed = numpyro.sample(
                        "a_fixed", dist.TruncatedNormal(
                            a_fixed_loc, a_fixed_scale, low=0
                        )
                    )

        # Delta
        a_delta_loc_loc = numpyro.sample(
            "a_delta_loc_loc", dist.Normal(0., 100.)
        )
        a_delta_loc_scale = numpyro.sample(
            "a_delta_loc_scale", dist.HalfNormal(100.)
        )

        a_delta_scale = numpyro.sample(
            "a_delta_scale", dist.HalfNormal(100.)
        )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_delta", n_delta):
                a_delta_loc = numpyro.sample(
                    "a_delta_loc", dist.Normal(a_delta_loc_loc, a_delta_loc_scale)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_delta = numpyro.sample(
                        "a_delta", dist.Normal(a_delta_loc, a_delta_scale)
                    )

                    # Penalty for negative a
                    penalty_for_negative_a = (
                        jnp.fabs(a_fixed + a_delta) - (a_fixed + a_delta)
                    )
                    numpyro.factor(
                        "penalty_for_negative_a", -penalty_for_negative_a
                    )

        # Hyper-priors
        b_scale = numpyro.sample(
            "b_scale", dist.HalfNormal(5.)
        )

        L_scale = numpyro.sample(
            "L_scale", dist.HalfNormal(.5)
        )
        ell_scale = numpyro.sample(
            "ell_scale", dist.HalfNormal(10.)
        )
        H_scale = numpyro.sample(
            "H_scale", dist.HalfNormal(5.)
        )

        c_1_scale = numpyro.sample(
            "c_1_scale", dist.HalfNormal(5.)
        )
        c_2_scale = numpyro.sample(
            "c_2_scale", dist.HalfNormal(5.)
        )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed + a_delta], axis=1)
                    )
                    b = numpyro.sample("b", dist.HalfNormal(scale=b_scale))

                    L = numpyro.sample("L", dist.HalfNormal(scale=L_scale))
                    ell = numpyro.sample("ell", dist.HalfNormal(scale=ell_scale))
                    H = numpyro.sample("H", dist.HalfNormal(scale=H_scale))

                    c_1 = numpyro.sample("c_1", dist.HalfNormal(scale=c_1_scale))
                    c_2 = numpyro.sample("c_2", dist.HalfNormal(scale=c_2_scale))

        # Outlier Distribution
        outlier_prob = numpyro.sample(site.outlier_prob, dist.Uniform(0., .01))
        outlier_scale = numpyro.sample(site.outlier_scale, dist.HalfNormal(10))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                # Model
                mu = numpyro.deterministic(
                    site.mu,
                    F.rectified_logistic(
                        x=intensity,
                        a=a[feature0, feature1],
                        b=b[feature0, feature1],
                        L=L[feature0, feature1],
                        ell=ell[feature0, feature1],
                        H=H[feature0, feature1]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    self.rate(
                        mu,
                        c_1[feature0, feature1],
                        c_2[feature0, feature1]
                    )
                )
                alpha = numpyro.deterministic(
                    site.alpha,
                    self.concentration(mu, beta)
                )

                # Mixture
                q = numpyro.deterministic(
                    site.q, outlier_prob * jnp.ones((n_data, self.n_response))
                )
                bg_scale = numpyro.deterministic(
                    site.bg_scale,
                    outlier_scale * jnp.ones((n_data, self.n_response))
                )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data, subsample_size=subsample_size) as ind:

                mixing_distribution = dist.Categorical(
                    probs=jnp.stack(
                        [
                            1 - (jnp.take(q, ind, axis=0) if subsample_size is not None else q),
                            jnp.take(q, ind, axis=0) if subsample_size is not None else q
                        ],
                        axis=-1
                    )
                )
                component_distributions=[
                    dist.Gamma(
                        concentration=jnp.take(alpha, ind, axis=0) if subsample_size is not None else alpha,
                        rate=jnp.take(beta, ind, axis=0) if subsample_size is not None else beta
                    ),
                    dist.HalfNormal(
                        scale=jnp.take(bg_scale, ind, axis=0) if subsample_size is not None else bg_scale
                    )
                ]

                Mixture = dist.MixtureGeneral(
                    mixing_distribution=mixing_distribution,
                    component_distributions=component_distributions
                )

                # Observation
                numpyro.sample(
                    site.obs,
                    Mixture,
                    obs=jnp.take(response_obs, ind, axis=0) if response_obs is not None else response_obs
                )


class SVIHierarchicalBayesianModel(HierarchicalBayesianModel):
    NAME = "svi_hierarchical_bayesian_model"

    def __init__(self, config: Config):
        super(SVIHierarchicalBayesianModel, self).__init__(config=config)

    @timing
    def run_inference(
        self,
		df,
		num_steps = 10000,
		step_size = 1e-2,
		num_particles = 100,
        subsample_size=100
    ):
        loss = Trace_ELBO(num_particles=num_particles)
        optimizer = numpyro.optim.ClippedAdam(step_size=step_size)
        _guide = autoguide.AutoLowRankMultivariateNormal(self._model)
        # _guide = autoguide.AutoMultivariateNormal(self._model)
        svi = SVI(self._model, _guide, optimizer, loss=loss)
        svi_result = svi.run(
            self.rng_key,
            num_steps,
            *self._get_regressors(df),
            *self._get_response(df=df),
            subsample_size=subsample_size
        )
        losses = svi_result.losses

        if np.isnan(losses).any():
            logger.info("NaNs in losses")

        num_samples = int(
            (self.mcmc_params["num_samples"] * self.mcmc_params["num_chains"])
            // self.mcmc_params["thinning"]
        )
        predictive = Predictive(
            _guide,
            params=svi_result.params,
            num_samples=num_samples
        )
        posterior_samples = predictive(self.rng_key, *self._get_regressors(df=df))
        posterior_samples = {u: np.array(v) for u, v in posterior_samples.items()}
        return _guide, svi_result
