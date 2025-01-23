import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep import functional as F
from hbmep import smooth_functional as S
from hbmep.model import GammaModel
from hbmep.model.utils import Site as site

EPS = 1e-3


class HierarchicalBayesianModel(GammaModel):
    NAME = "hierarchical_bayesian"

    def __init__(self, config: Config):
        super(HierarchicalBayesianModel, self).__init__(config=config)
        self.mcmc_params = {
            "num_warmup": 4000,
            "num_samples": 4000,
            "num_chains": 4,
            "thinning": 4,
        }
        self.run_kwargs = {
            "max_tree_depth": (20, 20),
            "target_accept_prob": .95,
            "extra_fields": [
                "potential_energy",
                "num_steps",
                "accept_prob",
            ]
        }
        self.NAME += (
            f'_{self.mcmc_params["num_warmup"]}W'
            f'_{self.mcmc_params["num_samples"]}S'
            f'_{self.mcmc_params["num_chains"]}C'
            f'_{self.mcmc_params["thinning"]}T'
            f'_{self.run_kwargs["max_tree_depth"][0]}D'
            f'_{self.run_kwargs["target_accept_prob"]}A'
        )

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        # Hyper Priors
        a_loc = numpyro.sample("a_loc", dist.Normal(3., 5.,))
        a_scale = numpyro.sample("a_scale", dist.HalfNormal(5.))

        b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))
        L_scale = numpyro.sample("L_scale", dist.HalfNormal(.1))
        ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(1.))
        H_scale = numpyro.sample("H_scale", dist.HalfNormal(5.))

        c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
        c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(.5))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a_raw = numpyro.sample("a_raw", dist.Normal(0, 1))
                    a = numpyro.deterministic(site.a, a_loc + jnp.multiply(a_scale, a_raw))

                    b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                    b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                    L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                    L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

                    ell_raw = numpyro.sample("ell_raw", dist.HalfNormal(scale=1))
                    ell = numpyro.deterministic(site.ell, jnp.multiply(ell_scale, ell_raw))

                    H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
                    H = numpyro.deterministic(site.H, jnp.multiply(H_scale, H_raw))

                    c_1_raw = numpyro.sample("c_1_raw", dist.HalfNormal(scale=1))
                    c_1 = numpyro.deterministic(site.c_1, jnp.multiply(c_1_scale, c_1_raw))

                    c_2_raw = numpyro.sample("c_2_raw", dist.HalfNormal(scale=1))
                    c_2 = numpyro.deterministic(site.c_2, jnp.multiply(c_2_scale, c_2_raw))

        # Outlier Distribution
        q = numpyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                # Model
                mu = numpyro.deterministic(
                    site.mu,
                    S.rectified_logistic(
                        x=intensity,
                        a=a[feature0, feature1],
                        b=b[feature0, feature1],
                        L=L[feature0, feature1],
                        ell=ell[feature0, feature1],
                        H=H[feature0, feature1],
                        eps=EPS
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
                mixing_distribution = dist.Categorical(
                    probs=jnp.stack([1 - q, q], axis=-1)
                )
                component_distributions=[
                    dist.Gamma(concentration=alpha, rate=beta),
                    dist.HalfNormal(scale=L[feature0, feature1] + H[feature0, feature1])
                ]
                Mixture = dist.MixtureGeneral(
                    mixing_distribution=mixing_distribution,
                    component_distributions=component_distributions
                )

                # Observation
                numpyro.sample(
                    site.obs,
                    # dist.Gamma(concentration=alpha, rate=beta),
                    Mixture,
                    obs=response_obs
                )


class HBe(GammaModel):
    NAME = "estimation"

    def __init__(self, config: Config):
        super(HBe, self).__init__(config=config)
        self.mcmc_params = {
            # "num_warmup": 4000,
            # "num_samples": 4000,
            "num_warmup": 200,
            "num_samples": 200,
            "num_chains": 4,
            # "thinning": 4,
            "thinning": 1,
        }
        self.run_kwargs = {
            "max_tree_depth": (20, 20),
            "target_accept_prob": .95,
            "extra_fields": [
                "potential_energy",
                "num_steps",
                "accept_prob",
            ]
        }
        self.NAME += (
            f'_{self.mcmc_params["num_warmup"]}W'
            f'_{self.mcmc_params["num_samples"]}S'
            f'_{self.mcmc_params["num_chains"]}C'
            f'_{self.mcmc_params["thinning"]}T'
            f'_{self.run_kwargs["max_tree_depth"][0]}D'
            f'_{self.run_kwargs["target_accept_prob"]}A'
        )

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        n_fixed = 1
        n_delta = n_features[1] - 1

        # # Fixed
        # a_fixed_loc = numpyro.sample("a_fixed_loc", dist.Normal(3., 5.))
        # a_fixed_scale = numpyro.sample("a_fixed_scale", dist.HalfNormal(5.))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed = numpyro.sample("a_fixed", dist.Normal(3., 5.))
                    # a_fixed_raw = numpyro.sample(
                    #     "a_fixed_raw", dist.Normal(0., 1.)
                    # )
                    # a_fixed = numpyro.deterministic(
                    #     "a_fixed", a_fixed_loc + (a_fixed_scale * a_fixed_raw)
                    # )

        # Delta
        with numpyro.plate("n_delta", n_delta):
            a_delta_loc = numpyro.sample("a_delta_loc", dist.Normal(0., 5.))
            a_delta_scale = numpyro.sample("a_delta_scale", dist.HalfNormal(5.))

            with numpyro.plate(site.n_response, self.n_response):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_delta_raw = numpyro.sample("a_delta_raw", dist.Normal(0., 1.))
                    a_delta = numpyro.deterministic(
                        "a_delta", a_delta_loc + (a_delta_scale * a_delta_raw)
                    )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_delta", n_delta):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed_plus_delta = a_fixed + jnp.swapaxes(a_delta, -1, -2)

        # Hyper-priors
        b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))
        L_scale = numpyro.sample("L_scale", dist.HalfNormal(.1))
        ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(1.))
        H_scale = numpyro.sample("H_scale", dist.HalfNormal(5.))

        c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
        c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(.5))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed_plus_delta], axis=1)
                    )

                    b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                    b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                    L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                    L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

                    ell_raw = numpyro.sample("ell_raw", dist.HalfNormal(scale=1))
                    ell = numpyro.deterministic(site.ell, jnp.multiply(ell_scale, ell_raw))

                    H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
                    H = numpyro.deterministic(site.H, jnp.multiply(H_scale, H_raw))

                    c_1_raw = numpyro.sample("c_1_raw", dist.HalfNormal(scale=1))
                    c_1 = numpyro.deterministic(site.c_1, jnp.multiply(c_1_scale, c_1_raw))

                    c_2_raw = numpyro.sample("c_2_raw", dist.HalfNormal(scale=1))
                    c_2 = numpyro.deterministic(site.c_2, jnp.multiply(c_2_scale, c_2_raw))

        # # Outlier Distribution
        # q = numpyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                # Model
                mu = numpyro.deterministic(
                    site.mu,
                    S.rectified_logistic(
                        x=intensity,
                        a=a[feature0, feature1],
                        b=b[feature0, feature1],
                        L=L[feature0, feature1],
                        ell=ell[feature0, feature1],
                        H=H[feature0, feature1],
                        eps=EPS
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

                # # Mixture
                # mixing_distribution = dist.Categorical(
                #     probs=jnp.stack([1 - q, q], axis=-1)
                # )
                # component_distributions=[
                #     dist.Gamma(concentration=alpha, rate=beta),
                #     dist.HalfNormal(scale=L[feature0, feature1] + H[feature0, feature1])
                # ]
                # Mixture = dist.MixtureGeneral(
                #     mixing_distribution=mixing_distribution,
                #     component_distributions=component_distributions
                # )

                # Observation
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    # Mixture,
                    obs=response_obs
                )
