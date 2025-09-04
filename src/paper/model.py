import logging

import numpy as np
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist

from hbmep import functional as F, smooth_functional as SF
from hbmep.model import BaseModel, NonHierarchicalBaseModel
from hbmep.util import site

from paper.util import get_subname

logger = logging.getLogger(__name__)
EPS = 1e-3


class Estimation(BaseModel):
    def __init__(self, *args, **kw):
        super(Estimation, self).__init__(*args, **kw)
        self.use_mixture = False
        self.run_id = None
        self.test_run = False

    @property
    def name(self): return get_subname(self)

    @name.setter
    def name(self, value): return value

    def circ_est_mvn_reference_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        num_fixed = 1
        num_delta = num_features[1] - 1

        a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
        a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))
        Rho_fixed = pyro.sample("Rho_fixed", dist.LKJ(self.num_response, 1.))

        with pyro.plate("num_fixed", num_fixed):
            with pyro.plate(site.num_features[0], num_features[0]):
                a_fixed_raw = pyro.sample(
                    "a_fixed_raw",
                    dist.MultivariateNormal(0, (a_fixed_scale ** 2) * Rho_fixed)
                )
                a_fixed = pyro.deterministic("a_fixed", a_fixed_loc + a_fixed_raw)

        with pyro.plate("num_delta", num_delta):
            a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 5.))
            a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(5.))
            Rho_delta = pyro.sample("Rho_delta", dist.LKJ(self.num_response, 1.))

            with pyro.plate(site.num_features[0], num_features[0]):
                a_delta_raw = pyro.sample(
                    "a_delta_raw",
                    dist.MultivariateNormal(
                        0, (a_delta_scale[:, None, None] ** 2) * Rho_delta
                    )
                )
                a_delta = pyro.deterministic("a_delta", a_delta_loc[None, :, None] + a_delta_raw)
                a_fixed_plus_delta = a_fixed + a_delta

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[1], num_features[1]):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a = pyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed_plus_delta], axis=1)
                    )

                    b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                    b = pyro.deterministic(site.b, b_scale * b_raw)

                    g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                    g = pyro.deterministic(site.g, g_scale * g_raw)

                    h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                    h = pyro.deterministic(site.h, h_scale * h_raw)

                    v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                    v = pyro.deterministic(site.v, v_scale * v_raw)

                    c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                    c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                    c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                    c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu, alpha, beta = self.gamma_likelihood(
                        SF.rectified_logistic,
                        intensity,
                        (
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            # h[*features.T],
                            EPS,
                        ),
                        c1[*features.T],
                        c2[*features.T],
                    )
                    pyro.deterministic(site.mu, mu)

                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(scale=(g[*features.T] + h[*features.T]))
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )

    def robust_circ_est_mvn_reference_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        num_fixed = 1
        num_delta = num_features[1] - 1

        a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
        a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))
        Rho_fixed = pyro.sample("Rho_fixed", dist.LKJ(self.num_response, 1.))

        df_fixed = pyro.sample("df_fixed", dist.Gamma(2, 0.1))
        chi_fixed = pyro.sample("chi_fixed", dist.Chi2(3 + df_fixed))

        with pyro.plate("num_fixed", num_fixed):
            with pyro.plate(site.num_features[0], num_features[0]):
                a_fixed_raw = pyro.sample(
                    "a_fixed_raw",
                    dist.MultivariateNormal(0, (a_fixed_scale ** 2) * Rho_fixed)
                )
                a_fixed = pyro.deterministic(
                    "a_fixed",
                    a_fixed_loc + (a_fixed_raw / jnp.sqrt(chi_fixed / (3 + df_fixed)))
                )

        with pyro.plate("num_delta", num_delta):
            a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 5.))
            a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(5.))
            Rho_delta = pyro.sample("Rho_delta", dist.LKJ(self.num_response, 1.))

            df_delta = pyro.sample("df_delta", dist.Gamma(2, 0.1))
            chi_delta = pyro.sample("chi_delta", dist.Chi2(3 + df_delta))

            with pyro.plate(site.num_features[0], num_features[0]):
                a_delta_raw = pyro.sample(
                    "a_delta_raw",
                    dist.MultivariateNormal(
                        0, (a_delta_scale[:, None, None] ** 2) * Rho_delta
                    )
                )
                a_delta = pyro.deterministic(
                    "a_delta",
                    a_delta_loc[None, :, None] 
                    + (
                        a_delta_raw
                        / jnp.sqrt(chi_delta[None, :, None] / (3 + df_delta[None, :, None]))
                    )
                )
                a_fixed_plus_delta = a_fixed + a_delta

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[1], num_features[1]):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a = pyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed_plus_delta], axis=1)
                    )

                    b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                    b = pyro.deterministic(site.b, b_scale * b_raw)

                    g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                    g = pyro.deterministic(site.g, g_scale * g_raw)

                    h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                    h = pyro.deterministic(site.h, h_scale * h_raw)

                    v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                    v = pyro.deterministic(site.v, v_scale * v_raw)

                    c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                    c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                    c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                    c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu, alpha, beta = self.gamma_likelihood(
                        SF.rectified_logistic,
                        intensity,
                        (
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            # h[*features.T],
                            EPS,
                        ),
                        c1[*features.T],
                        c2[*features.T],
                    )
                    pyro.deterministic(site.mu, mu)

                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(scale=(g[*features.T] + h[*features.T]))
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )

    def lat_est_mvn_block_reference_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        num_fixed = 1
        num_delta = num_features[1] - 1

        a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
        a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))

        with pyro.plate(site.num_features[2], num_features[2]):
            rho_block_fixed = pyro.sample("rho_block_fixed", dist.LKJ(self.num_response, 1.))

        zeros = jnp.zeros((self.num_response, self.num_response))
        rho_fixed = jnp.block([[rho_block_fixed[0], zeros], [zeros, rho_block_fixed[0]]])

        with pyro.plate("num_fixed", num_fixed):
            with pyro.plate(site.num_features[0], num_features[0]):
                a_fixed_raw = pyro.sample(
                    "a_fixed_raw",
                    dist.MultivariateNormal(0, (a_fixed_scale ** 2) * rho_fixed)
                )
                a_fixed_flat = a_fixed_loc + a_fixed_raw

        with pyro.plate("num_delta", num_delta):
            with pyro.plate(site.num_features[2], num_features[2]):
                rho_block_delta = pyro.sample("rho_block_delta", dist.LKJ(self.num_response, 1.))

        with pyro.plate("num_delta", num_delta):
            a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 5.))
            a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(5.))

            zeros = jnp.zeros((num_delta, self.num_response, self.num_response))
            rho_delta = jnp.block([[rho_block_delta[0], zeros], [zeros, rho_block_delta[0]]])

            with pyro.plate(site.num_features[0], num_features[0]):
                a_delta_raw = pyro.sample(
                    "a_delta_raw",
                    dist.MultivariateNormal(
                        0, (a_delta_scale[:, None, None] ** 2) * rho_delta
                    )
                )
                a_delta_flat = a_delta_loc[None, :, None] + a_delta_raw
                a_fixed_plus_delta_flat = a_fixed_flat + a_delta_flat

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[2], num_features[2]):
                with pyro.plate(site.num_features[1], num_features[1]):
                    with pyro.plate(site.num_features[0], num_features[0]):
                        a_flat = jnp.concatenate([a_fixed_flat, a_fixed_plus_delta_flat], axis=1) 
                        a = pyro.deterministic(
                            site.a, a_flat.reshape(*num_features, self.num_response)
                        )

                        b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                        b = pyro.deterministic(site.b, b_scale * b_raw)

                        g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                        g = pyro.deterministic(site.g, g_scale * g_raw)

                        h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                        h = pyro.deterministic(site.h, h_scale * h_raw)

                        v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                        v = pyro.deterministic(site.v, v_scale * v_raw)

                        c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                        c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                        c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                        c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu, alpha, beta = self.gamma_likelihood(
                        SF.rectified_logistic,
                        intensity,
                        (
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            # h[*features.T],
                            EPS,
                        ),
                        c1[*features.T],
                        c2[*features.T],
                    )
                    pyro.deterministic(site.mu, mu)

                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(scale=(g[*features.T] + h[*features.T]))
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )

    def robust_lat_est_mvn_block_reference_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        num_fixed = 1
        num_delta = num_features[1] - 1

        a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
        a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))

        with pyro.plate(site.num_features[2], num_features[2]):
            rho_block_fixed = pyro.sample("rho_block_fixed", dist.LKJ(self.num_response, 1.))

        zeros = jnp.zeros((self.num_response, self.num_response))
        rho_fixed = jnp.block([[rho_block_fixed[0], zeros], [zeros, rho_block_fixed[0]]])

        df_fixed = pyro.sample("df_fixed", dist.Gamma(2, 0.1))
        chi_fixed = pyro.sample("chi_fixed", dist.Chi2(3 + df_fixed))

        with pyro.plate("num_fixed", num_fixed):
            with pyro.plate(site.num_features[0], num_features[0]):
                a_fixed_raw = pyro.sample(
                    "a_fixed_raw",
                    dist.MultivariateNormal(0, (a_fixed_scale ** 2) * rho_fixed)
                )
                a_fixed_flat = a_fixed_loc + (a_fixed_raw / jnp.sqrt(chi_fixed / (3 + df_fixed)))

        with pyro.plate("num_delta", num_delta):
            with pyro.plate(site.num_features[2], num_features[2]):
                rho_block_delta = pyro.sample("rho_block_delta", dist.LKJ(self.num_response, 1.))

        with pyro.plate("num_delta", num_delta):
            a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 5.))
            a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(5.))

            zeros = jnp.zeros((num_delta, self.num_response, self.num_response))
            rho_delta = jnp.block([[rho_block_delta[0], zeros], [zeros, rho_block_delta[0]]])

            df_delta = pyro.sample("df_delta", dist.Gamma(2, 0.1))
            chi_delta = pyro.sample("chi_delta", dist.Chi2(3 + df_delta))

            with pyro.plate(site.num_features[0], num_features[0]):
                a_delta_raw = pyro.sample(
                    "a_delta_raw",
                    dist.MultivariateNormal(
                        0, (a_delta_scale[:, None, None] ** 2) * rho_delta
                    )
                )
                a_delta_flat = pyro.deterministic(
                    "a_delta_flat",
                    a_delta_loc[None, :, None] 
                    + (
                        a_delta_raw
                        / jnp.sqrt(chi_delta[None, :, None] / (3 + df_delta[None, :, None]))
                    )
                )
                a_fixed_plus_delta_flat = a_fixed_flat + a_delta_flat

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[2], num_features[2]):
                with pyro.plate(site.num_features[1], num_features[1]):
                    with pyro.plate(site.num_features[0], num_features[0]):
                        a_flat = jnp.concatenate([a_fixed_flat, a_fixed_plus_delta_flat], axis=1) 
                        a = pyro.deterministic(
                            site.a, a_flat.reshape(*num_features, self.num_response)
                        )

                        b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                        b = pyro.deterministic(site.b, b_scale * b_raw)

                        g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                        g = pyro.deterministic(site.g, g_scale * g_raw)

                        h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                        h = pyro.deterministic(site.h, h_scale * h_raw)

                        v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                        v = pyro.deterministic(site.v, v_scale * v_raw)

                        c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                        c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                        c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                        c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu, alpha, beta = self.gamma_likelihood(
                        SF.rectified_logistic,
                        intensity,
                        (
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            # h[*features.T],
                            EPS,
                        ),
                        c1[*features.T],
                        c2[*features.T],
                    )
                    pyro.deterministic(site.mu, mu)

                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(scale=(g[*features.T] + h[*features.T]))
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )

    def size_est_mvn_reference_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        num_fixed = 1
        num_delta = num_features[1] - 1

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_fixed", num_fixed):
                a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
                a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))
                rho_block_fixed = pyro.sample("rho_block_fixed", dist.LKJ(self.num_response, 1.))

                zeros = jnp.zeros((num_fixed, num_features[2], self.num_response, self.num_response))
                rho_fixed = jnp.block([[rho_block_fixed, zeros], [zeros, rho_block_fixed]])

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_fixed", num_fixed):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a_fixed_raw = pyro.sample(
                        "a_fixed_raw",
                        dist.MultivariateNormal(0, (a_fixed_scale[..., None, None] ** 2) * rho_fixed)
                    )
                    a_fixed_flat = a_fixed_loc[None, ..., None] + a_fixed_raw

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_delta", num_delta):
                a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 5.))
                a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(5.))
                rho_block_delta = pyro.sample("rho_block_delta", dist.LKJ(self.num_response, 1.))

                zeros = jnp.zeros((num_delta, num_features[2], self.num_response, self.num_response))
                rho_delta = jnp.block([[rho_block_delta, zeros], [zeros, rho_block_delta]])

                with pyro.plate(site.num_features[0], num_features[0]):
                    a_delta_raw = pyro.sample(
                        "a_delta_raw",
                        dist.MultivariateNormal(
                            0, (a_delta_scale[..., None, None] ** 2) * rho_delta
                        )
                    )
                    a_delta_flat = a_delta_loc[None, ..., None] + a_delta_raw
                    a_fixed_plus_delta_flat = a_fixed_flat + a_delta_flat

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[3], num_features[3]):
                with pyro.plate(site.num_features[2], num_features[2]):
                    with pyro.plate(site.num_features[1], num_features[1]):
                        with pyro.plate(site.num_features[0], num_features[0]):
                            a_flat = jnp.concatenate([a_fixed_flat, a_fixed_plus_delta_flat], axis=1) 
                            a = pyro.deterministic(
                                site.a, a_flat.reshape(*num_features, self.num_response)
                            )

                            b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                            b = pyro.deterministic(site.b, b_scale * b_raw)

                            g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                            g = pyro.deterministic(site.g, g_scale * g_raw)

                            h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                            h = pyro.deterministic(site.h, h_scale * h_raw)

                            v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                            v = pyro.deterministic(site.v, v_scale * v_raw)

                            c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                            c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                            c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                            c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu, alpha, beta = self.gamma_likelihood(
                        SF.rectified_logistic,
                        intensity,
                        (
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            # h[*features.T],
                            EPS,
                        ),
                        c1[*features.T],
                        c2[*features.T],
                    )
                    pyro.deterministic(site.mu, mu)

                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(scale=(g[*features.T] + h[*features.T]))
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )

    def size_est_mvn_block_reference_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        num_fixed = 1
        num_delta = num_features[1] - 1

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_fixed", num_fixed):
                with pyro.plate(site.num_features[3], num_features[3]):
                    rho_block_fixed = pyro.sample("rho_block_fixed", dist.LKJ(self.num_response, 1.))

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_fixed", num_fixed):
                a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
                a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))

                zeros = jnp.zeros((num_fixed, num_features[2], self.num_response, self.num_response))
                rho_fixed = jnp.block([[rho_block_fixed[0], zeros], [zeros, rho_block_fixed[0]]])

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_fixed", num_fixed):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a_fixed_raw = pyro.sample(
                        "a_fixed_raw",
                        dist.MultivariateNormal(0, (a_fixed_scale[..., None, None] ** 2) * rho_fixed)
                    )
                    a_fixed_flat = a_fixed_loc[None, ..., None] + a_fixed_raw

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_delta", num_delta):
                with pyro.plate(site.num_features[3], num_features[3]):
                    rho_block_delta = pyro.sample("rho_block_delta", dist.LKJ(self.num_response, 1.))

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_delta", num_delta):
                a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 5.))
                a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(5.))

                zeros = jnp.zeros((num_delta, num_features[2], self.num_response, self.num_response))
                rho_delta = jnp.block([[rho_block_delta[0], zeros], [zeros, rho_block_delta[0]]])

                with pyro.plate(site.num_features[0], num_features[0]):
                    a_delta_raw = pyro.sample(
                        "a_delta_raw",
                        dist.MultivariateNormal(
                            0, (a_delta_scale[..., None, None] ** 2) * rho_delta
                        )
                    )
                    a_delta_flat = a_delta_loc[None, ..., None] + a_delta_raw
                    a_fixed_plus_delta_flat = a_fixed_flat + a_delta_flat

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[3], num_features[3]):
                with pyro.plate(site.num_features[2], num_features[2]):
                    with pyro.plate(site.num_features[1], num_features[1]):
                        with pyro.plate(site.num_features[0], num_features[0]):
                            a_flat = jnp.concatenate([a_fixed_flat, a_fixed_plus_delta_flat], axis=1) 
                            a = pyro.deterministic(
                                site.a, a_flat.reshape(*num_features, self.num_response)
                            )

                            b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                            b = pyro.deterministic(site.b, b_scale * b_raw)

                            g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                            g = pyro.deterministic(site.g, g_scale * g_raw)

                            h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                            h = pyro.deterministic(site.h, h_scale * h_raw)

                            v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                            v = pyro.deterministic(site.v, v_scale * v_raw)

                            c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                            c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                            c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                            c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu, alpha, beta = self.gamma_likelihood(
                        SF.rectified_logistic,
                        intensity,
                        (
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            # h[*features.T],
                            EPS,
                        ),
                        c1[*features.T],
                        c2[*features.T],
                    )
                    pyro.deterministic(site.mu, mu)

                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(scale=(g[*features.T] + h[*features.T]))
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )

    def robust_size_est_mvn_block_reference_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        num_fixed = 1
        num_delta = num_features[1] - 1

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_fixed", num_fixed):
                with pyro.plate(site.num_features[3], num_features[3]):
                    rho_block_fixed = pyro.sample("rho_block_fixed", dist.LKJ(self.num_response, 1.))

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_fixed", num_fixed):
                a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(5., 5.))
                a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(5.))

                zeros = jnp.zeros((num_fixed, num_features[2], self.num_response, self.num_response))
                rho_fixed = jnp.block([[rho_block_fixed[0], zeros], [zeros, rho_block_fixed[0]]])

                df_fixed = pyro.sample("df_fixed", dist.Gamma(2, 0.1))
                chi_fixed = pyro.sample("chi_fixed", dist.Chi2(3 + df_fixed))

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_fixed", num_fixed):
                with pyro.plate(site.num_features[0], num_features[0]):
                    a_fixed_raw = pyro.sample(
                        "a_fixed_raw",
                        dist.MultivariateNormal(0, (a_fixed_scale[..., None, None] ** 2) * rho_fixed)
                    )
                    a_fixed_flat = (
                        a_fixed_loc[None, ..., None]
                        + (
                            a_fixed_raw
                            / jnp.sqrt(chi_fixed[None, ..., None] / (3 + df_fixed[None, ..., None]))
                        )
                    )

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_delta", num_delta):
                with pyro.plate(site.num_features[3], num_features[3]):
                    rho_block_delta = pyro.sample("rho_block_delta", dist.LKJ(self.num_response, 1.))

        with pyro.plate(site.num_features[2], num_features[2]):
            with pyro.plate("num_delta", num_delta):
                a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 5.))
                a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(5.))

                zeros = jnp.zeros((num_delta, num_features[2], self.num_response, self.num_response))
                rho_delta = jnp.block([[rho_block_delta[0], zeros], [zeros, rho_block_delta[0]]])

                df_delta = pyro.sample("df_delta", dist.Gamma(2, 0.1))
                chi_delta = pyro.sample("chi_delta", dist.Chi2(3 + df_delta))

                with pyro.plate(site.num_features[0], num_features[0]):
                    a_delta_raw = pyro.sample(
                        "a_delta_raw",
                        dist.MultivariateNormal(
                            0, (a_delta_scale[..., None, None] ** 2) * rho_delta
                        )
                    )
                    a_delta_flat = pyro.deterministic(
                        "a_delta_flat",
                        a_delta_loc[None, ..., None] 
                        + (
                            a_delta_raw
                            / jnp.sqrt(chi_delta[None, ..., None] / (3 + df_delta[None, ..., None]))
                        )
                    )
                    a_fixed_plus_delta_flat = a_fixed_flat + a_delta_flat

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate(site.num_features[3], num_features[3]):
                with pyro.plate(site.num_features[2], num_features[2]):
                    with pyro.plate(site.num_features[1], num_features[1]):
                        with pyro.plate(site.num_features[0], num_features[0]):
                            a_flat = jnp.concatenate([a_fixed_flat, a_fixed_plus_delta_flat], axis=1) 
                            a = pyro.deterministic(
                                site.a, a_flat.reshape(*num_features, self.num_response)
                            )

                            b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                            b = pyro.deterministic(site.b, b_scale * b_raw)

                            g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                            g = pyro.deterministic(site.g, g_scale * g_raw)

                            h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                            h = pyro.deterministic(site.h, h_scale * h_raw)

                            v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                            v = pyro.deterministic(site.v, v_scale * v_raw)

                            c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                            c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                            c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                            c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu, alpha, beta = self.gamma_likelihood(
                        SF.rectified_logistic,
                        intensity,
                        (
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            # h[*features.T],
                            EPS,
                        ),
                        c1[*features.T],
                        c2[*features.T],
                    )
                    pyro.deterministic(site.mu, mu)

                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(scale=(g[*features.T] + h[*features.T]))
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )


class HB(BaseModel):
    def __init__(self, *args, **kw):
        super(HB, self).__init__(*args, **kw)
        self.use_mixture = False
        self.run_id = None
        self.test_run = False

    @property
    def name(self): return get_subname(self)

    @name.setter
    def name(self, value): return value

    def hb_mvn_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        mask_features = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))
            mask_features = np.full((*num_features, self.num_response), False)
            mask_features[*features.T] = True

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))
        Rho = pyro.sample("Rho" ,dist.LKJ(self.num_response, 1.))

        with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-1):
            a_raw = pyro.sample(
                site.a.raw,
                dist.MultivariateNormal(0, (a_scale ** 2) * Rho)
            )
            a = pyro.deterministic(site.a, a_loc + a_raw)

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-2):
                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)

                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                h = pyro.deterministic(site.h, h_scale * h_raw)

                v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                v = pyro.deterministic(site.v, v_scale * v_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu, alpha, beta = self.gamma_likelihood(
                        SF.rectified_logistic,
                        intensity,
                        (
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            # h[*features.T],
                            EPS,
                        ),
                        c1[*features.T],
                        c2[*features.T],
                    )
                    pyro.deterministic(site.mu, mu)

                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(scale=(g[*features.T] + h[*features.T]))
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )

    def robust_hb_mvn_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        mask_features = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))
            mask_features = np.full((*num_features, self.num_response), False)
            mask_features[*features.T] = True

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))
        Rho = pyro.sample("Rho" ,dist.LKJ(self.num_response, 1.))

        df = pyro.sample("df", dist.Gamma(2, 0.1))
        chi = pyro.sample("chi", dist.Chi2(3 + df))

        with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-1):
            a_raw = pyro.sample(
                site.a.raw,
                dist.MultivariateNormal(0, (a_scale ** 2) * Rho)
            )
            a = pyro.deterministic(
                site.a,
                a_loc +  (a_raw / jnp.sqrt(chi / (3 + df)))
            )

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-2):
                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)

                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                h = pyro.deterministic(site.h, h_scale * h_raw)

                v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                v = pyro.deterministic(site.v, v_scale * v_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu, alpha, beta = self.gamma_likelihood(
                        SF.rectified_logistic,
                        intensity,
                        (
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            # h[*features.T],
                            EPS,
                        ),
                        c1[*features.T],
                        c2[*features.T],
                    )
                    pyro.deterministic(site.mu, mu)

                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(scale=(g[*features.T] + h[*features.T]))
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )

    def hb_rl_masked(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None: mask_obs = np.invert(np.isnan(response))

        a_loc = pyro.sample(site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(site.a.scale, dist.HalfNormal(5.))

        b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(site.num_features, num_features, rightmost_dim=-2):
                a_raw = pyro.sample(site.a.raw, dist.Normal(0, 1))
                a = pyro.deterministic(site.a, a_loc + a_scale * a_raw)

                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)

                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                h = pyro.deterministic(site.h, h_scale * h_raw)

                v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                v = pyro.deterministic(site.v, v_scale * v_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        if self.use_mixture: q = pyro.sample(
            site.outlier_prob, dist.Uniform(0., 0.01)
        )

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu, alpha, beta = self.gamma_likelihood(
                        SF.rectified_logistic,
                        intensity,
                        (
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            EPS,
                        ),
                        c1[*features.T],
                        c2[*features.T],
                    )
                    pyro.deterministic(site.mu, mu)

                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(scale=(g[*features.T] + h[*features.T]))
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )
