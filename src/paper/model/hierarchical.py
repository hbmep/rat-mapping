import hbmep as mep
import numpy as np
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist

from paper.util.util import get_subname

EPS = 1e-3


class HB(mep.BaseModel):
    def __init__(self, *args, **kw):
        super(HB, self).__init__(*args, **kw)
        self.use_mixture = True
        self.test_run = True
        self.run_id = None

    @property
    def name(self): return get_subname(self)

    @name.setter
    def name(self, value): return value

    def log2_hb_mvn(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        b_scale = pyro.sample(mep.site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(mep.site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(mep.site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(mep.site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(mep.site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(mep.site.c2.scale, dist.HalfNormal(.5))

        a_loc = pyro.sample(mep.site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(mep.site.a.scale, dist.HalfNormal(5.))
        Rho = pyro.sample(mep.site.Rho, dist.LKJ(self.num_response, 1.))

        with pyro.plate_stack(mep.site.num_features, num_features, rightmost_dim=-1):
            a_raw = pyro.sample(
                mep.site.a.raw,
                dist.MultivariateNormal(0, (a_scale ** 2) * Rho)
            )
            a = pyro.deterministic(mep.site.a, a_loc + a_raw)

        with pyro.plate(mep.site.num_response, self.num_response):
            with pyro.plate_stack(mep.site.num_features, num_features, rightmost_dim=-2):
                b_raw = pyro.sample(mep.site.b.raw, dist.HalfNormal(1))
                b = b_scale * b_raw

                g_raw = pyro.sample(mep.site.g.raw, dist.HalfNormal(1))
                g = g_scale * g_raw

                h_raw = pyro.sample(mep.site.h.raw, dist.HalfNormal(1))
                h = h_scale * h_raw

                v_raw = pyro.sample(mep.site.v.raw, dist.HalfNormal(1))
                v = v_scale * v_raw

                c1_raw = pyro.sample(mep.site.c1.raw, dist.HalfNormal(1))
                c1 = c1_scale * c1_raw

                c2_raw = pyro.sample(mep.site.c2.raw, dist.HalfNormal(1))
                c2 = c2_scale * c2_raw

        if self.use_mixture:
            q = pyro.sample(mep.site.outlier_prob, dist.Uniform(0., 0.01))

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(mep.site.num_response, self.num_response):
                with pyro.plate(mep.site.num_data, num_data):
                    mu = mep.functional.rectified_logistic(
                        intensity,
                        a[*features.T],
                        b[*features.T],
                        g[*features.T],
                        h[*features.T],
                        v[*features.T],
                        EPS
                    )
                    alpha, beta = self.gamma_likelihood(
                        mu, c1[*features.T], c2[*features.T],
                    )
                    pyro.deterministic(mep.site.mu, mu)

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
                        mep.site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )

    def log2_hb_mvn_gfix(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        b_scale = pyro.sample(mep.site.b.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(mep.site.g.scale, dist.HalfNormal(.1))
        h_scale = pyro.sample(mep.site.h.scale, dist.HalfNormal(5.))
        v_scale = pyro.sample(mep.site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(mep.site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(mep.site.c2.scale, dist.HalfNormal(.5))

        a_loc = pyro.sample(mep.site.a.loc, dist.Normal(5., 5.))
        a_scale = pyro.sample(mep.site.a.scale, dist.HalfNormal(5.))
        Rho = pyro.sample(mep.site.Rho, dist.LKJ(self.num_response, 1.))

        with pyro.plate_stack(mep.site.num_features[1], num_features[1:], rightmost_dim=-1):
            with pyro.plate(mep.site.num_features[0], num_features[0]):
                a_raw = pyro.sample(
                    mep.site.a.raw,
                    dist.MultivariateNormal(0, (a_scale ** 2) * Rho)
                )
                a = pyro.deterministic(mep.site.a, a_loc + a_raw)

        with pyro.plate(mep.site.num_response, self.num_response):
            with pyro.plate(mep.site.num_features[0], num_features[0]):
                g_raw = pyro.sample(mep.site.g.raw, dist.HalfNormal(1))
                g = g_scale * g_raw     # (P, M)
                g = g[:, None]

        with pyro.plate(mep.site.num_response, self.num_response):
            with pyro.plate_stack(mep.site.num_features[1], num_features[1:], rightmost_dim=-2):
                with pyro.plate(mep.site.num_features[0], num_features[0]):
                    b_raw = pyro.sample(mep.site.b.raw, dist.HalfNormal(1))
                    b = b_scale * b_raw

                    h_raw = pyro.sample(mep.site.h.raw, dist.HalfNormal(1))
                    h = h_scale * h_raw

                    v_raw = pyro.sample(mep.site.v.raw, dist.HalfNormal(1))
                    v = v_scale * v_raw

                    c1_raw = pyro.sample(mep.site.c1.raw, dist.HalfNormal(1))
                    c1 = c1_scale * c1_raw

                    c2_raw = pyro.sample(mep.site.c2.raw, dist.HalfNormal(1))
                    c2 = c2_scale * c2_raw

        if self.use_mixture:
            q = pyro.sample(mep.site.outlier_prob, dist.Uniform(0., 0.01))

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(mep.site.num_response, self.num_response):
                with pyro.plate(mep.site.num_data, num_data):
                    mu = mep.functional.rectified_logistic(
                        intensity,
                        a[*features.T],
                        b[*features.T],
                        g[*features.T],
                        h[*features.T],
                        v[*features.T],
                        EPS
                    )
                    alpha, beta = self.gamma_likelihood(
                        mu, c1[*features.T], c2[*features.T],
                    )
                    pyro.deterministic(mep.site.mu, mu)

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
                        mep.site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )
