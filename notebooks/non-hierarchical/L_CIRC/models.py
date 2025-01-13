import numpy as np
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep import functional as F
from hbmep.model import GammaModel
from hbmep.model.non_hierarchical import NonHierarchicalBaseModel
from hbmep.model.utils import Site as site


class NonHierarchicalBayesianModel(NonHierarchicalBaseModel, GammaModel):
    NAME = "non_hierarchical_bayesian_model"

    def __init__(self, config: Config):
        super(NonHierarchicalBayesianModel, self).__init__(config=config)
        self.n_jobs = -1

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.sample(
                        site.a, dist.TruncatedNormal(150., 100., low=0)
                    )
                    b = numpyro.sample(site.b, dist.HalfNormal(scale=1.))

                    L = numpyro.sample(site.L, dist.HalfNormal(scale=.1))
                    ell = numpyro.sample(site.ell, dist.HalfNormal(scale=1.))
                    H = numpyro.sample(site.H, dist.HalfNormal(scale=5.))

                    c_1 = numpyro.sample(site.c_1, dist.HalfNormal(scale=5.))
                    c_2 = numpyro.sample(site.c_2, dist.HalfNormal(scale=.5))

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

                # Observation
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )
