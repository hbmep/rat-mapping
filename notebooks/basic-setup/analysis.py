import os
import pickle

from hbmep.model.utils import Site as site


from constants import (
    BUILD_DIR,
    INFERENCE_FILE
)



if __name__ == "__main__":
    src = "/Users/andres/repos/rat-mapping-paper/reports/basic-setup/rectified_logistic/inference.pkl"
    with open(src, "rb") as g:
        model, mcmc, posterior_samples = pickle.load(g)


    print(type(posterior_samples))
    print(posterior_samples.keys())
    
    a = posterior_samples[site.a]

    print(a.shape)

    print(a.mean(axis=0))

    print(a.std(axis=0))

    #manipulate posterior samples and understand what they are
    #use PS to get the overlap between patients -- use 
    print(a.median(axis=0)
    
          )