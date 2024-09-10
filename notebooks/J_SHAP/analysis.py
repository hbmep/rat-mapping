import os

import pickle
import pandas as pd

from hbmep.model.utils import Site as site

from constants import(
    DATA_PATH
)

if __name__ == "__main__":
    src = DATA_PATH 
    d = {}
    df = pd.read_csv(src)
    combo = df[["participant", "compound_position", "compound_charge_params"]].apply(tuple, axis=1).unique().tolist()
    muscle = ["LADM", "LBiceps", "LDeltoid", "LECR", "LFCR", "LTriceps"]
    combo = [(c[0], c[1], c[2], m,) for c in combo for m in muscle]

    for par, pos,comp, musc  in combo:
        print(par, pos, comp, musc)
        src= f"/home/andres/repos/rat-mapping-paper/reports/J_SHAP/sub__{par}/pos__{pos}/size__{comp}/resp__{musc}/non_hierarchical_bayesian_model/inference.pkl"
        with open(src, "rb") as f:
            _,_,posterior_samples = pickle.load(f)
        d[(par, pos, comp, musc)] = posterior_samples 
    
    path = "/home/andres/repos/rat-mapping-paper/reports/J_SHAP/combine.pkl"
    with open(path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)