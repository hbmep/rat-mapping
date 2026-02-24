import os

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from core_circ import fit, summarize_orientation
from core_circ import BUILD_DIR
from util_circ import cos, sin


def main():
    result, indicator_columns = fit()
    correction(result, indicator_terms=indicator_columns)
    return


def correction(result, indicator_terms, alpha=0.05):
    (
        model,
        result,
        rhs_terms,
        set_reference,
        formula,
        df,
    ) = result
    p = result.pvalues.copy()

    fixed_names = [name for name in p.index if name != "Group Var"]
    p_fixed = p.loc[fixed_names]

    not_correct = ["Intercept", cos, sin]
    test_names = [n for n in p_fixed.index if n not in not_correct]

    p_raw = p_fixed.loc[test_names].to_numpy()

    p_orientation = summarize_orientation(result, cos_name=cos, sin_name=sin)
    p_raw = np.concatenate([
        p_fixed.loc[test_names].to_numpy(),
        np.array([p_orientation])
    ])
    test_names = test_names + ["orientation"]

    holm_reject, holm_padj, _, _ = multipletests(p_raw, alpha=alpha, method="holm")
    df = pd.DataFrame({
        "term": test_names,
        "p_raw": p_raw,
        "p_holm": holm_padj,
        "reject_holm": holm_reject,
    }).set_index("term").sort_values("p_raw")

    print(formula)
    print(df.to_string(float_format=lambda x: f"{x:0.4g}"))

    out = os.path.join(BUILD_DIR, "multiple_circ.csv")
    df.to_csv(out, index=True)
    print(f"Saved to {out}")

    return out


if __name__ == "__main__":
    main()
