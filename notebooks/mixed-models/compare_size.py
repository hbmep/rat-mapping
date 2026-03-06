import math

from core_size import flatten, fit_model
from util_circ import (
    is_bip as IS_BIPOLAR,
    bip_dm as BIP_DISTANCE,
    inv_dm as INVERSE_DISTANCE,
)


def compare_models(show_summaries: bool = True) -> None:
    flat, base_locs = flatten()
    indicator_columns = base_locs
    set_reference = "M"

    m1 = fit_model(
        flat,
        indicator_columns=indicator_columns,
        set_reference=set_reference,
        with_reml=False,
        inverse_distance=True,
    )

    m3 = fit_model(
        flat,
        indicator_columns=indicator_columns,
        set_reference=set_reference,
        with_reml=False,
        inverse_distance=False,
    )

    (_, res1, rhs1, _, formula1, _) = m1
    (_, res3, rhs3, _, formula3, _) = m3

    print("\n=== Model 1 ===")
    print(formula1)
    if show_summaries:
        print(res1.summary())

    print("\n=== Model 3 ===")
    print(formula3)
    if show_summaries:
        print(res3.summary())

    ll1 = res1.llf
    ll3 = res3.llf

    k1 = res1.df_modelwc
    k3 = res3.df_modelwc

    aic1 = -2 * ll1 + 2 * k1
    aic3 = -2 * ll3 + 2 * k3

    bic1 = -2 * ll1 + k1 * math.log(res1.model.nobs)
    bic3 = -2 * ll3 + k3 * math.log(res3.model.nobs)

    print("\n=== Fit comparison (ML) ===")
    print(f"Model 1: LL={ll1:.6f}  AIC={aic1:.6f}  BIC={bic1:.6f}  k={k1}")
    print(f"Model 3: LL={ll3:.6f}  AIC={aic3:.6f}  BIC={bic3:.6f}  k={k3}")
    print(f"Delta:   LL={ll3-ll1:+.6f}  AIC={aic3-aic1:+.6f}  BIC={bic3-bic1:+.6f}")

    lr = 2 * (ll3 - ll1)
    df = k3 - k1

    from scipy.stats import chi2

    p = chi2.sf(lr, df)

    print("\n=== LR-style comparison ===")
    print(f"LR = {lr:.6f}, df = {df}, p = {p:.6g}")


def main() -> None:
    compare_models(show_summaries=True)


if __name__ == "__main__":
    main()
