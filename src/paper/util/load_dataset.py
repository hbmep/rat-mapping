import os

import numpy as np
import pandas as pd

from paper.constants import DATA, REPO, REPORTS
from paper.constants import (
    circ as circ_constants,
    shie as shie_constants,
    smalar as smalar_constants,
    rcml as rcml_constants
)

SEPARATOR = "___"
COMBINATION_CDF = "combination_cdf"


def get_paths(experiment):
    build_dir = os.path.join(REPORTS, experiment.lower()[2:].replace('_', ""))
    toml_path = os.path.join(REPO, "configs", f"{experiment}.toml")
    data_path = os.path.join(DATA, f"{experiment}.csv")
    mep_matrix_path = os.path.join(DATA, f"{experiment}.mat")
    return build_dir, toml_path, data_path, mep_matrix_path


def load_circ(
    *,
    features,
    run_id,
    set_reference=False,
    add_zero=True,
    **kw
):
    _, _, DATA_PATH, _ = get_paths(circ_constants.EXPERIMENT)
    MAP = circ_constants.MAP
    DIAM = circ_constants.DIAM
    VERTICES = circ_constants.VERTICES
    RADII = circ_constants.RADII
    
    # Load data
    src = DATA_PATH
    data = pd.read_csv(src)
    df = data.copy()

    if add_zero:
        feat = ["compound_position"]
        pid = "participant"
        y = "pulse_amplitude"
        idx0 = df[y].eq(0)
        zero_df = df.loc[idx0].copy()
        rest_df = df.loc[~idx0].copy()
        assert set(df[pid].unique()) <= set(zero_df[pid].unique())
        ncond = zero_df.groupby(pid)[feat].apply(lambda x: x.apply(tuple, axis=1).nunique())
        assert (ncond == 1).all(), ncond[ncond != 1]
        combos = df[[pid] + feat].drop_duplicates()
        zero_broadcast = (
            zero_df.drop(columns=feat)
            .merge(combos, on=pid, how="inner")
        )
        df2 = pd.concat([rest_df, zero_broadcast], ignore_index=True)
        df = df2.copy()

    cats = df[features[1]].unique().tolist()
    mapping = {}
    for cat in cats:
        assert cat not in mapping
        l, r = cat.split("-")
        mapping[cat] = l[3:] + "-" + r[3:]
    assert mapping == MAP
    df[features[1]] = df[features[1]].replace(mapping)
    cats = set(df[features[1]].tolist())
    for t_, t_len in zip([DIAM, RADII, VERTICES], [4, 8, 9]):
        assert set(t_) <= cats
        assert len(set(t_)) == len(t_)
        assert len(t_) == t_len

    assert run_id in {"diam", "radii", "vertices", "all"}
    match run_id:
        case "diam":
            subset = DIAM
        case "radii":
            subset = RADII
        case "vertices":
            subset = VERTICES
        case "all":
            subset = DIAM + RADII + VERTICES

    if set_reference:
        match run_id:
            case "diam" | "radii":
                reference = "-C"
                subset += [reference]
            case "vertices":
                reference = "S-N"
                subset += [reference]
            case "all":
                reference = "-C"

    assert len(set(subset)) == len(subset)
    assert set(subset) <= set(df[features[1]].values.tolist())
    idx = df[features[1]].isin(subset)
    df = df.loc[idx].reset_index(drop=True).copy()
    if set_reference:
        df[features[1]] = (
            df[features[1]]
            .replace({reference: " " + reference})
        )
    return df
    

def load_shie(
    *,
    features,
    run_id,
    set_reference=False,
    add_zero=True,
    **kw
):
    _, _, DATA_PATH, _ = get_paths(shie_constants.EXPERIMENT)
    POSITIONS_MAP = shie_constants.POSITIONS_MAP
    CHARGES_MAP = shie_constants.CHARGES_MAP
    WITH_GROUND = shie_constants.WITH_GROUND
    NO_GROUND = shie_constants.NO_GROUND

    # Load data
    src = DATA_PATH
    data = pd.read_csv(src)
    df = data.copy()

    if add_zero:
        feat = ["compound_position", "compound_charge_params"]
        pid = "participant"
        y = "pulse_amplitude"
        idx0 = df[y].eq(0)
        zero_df = df.loc[idx0].copy()
        rest_df = df.loc[~idx0].copy()
        assert set(df[pid].unique()) <= set(zero_df[pid].unique())
        nfeat = zero_df.groupby(pid)[feat].apply(lambda x: x.drop_duplicates().shape[0])
        assert (nfeat == 1).all(), nfeat[nfeat != 1]
        combos = df[[pid] + feat].drop_duplicates()
        zero_broadcast = (
            zero_df.drop(columns=feat)
            .merge(combos, on=pid, how="inner")
        )
        df2 = pd.concat([rest_df, zero_broadcast], ignore_index=True)
        df = df2.copy()

    cats = df[features[1]].unique().tolist()
    mapping = {}
    for cat in cats:
        assert cat not in mapping
        l, r = cat.split("-")
        mapping[cat] = l[3:] + "-" + r[3:]
    assert mapping == POSITIONS_MAP
    df[features[1]] = df[features[1]].replace(mapping)

    cats = df[features[2]].unique().tolist()
    assert set(cats) == set(CHARGES_MAP.keys())
    df[features[2]] = df[features[2]].replace(CHARGES_MAP)
   
    combos = df[features[1:]].apply(tuple, axis=1).unique().tolist()
    assert sorted(combos) == sorted(WITH_GROUND + NO_GROUND)

    assert run_id in {"ground", "no-ground", "all"}
    match run_id:
        case "ground":
            subset = WITH_GROUND
        case "no-ground":
            subset = NO_GROUND
        case "all":
            subset = WITH_GROUND + NO_GROUND
 
    if set_reference:
        reference = ('-C', 'Biphasic')
        match run_id:
            case "no-ground":
                subset += [reference]
            case "ground" | "all":
                ...

    assert set(subset) <= set(
        df[features[1:]].apply(tuple, axis=1).values.tolist()
    )
    idx = df[features[1:]].apply(tuple, axis=1).isin(subset)
    df = df.loc[idx].reset_index(drop=True).copy()

    if set_reference:
        idx = df[features[1:]].apply(tuple, axis=1).isin([reference])
        assert df.loc[idx, features[1]].nunique() == 1
        assert df.loc[idx, features[1]].unique()[0] == "-C"
        df.loc[idx, features[1]] = " -C"

    return df


def load_csmalar_data(data: pd.DataFrame, mat=None):
    data = data.copy()
    # make sure columns channel1_segment and channel2_segment are correct
    ch1 = data.compound_position.apply(
        lambda x: np.nan if not x.split("-")[0] else x.split("-")[0][:2]
    )
    pd.testing.assert_series_equal(
        ch1, data.channel1_segment, check_names=False
    )
    ch2 = data.compound_position.apply(
        lambda x: np.nan if not x.split("-")[1] else x.split("-")[1][:2]
    )
    pd.testing.assert_series_equal(
        ch2, data.channel2_segment, check_names=False
    )
    # make sure channel2_segment is never nan
    assert not data.channel2_segment.isna().any()
    # make sure columns channel1_designation and channel2_designation are correct
    ch1_lat = data.compound_position.apply(
        lambda x: np.nan if not x.split("-")[0] else x.split("-")[0][2:]
    )
    pd.testing.assert_series_equal(
        ch1_lat, data.channel1_designation, check_names=False
    )
    ch2_lat = data.compound_position.apply(
        lambda x: np.nan if not x.split("-")[1] else x.split("-")[1][2:]
    )
    pd.testing.assert_series_equal(
        ch2_lat, data.channel2_designation, check_names=False       
    )
    # make sure channel2_designation is never nan
    assert not data.channel2_designation.isna().any()

    # create the relevant feature columns
    data["segment"] = np.where(
        data.channel1_segment.isna(),
        "-" + data.channel2_segment,
        data.channel1_segment + "-" + data.channel2_segment
    )
    assert not data["segment"].isna().any()
    data["lat"] = np.where(
        data.channel1_designation.isna(),
        "-" + data.channel2_designation,
        data.channel1_designation + "-" + data.channel2_designation
    )
    assert not data["lat"].isna().any()
    # make sure the new feature columns are correct
    temp = (
        data[["segment", "lat"]]
        .apply(
            lambda x: (
                x[0].split("-")[0] + x[1].split("-")[0]
                + "-" + x[0].split("-")[1] + x[1].split("-")[1]
            ),
            axis=1
        )
    )
    pd.testing.assert_series_equal(
        temp, data.compound_position, check_names=False
    )

    df = data.copy()
    # Remove contacts with size B-S and S-B
    remove_size = ["B-S", "S-B"]
    idx = df.compound_size.isin(remove_size)
    df = df[~idx].reset_index(drop=True).copy()
    if mat is not None:
        mat = mat[~idx]

    flipped = df.lat.unique()
    flipped = sorted([tuple(sorted(u.split("-"))) for u in flipped])
    flipped_set = sorted(set(flipped))
    print(len(flipped), len(flipped_set))
    from collections import Counter
    flipped_counts = Counter(flipped)
    print(f"Duplicated: {[u for u, v in flipped_counts.items() if v > 1]}")

    [print("clear") for _ in range(10)]
    idx = df.lat == "LM-M"
    print(df[idx].participant.unique())
    idx = df.lat == "M-LM"
    print(df[idx].participant.unique())

    [print("clear") for _ in range(10)]
    idx = df.lat == "L-LL"
    print(df[idx].participant.unique())
    idx = df.lat == "LL-L"
    print(df[idx].participant.unique())

    [print("clear") for _ in range(10)]
    idx = df.lat == "LM2-M"
    print(df[idx].participant.unique())
    idx = df.lat == "M-LM2"
    print(df[idx].participant.unique())

    flipped_combinations = [
        ("amap01", "LL-L"),
        ("amap02", "LL-L"),
        ("amap01", "LM-M"),
        ("amap02", "LM-M"),
    ]
    flipped_features = df[["participant", "lat"]].apply(tuple, axis=1)
    flipped_idx = flipped_features.isin(flipped_combinations)
    df = df[~flipped_idx].reset_index(drop=True).copy()
    if mat is not None:
        mat = mat[~flipped_idx]
    df.lat = df.lat.replace({"LM2-M": "M-LM2"})

    [print("clear") for _ in range(10)]
    idx = df.lat == "LM-M"
    print(df[idx].participant.unique())
    idx = df.lat == "M-LM"
    print(df[idx].participant.unique())

    [print("clear") for _ in range(10)]
    idx = df.lat == "L-LL"
    print(df[idx].participant.unique())
    idx = df.lat == "LL-L"
    print(df[idx].participant.unique())

    [print("clear") for _ in range(10)]
    idx = df.lat == "LM2-M"
    print(df[idx].participant.unique())
    idx = df.lat == "M-LM2"
    print(df[idx].participant.unique())

    # Remove contacts with designation RM, R, RR
    remove_designation = ["RM", "R", "RR"]
    idx = df.channel1_designation.isin(remove_designation)
    df = df[~idx].reset_index(drop=True).copy()
    if mat is not None:
        mat = mat[~idx]
    idx = df.channel2_designation.isin(remove_designation)
    df = df[~idx].reset_index(drop=True).copy()
    if mat is not None:
        mat = mat[~idx]
    # Remove C7 segment
    remove_segments = ["C7"]
    idx = df.channel1_segment.isin(remove_segments)
    df = df[~idx].reset_index(drop=True).copy()
    if mat is not None:
        mat = mat[~idx]
    idx = df.channel2_segment.isin(remove_segments)
    df = df[~idx].reset_index(drop=True).copy()
    if mat is not None:
        mat = mat[~idx]
    # Remove bipolar contacts that connect between two different segments.
    # these were recorded by mistake during experiments and won't be analyzed
    idx = (
        (df.channel1_segment == df.channel2_segment)
        | df.channel1_segment.isna()
    )
    df = df[idx].reset_index(drop=True).copy()
    if mat is not None:
        mat = mat[idx]
    assert (
        (df.channel1_segment == df.channel2_segment)
        | df.channel1_segment.isna()
    ).all()
    if mat is not None:
        return df, mat
    return df


def load_lat(
    *,
    intensity,
    features,
    run_id,
    set_reference=False,
    **kw
):
    _, _, DATA_PATH, _ = get_paths(smalar_constants.EXPERIMENT)
    GROUND_BIG = smalar_constants.GROUND_BIG
    GROUND_SMALL = smalar_constants.GROUND_SMALL
    NO_GROUND_BIG = smalar_constants.NO_GROUND_BIG
    NO_GROUND_SMALL = smalar_constants.NO_GROUND_SMALL
    INBETWEEN_BIG = smalar_constants.INBETWEEN_BIG
    INBETWEEN_SMALL = smalar_constants.INBETWEEN_SMALL

    # Load data
    src = DATA_PATH
    if "between" in run_id:
        src = src.replace(".csv", "_inbetween.csv")
    data = pd.read_csv(src)
    df = log_transform_intensity(data, intensity)
    
    assert run_id in {
        "lat-small-ground", "lat-big-ground", "lat-small-no-ground",
        "lat-big-no-ground", "lat-small-inbetween", "lat-big-inbetween"
    }
    subset = []
    match run_id:
        case "lat-small-ground": subset = GROUND_SMALL
        case "lat-big-ground": subset = GROUND_BIG
        case "lat-small-no-ground": subset = NO_GROUND_SMALL
        case "lat-big-no-ground": subset = NO_GROUND_BIG
        case "lat-small-inbetween": subset = GROUND_SMALL + INBETWEEN_SMALL
        case "lat-big-inbetween": subset = GROUND_BIG + INBETWEEN_BIG
        case _: raise ValueError

    if set_reference:
        reference = "-M"
        match run_id:
            case "lat-small-ground": pass
            case "lat-big-ground": pass
            case "lat-small-no-ground": subset += [
                ('-M', '-C5', 'S'), ('-M', '-C6', 'S')
            ]
            case "lat-big-no-ground": subset += [
                ('-M', '-C5', 'B'), ('-M', '-C6', 'B')
            ]
            case _: raise ValueError

    assert len(set(subset)) == len(subset)
    cols = ["lat", "segment", "compound_size"]
    if "between" not in run_id:
        assert set(subset) <= set(df[cols].apply(tuple, axis=1).tolist())
    idx = df[cols].apply(tuple, axis=1).isin(subset)
    df = df[idx].reset_index(drop=True).copy()
    df[features[-1]] = df[features[-1]].replace(
        {
            "-LM1": "-LM", "M-LM1": "M-LM",
            "LM1-L": "LM-L",
            "LM1-LL": "LM-LL",
            "LM2-LM1": "LM2-LM",
            "M-LM1": "M-LM"
        }
    )

    if "between" in run_id:
        df.segment = df.segment.apply(lambda x: "-" + x.split("-")[1])

    if set_reference:
        if "no-ground" in run_id:
            df["segment"] = df["segment"].apply(
                lambda x: f"-{x.split('-')[-1]}"
            )
        df["lat"] = df["lat"].replace({reference: " " + reference})

    # if set_reference:
    #     t = df.groupby(cols, as_index=False)[features[0]].agg(
    #         lambda x: x.nunique()
    #     )
    #     keys = t[cols].apply(tuple, axis=1); values = t[features[0]]
    #     key, values = zip(*sorted(zip(keys, values)))
    #     print(list(zip(keys, values)))
    return df


def load_size(
    *,
    intensity,
    features,
    run_id,
    set_reference=False,
    **kw
):
    _, _, DATA_PATH, _ = get_paths(smalar_constants.EXPERIMENT)
    NO_GROUND = smalar_constants.NO_GROUND
    GROUND = smalar_constants.GROUND
    GROUND_BIG = smalar_constants.GROUND_BIG
    GROUND_SMALL = smalar_constants.GROUND_SMALL
    NO_GROUND_BIG = smalar_constants.NO_GROUND_BIG
    NO_GROUND_SMALL = smalar_constants.NO_GROUND_SMALL

    # Load data
    src = DATA_PATH
    data = pd.read_csv(src)
    df = log_transform_intensity(data, intensity)
    
    assert run_id in {"size-ground", "size-no-ground", "all"}
    if run_id == "all": assert not set_reference
    subset = []
    match run_id:
        case "size-ground": subset = GROUND
        case "size-no-ground": subset = NO_GROUND
        case "all": subset = (
            GROUND
            + NO_GROUND
            + GROUND_BIG
            + GROUND_SMALL
            + NO_GROUND_BIG
            + NO_GROUND_SMALL
        ); subset = list(set(subset))
        case _: raise ValueError
    assert len(set(subset)) == len(subset)
    cols = ["lat", "segment", "compound_size"]
    assert set(subset) <= set(df[cols].apply(tuple, axis=1).tolist())
    idx = df[cols].apply(tuple, axis=1).isin(subset)
    df = df[idx].reset_index(drop=True).copy()
    df[features[-2]] = df[features[-2]].replace(
        {"-LM1": "-LM", "M-LM1": "M-LM"}
    )

    if set_reference:
        df["compound_size"] = df["compound_size"].replace({"S": " S"})

    # if set_reference:
    #     t = df.groupby(cols, as_index=False)[features[0]].agg(
    #         lambda x: x.nunique()
    #     )
    #     keys = t[cols].apply(tuple, axis=1); values = t[features[0]]
    #     key, values = zip(*sorted(zip(keys, values)))
    #     print(list(zip(keys, values)))
    return df


def load_rcml_data(data: pd.DataFrame):
    data = data.copy()
    ch1 = data.compound_position.apply(
        lambda x: np.nan if not x.split("-")[0] else x.split("-")[0][:2]
    )
    pd.testing.assert_series_equal(
        ch1, data.channel1_segment, check_names=False
    )
    ch2 = data.compound_position.apply(
        lambda x: np.nan if not x.split("-")[1] else x.split("-")[1][:2]
    )
    pd.testing.assert_series_equal(
        ch2, data.channel2_segment, check_names=False
    )
    # make sure channel2_segment is never nan
    assert not data.channel2_segment.isna().any()
    # make sure columns channel1_designation and channel2_designation are correct
    ch1_lat = data.compound_position.apply(
        lambda x: np.nan if not x.split("-")[0] else x.split("-")[0][2:]
    )
    pd.testing.assert_series_equal(
        ch1_lat, data.channel1_laterality, check_names=False
    )
    ch2_lat = data.compound_position.apply(
        lambda x: np.nan if not x.split("-")[1] else x.split("-")[1][2:]
    )
    pd.testing.assert_series_equal(
        ch2_lat, data.channel2_laterality, check_names=False
    )
    # make sure channel2_designation is never nan
    assert not data.channel2_laterality.isna().any()

    # create the relevant feature columns
    data["segment"] = np.where(
        data.channel1_segment.isna(),
        "-" + data.channel2_segment,
        data.channel1_segment + "-" + data.channel2_segment
    )
    assert not data["segment"].isna().any()
    data["lat"] = np.where(
        data.channel1_laterality.isna(),
        "-" + data.channel2_laterality,
        data.channel1_laterality + "-" + data.channel2_laterality
    )
    assert not data["lat"].isna().any()
    # make sure the new feature columns are correct
    temp = (
        data[["segment", "lat"]]
        .apply(
            lambda x: (
                x[0].split("-")[0] + x[1].split("-")[0] + "-" +
                x[0].split("-")[1] + x[1].split("-")[1]
            ),
            axis=1
        )
    )
    pd.testing.assert_series_equal(
        temp, data.compound_position, check_names=False
    )
    return data


def load_rcml(
    *,
    intensity,
    features,
    run_id,
    set_reference=False,
    **kw
):
    GROUND = rcml_constants.GROUND
    ROSTRAL_CAUDAL = rcml_constants.ROSTRAL_CAUDAL
    MIDLINE_LATERAL = rcml_constants.MIDLINE_LATERAL
    ROOT_ALIGNED = rcml_constants.ROOT_ALIGNED
    _, _, DATA_PATH, _ = get_paths(rcml_constants.EXPERIMENT)

    # Load data
    src = DATA_PATH
    data = pd.read_csv(src)
    data = load_rcml_data(data)
    df = log_transform_intensity(data, intensity)
    assert run_id in {"all", "ground", "rostral-caudal"}

    subset = []
    match run_id:
        case "ground": subset = GROUND
        case "rostral-caudal": subset = ROSTRAL_CAUDAL
        case "all": subset = (
            GROUND + ROSTRAL_CAUDAL + MIDLINE_LATERAL + ROOT_ALIGNED
        )
        case _: raise ValueError
    assert len(set(subset)) == len(subset)
    cols = ["segment", "lat"]
    assert set(subset) <= set(df[cols].apply(tuple, axis=1).tolist())
    idx = df[cols].apply(tuple, axis=1).isin(subset)
    df = df[idx].reset_index(drop=True).copy()
    return df
