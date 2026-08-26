import pyfixest as pf
import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg
import matplotlib.pyplot as plt

main_dir = Path(__file__).parent.parent
panel_folder = main_dir / "data" / "panels"
tables_folder = main_dir / "output" / "tables"
route_group = pd.read_parquet(panel_folder / "route_group.parquet") 
figures_folder = main_dir / "output" / "figures"
figures_folder.mkdir(parents=True, exist_ok=True)

def estimation_sample_function(df, segment, group):
    df = df[df["nonstop"] == segment]

    df = df[df["group"].isin(["treated", group])].copy()

    df["post"] = (df["period"] == "post")
    df["treat_post"] = (df["post"] & (df["group"] == "treated"))

    assert df["group"].nunique() == 2
    assert df["treat_post"].any()

    return df

def estimation_function(test):
    estimation_sample = route_group[route_group["group"] != "excluded"].copy()
    estimation_sample = estimation_sample[estimation_sample["period"] != "transition"]

    if test == "balanced_routes":
        estimation_sample["qpr"] = estimation_sample.groupby("route", observed=True)["t"].transform("nunique")
        estimation_sample = estimation_sample[estimation_sample["qpr"] == estimation_sample["t"].nunique()]

    if test != "confounded_allowed":
        estimation_sample = estimation_sample[estimation_sample["confounded"] != True]

    if test == "placebo_merger":
        estimation_sample = estimation_sample[estimation_sample["t"] <= 8]
        estimation_sample["period"] = np.where(estimation_sample["t"] > 4, "post", "pre")

    estimation_sample = estimation_sample_function(estimation_sample, False, "control_A")
    estimation_sample["treated"] = estimation_sample["group"] == "treated"
    estimation_sample["treated"] = estimation_sample["treated"].astype(int)

    if test == "pax_weighted":
        m = pf.feols("log_fare ~ treat_post | route + yearq", data=estimation_sample, weights="pax", vcov={"CRV1": "route"})
    else:
        m = pf.feols("log_fare ~ treat_post | route + yearq", data=estimation_sample, vcov={"CRV1": "route"})
    
    m_values = m.tidy().iloc[0]
    return m_values

baseline = estimation_function("baseline")
confounded_allowed = estimation_function("confounded_allowed")
balanced_routes = estimation_function("balanced_routes")
placebo_merger = estimation_function("placebo_merger")
pax_weighted = estimation_function("pax_weighted")

robustness_checks = {"Baseline": baseline, "Confounded Allowed": confounded_allowed, "Balanced Routes": balanced_routes,
                     "Placebo Merger": placebo_merger, "Pax Weighted": pax_weighted}

robustness_checks = pd.DataFrame(robustness_checks)
print(robustness_checks)