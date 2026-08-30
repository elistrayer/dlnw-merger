import pyfixest as pf
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm

main_dir = Path(__file__).parent.parent
panel_folder = main_dir / "data" / "panels"
tables_folder = main_dir / "output" / "tables"
route_group = pd.read_parquet(panel_folder / "route_group.parquet") 

def estimation_sample_function(df, segment, group):
    df = df[df["nonstop"] == segment]

    df = df[df["group"].isin(["treated", group])].copy()

    df["post"] = (df["period"] == "post")
    df["treat_post"] = (df["post"] & (df["group"] == "treated"))

    assert df["group"].nunique() == 2
    assert df["treat_post"].any()

    return df

tests = ["baseline", "confounded_allowed", "balanced_routes", "placebo_merger", "pax_weighted"]

def estimation_function(test):
    if test not in tests:
        raise ValueError(f"Unknown Test: {test}")
    
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
    observations = m._N
    return m_values, observations

rows = []
for test in tests:
    r, observations = estimation_function(test)
    rows.append({"variant": test, "coef": r["Estimate"], "se": r["Std. Error"],
                 "ci_lo": r["2.5%"], "ci_hi": r["97.5%"], "p": r["Pr(>|t|)"], "n": observations})

robustness = pd.DataFrame(rows)

Z = norm.ppf(0.975) + norm.ppf(0.80)
robustness["mde_pct"] = (np.exp(Z * robustness["se"]) - 1) * 100

robustness["coef_pct"] = (np.exp(robustness["coef"]) - 1) * 100
robustness["ci_lo_pct"] = (np.exp(robustness["ci_lo"]) - 1) * 100
robustness["ci_hi_pct"] = (np.exp(robustness["ci_hi"]) - 1) * 100

robustness.to_csv(tables_folder / "robustness_results.csv", index=False)

def format_p_value(p):
    if p < 0.001:
        return "$<$0.001"
    else:
        return f"{p:.3f}"

display = robustness[["variant", "n", "coef_pct", "ci_lo_pct", "ci_hi_pct", "p"]].copy()
display_variant_names = {"baseline": "Baseline", "confounded_allowed": "Confounded Allowed", "balanced_routes": "Balanced Routes", 
                         "placebo_merger": "Placebo Merger (2007)", "pax_weighted": "Passenger Weighted"}
display["variant"] = display["variant"].replace(display_variant_names)
display["CI (\\%)"] = display.apply(lambda row: f"[{row['ci_lo_pct']:.2f}, {row['ci_hi_pct']:.2f}]", axis=1)
display = display.drop(columns=["ci_lo_pct", "ci_hi_pct"])
display = display.rename(columns={"coef_pct": "coef (\\%)"})
display["p"] = display["p"].apply(format_p_value)
display.to_latex(tables_folder / "table3_robustness.tex", float_format="%.2f", index=False)
