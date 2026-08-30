import pyfixest as pf
import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

main_dir = Path(__file__).parent.parent
panel_folder = main_dir / "data" / "panels"
tables_folder = main_dir / "output" / "tables"
tables_folder.mkdir(parents=True, exist_ok=True)
route_group = pd.read_parquet(panel_folder / "route_group.parquet") 

estimation_sample = route_group[route_group["group"] != "excluded"]
estimation_sample = estimation_sample[estimation_sample["confounded"] != True]
estimation_sample = estimation_sample[estimation_sample["period"] != "transition"]

def estimation_sample_function(df, segment, group):
    df = df[df["nonstop"] == segment]

    df = df[df["group"].isin(["treated", group])].copy()

    df["post"] = (df["period"] == "post")
    df["treat_post"] = (df["post"] & (df["group"] == "treated"))

    assert df["group"].nunique() == 2
    assert df["treat_post"].any()

    return df

models = {}
for segment, segment_lable in ([False, "connecting"], [True, "nonstop"]):
    for control in ["control_A", "control_B"]:
        d = estimation_sample_function(estimation_sample, segment, control)
        for y in ["log_fare", "log_pax", "HHI"]:
            models[(segment_lable, control, y)] = pf.feols(f"{y} ~ treat_post | route + yearq", data=d, vcov={"CRV1": "route"})


rows = []
for (segment, control, y), m in models.items():
    r = m.tidy().iloc[0]
    rows.append({"segment": segment, "control": control, "outcome": y,
                 "coef": r["Estimate"], "se": r["Std. Error"],
                 "ci_lo": r["2.5%"], "ci_hi": r["97.5%"], "n": m._N})


results = pd.DataFrame(rows)

is_log = results["outcome"].str.startswith("log")
results["pct"] = np.nan
results.loc[is_log, "pct"] = (np.exp(results.loc[is_log, "coef"]) - 1) * 100
results.loc[is_log, "pct_lo"] = (np.exp(results.loc[is_log, "ci_lo"]) - 1) * 100
results.loc[is_log, "pct_hi"] = (np.exp(results.loc[is_log, "ci_hi"]) - 1) * 100
results["units"] = np.where(is_log, "percent", "HHI_points")

results.to_csv(tables_folder /  "did_results.csv", index=False)

tab = results.copy()
tab["spec"] = tab["segment"] + " / " + tab["control"].str.replace("control_", "Control ")
wide = tab.pivot(index="spec", columns="outcome", values="pct") 
wide["HHI"] = tab[tab.outcome=="HHI"].set_index("spec")["coef"]
wide = wide[["log_fare", "log_pax", "HHI"]]
wide = wide.rename(columns={"log_fare": "Fare (\\%)", "log_pax": "Passengers (\\%)",
                            "HHI": "HHI (points)"})
wide = wide.reset_index().rename(columns={"spec": "Specification"})
wide.to_latex(tables_folder / f"table2.tex", float_format="%.2f", index=False)