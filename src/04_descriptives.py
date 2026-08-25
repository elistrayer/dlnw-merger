import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg
import matplotlib.pyplot as plt

main_dir = Path(__file__).parent.parent
panel_folder = main_dir / "data" / "panels"
tables_folder = main_dir / "output" / "tables"
tables_folder.mkdir(parents=True, exist_ok=True)
figures_folder = main_dir / "output" / "figures"
figures_folder.mkdir(parents=True, exist_ok=True)

route_group = pd.read_parquet(panel_folder / "route_group.parquet")
estimation_sample = route_group[route_group["group"] != "excluded"]
estimation_sample = estimation_sample[estimation_sample["confounded"] != True]

def make_descriptives(sample, is_nonstop, label):
    seg = sample[sample["nonstop"] == is_nonstop]
    summary_statistics = seg[seg["period"] == "pre"]

    summary_statistics = summary_statistics.groupby(["route", "group"], observed=True).agg(
        pax_total=("pax", "sum"), 
        dist=("dist", "mean"), 
        fx=("fx", "sum"), HHI=("HHI", "mean"), 
        pax_per_qtr=("pax", "mean"), 
        n_carrier=("n_carrier", "mean")
    ).reset_index()

    summary_statistics = summary_statistics.groupby(["group"], observed=True).agg(
        dist_mean=("dist", "mean"),
        dist_std=("dist", "std"),
        HHI_mean=("HHI", "mean"),
        HHI_std=("HHI", "std"), 
        fx=("fx", "sum"),
        pax=("pax_total", "sum"), 
        pax_per_qtr_mean=("pax_per_qtr", "mean"),
        pax_per_qtr_std=("pax_per_qtr", "std"),
        number_of_routes=("route", "nunique"),
        n_carrier_mean=("n_carrier", "mean"),
        n_carrier_std=("n_carrier", "std")
    ).reset_index()

    summary_statistics["fare_mean"] = summary_statistics["fx"] / summary_statistics["pax"]
    summary_statistics = summary_statistics.drop(columns=["fx", "pax"])

    summary_statistics = summary_statistics.set_index("group")
    summary_statistics = summary_statistics.T
    summary_statistics = summary_statistics.loc[[
        "number_of_routes", "fare_mean",
        "pax_per_qtr_mean", "pax_per_qtr_std",
        "n_carrier_mean", "n_carrier_std",
        "HHI_mean", "HHI_std",
        "dist_mean", "dist_std"
    ]]

    summary_statistics.style.format("{:.2f}").to_latex(tables_folder / f"table1_{label}.tex", hrules=True)

    raw_trends = seg.set_index(["route", "yearq"])
    raw_trends["qpr"] = raw_trends.groupby("route", observed=True)["t"].transform("nunique")
    raw_trends_balanced = raw_trends[raw_trends["qpr"] == raw_trends["t"].nunique()]
    raw_trends_balanced = raw_trends_balanced.drop(columns="qpr")
    raw_trends_balanced = raw_trends_balanced.groupby(["group", "yearq"], observed=True)["log_fare"].mean()
    raw_trends_balanced = raw_trends_balanced.unstack("group")
    raw_trends_balanced = raw_trends_balanced - raw_trends_balanced.loc[cfg.ES_REFERENCE]

    raw_trends_balanced.plot(figsize=(10, 6))

    events = {
        "2008Q2": "Announcement",
        "2008Q4": "Close",
        "2010Q1": "Integration"
    }

    for date, event in events.items():
        x_pos = raw_trends_balanced.index.get_loc(date)
        plt.axvline(x=x_pos, color="red", linestyle="--", linewidth=1.5)
        plt.text(x=x_pos - 0.5, y=0.11, s=event, color="red", 
                rotation=90, verticalalignment='top')
        
    plt.title(f"Time Series Plot of Log Mean Fares ({label})")
    plt.xlabel("Year / Quarter")
    plt.ylabel("Log fares relative to 2008Q1")
    plt.grid(True)
    plt.savefig(figures_folder / f"fig1_trends_{label}.pdf", format="pdf", bbox_inches="tight")
    plt.close()


make_descriptives(estimation_sample, False, "connecting")
make_descriptives(estimation_sample, True,  "nonstop")
