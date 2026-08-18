import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

main_dir = Path(__file__).parent.parent
panel_folder = main_dir / "data" / "panels"

route_group = pd.read_parquet(panel_folder / "route_group.parquet")
estimation_sample = route_group[route_group["group"] != "excluded"]
estimation_sample = estimation_sample[estimation_sample["confounded"] != True]


summary_statistics = estimation_sample[estimation_sample["period"] == "pre"]
summary_statistics = summary_statistics[summary_statistics["nonstop"] == False]

# one column per group
# number of routes, mean fare, passengers per quarter, HHI, carriers per route, distance

# pax per q -> groupby route ->

summary_statistics = summary_statistics.groupby(["route", "group"], observed=True).agg(pax_total=("pax", "sum"), dist=("dist", "mean"), 
                                                fx=("fx", "sum"), HHI=("HHI", "mean"), pax_per_qtr=("pax", "mean"), n_carrier=("n_carrier", "mean")).reset_index()

summary_statistics = summary_statistics.groupby(["group"], observed=True).agg(dist_mean=("dist", "mean"), HHI_mean=("HHI", "mean"), fx=("fx", "sum"), 
                                                pax=("pax_total", "sum"), pax_per_qtr=("pax_per_qtr", "mean"), number_of_routes=("route", "nunique"),
                                                n_carrier=("n_carrier", "mean")).reset_index()

summary_statistics["fare_mean"] = summary_statistics["fx"] / summary_statistics["pax"]
summary_statistics = summary_statistics.drop(columns=["fx", "pax"])

summary_statistics = summary_statistics.set_index("group")
summary_statistics = summary_statistics.T
print(summary_statistics)




