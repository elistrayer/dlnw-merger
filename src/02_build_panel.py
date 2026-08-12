import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

audit = []

main_dir = Path(__file__).parent.parent
clean_folder = main_dir / "data" / "clean"
panel_folder = main_dir / "data" / "panels"
output = main_dir / "output"
(main_dir / "output").mkdir(parents=True, exist_ok=True)
panel_folder.mkdir(parents=True, exist_ok=True)
paths = sorted(clean_folder.glob("*parquet"))
assert len(paths) == 24, f"Expected 24 Parquets, only found {len(paths)}"

data_list = []
for p in paths:
    df = pd.read_parquet(p)
    data_list.append(df)

df = pd.concat(data_list, ignore_index=True)
df["route"] = df["route"].astype("category")
df["yearq"] = df["Year"].astype(str) + 'Q' + df["Quarter"].astype(str)
df["t"] = (df["Year"] - 2006) * 4 + df["Quarter"]
assert not df.duplicated(subset=["route", "TkCarrier", "nonstop", "yearq"]).any()

carrier_panel = df.copy()
carrier_panel["route_pax"] = carrier_panel.groupby(["route", "nonstop", "yearq"], observed=True)["pax"].transform("sum")
carrier_panel = carrier_panel[carrier_panel["route_pax"] >= cfg.MIN_PAX_ROUTE_QTR].copy()

triples_before = df.drop_duplicates(subset=["route", "nonstop", "yearq"]).shape[0]
triples_after = carrier_panel.drop_duplicates(subset=["route", "nonstop", "yearq"]).shape[0]

audit.append({"Stage": "Min Pax Qtr", "Rows Before": len(df), "Rows After": len(carrier_panel), 
              "Passengers Before": df["pax"].sum(), "Passengers After": carrier_panel["pax"].sum(),
              "Triples Before": triples_before, "Triples After": triples_after})

carrier_panel["share"] = carrier_panel["pax"] / carrier_panel["route_pax"]
assert np.allclose(carrier_panel.groupby(["route", "yearq", "nonstop"], observed=True)["share"].transform("sum"), 1)


route_panel = carrier_panel.copy()
route_panel["share_sq"] = route_panel["share"] ** 2
route_panel = route_panel.groupby(["route", "nonstop", "yearq"], observed=True).agg(t = ("t", "max"), fx=("fx", "sum"), 
                                                                        pax=("pax", "sum"), HHI=("share_sq", "sum"),
                                                                        n_carrier=("TkCarrier", "nunique"),
                                                                        n_obs = ("n_obs", "sum"), dist=("dist", "mean"),
                                                                        mktId=("mktId", "first")).reset_index()
route_panel["HHI"] = route_panel["HHI"] * 10000
route_panel["fare"] = route_panel["fx"] / route_panel["pax"]

assert route_panel["fare"].min() > 0
assert route_panel["pax"].min() > 0

route_panel["log_fare"] = np.log(route_panel["fare"])
route_panel["log_pax"] = np.log(route_panel["pax"])

assert route_panel["pax"].sum() == carrier_panel["pax"].sum()
assert (route_panel.groupby(["route"], observed=True)["mktId"].nunique() == 1).all()
assert (route_panel["HHI"] >= 0).all() and (route_panel["HHI"] <= 10000).all()
assert not route_panel.duplicated(subset=["route", "nonstop", "yearq"]).any()
assert route_panel["pax"].min() >= cfg.MIN_PAX_ROUTE_QTR

summary = route_panel.groupby("nonstop", observed=True).agg(
    n_routes=("route", "nunique"),
    n_route_quarters=("route", "size"),
    n_quarters=("yearq", "nunique"),
    total_pax=("pax", "sum"),
    mean_hhi=("HHI", "mean"),
    mean_carriers=("n_carrier", "mean")
).reset_index()

grouped_route = route_panel.groupby(["nonstop"], observed=True)
summary["mean_fare"] = (grouped_route["fx"].sum() / grouped_route["pax"].sum()).values
summary["segment"] = np.where(summary["nonstop"], "nonstop", "connecting")

quarters_per_route = route_panel.groupby(["route", "nonstop"], observed=True)["t"].nunique()
is_balanced = quarters_per_route == route_panel["t"].nunique()
pct_balanced = is_balanced.groupby(level="nonstop").mean()
summary["pct_balanced"] = pct_balanced.values
summary = summary.drop(columns=["nonstop"])
summary.to_csv(output / "panel_summary.csv", index=False)

quarters_per_route = quarters_per_route.groupby(level="nonstop").value_counts().reset_index()
quarters_per_route.to_csv(output / "panel_balance.csv", index=False)

audit_df = pd.DataFrame(audit)
audit_df.to_csv(output / "audit_step2.csv", index=False)

carrier_panel.to_parquet(panel_folder / "carrier_shares.parquet", engine="pyarrow", index=False)
route_panel.to_parquet(panel_folder / "route_panel.parquet", engine="pyarrow", index=False)