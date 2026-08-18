import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

audit = []

main_dir = Path(__file__).parent.parent
panel_folder = main_dir / "data" / "panels"
carrier_panel = panel_folder / "carrier_shares.parquet"
route_panel = panel_folder / "route_panel.parquet"

carrier_shares = pd.read_parquet(carrier_panel)
carrier_shares = carrier_shares[carrier_shares["t"] <= cfg.q_to_t(cfg.PRE_END)]
assert (carrier_shares["t"].max() == cfg.q_to_t(cfg.PRE_END)) and (carrier_shares["t"].nunique() == cfg.q_to_t(cfg.PRE_END))

route_check = carrier_shares.groupby(["route", "nonstop"], observed=True).agg(pre_q=("t", "nunique"), pre_pax=("pax", "sum"))
pre_min_quarter_rows = len(route_check)
route_check = route_check[route_check["pre_q"] >= cfg.MIN_PRE_QUARTERS]
post_min_quarter_rows = len(route_check)

audit.append({"stage": "min_pre_quarters", "route_segments_before": pre_min_quarter_rows,
              "route_segments_after": post_min_quarter_rows})

carrier_shares = carrier_shares.groupby(["route", "nonstop", "TkCarrier"], observed=True).agg(carrier_pre_pax=("pax", "sum"), 
                                                                                             carrier_pre_quarters=("t", "nunique"))

pooled_shares = pd.merge(carrier_shares.reset_index(), route_check.reset_index(), on=["route", "nonstop"], how="inner", validate="m:1")
pooled_shares["pooled_share"] = pooled_shares["carrier_pre_pax"] / pooled_shares["pre_pax"]
assert np.allclose(pooled_shares.groupby(["route","nonstop"], observed=True)["pooled_share"].transform("sum"), 1)
pooled_shares = pooled_shares.rename(columns={"pre_q": "route_pre_quarters"})

eligible_routes = route_check.index
def presence_check(carrier):
    selected_shares = pooled_shares[pooled_shares["TkCarrier"] == carrier]
    selected_shares = selected_shares.set_index(["route", "nonstop"])
    selected_shares = (selected_shares["pooled_share"] >= cfg.OVERLAP_SHARE) & (selected_shares["carrier_pre_quarters"] >= cfg.MIN_PRE_QUARTERS)
    selected_shares = selected_shares.reindex(eligible_routes, fill_value=False)
    return selected_shares

dl_presence = presence_check("DL")
nw_presence = presence_check("NW")

treated = dl_presence & nw_presence
controlA = dl_presence ^ nw_presence
controlB = ~(dl_presence | nw_presence)

assert (treated.astype(int) + controlA.astype(int) + controlB.astype(int) == 1).all()
assert treated.index.equals(eligible_routes)
group = pd.Series(np.select([treated, controlA, controlB], ["treated", "control_A", "control_B"], default="UNASSIGNED"), 
                  index=eligible_routes, name="group")

pair_presence = {}
for pair in cfg.CONFOUND_PAIRS:
    if set(pair) == set(cfg.TREAT_PAIR): 
        continue
    series = presence_check(pair[0]) & presence_check(pair[1])
    pair_presence[f"{pair[0]}_{pair[1]}"] = series

pair_presence = pd.DataFrame(pair_presence)
confounded = pair_presence.any(axis=1)

route_panel = pd.read_parquet(route_panel)

periods = [
    route_panel["t"] <= cfg.q_to_t(cfg.PRE_END),
    (route_panel["t"] >= cfg.q_to_t(cfg.TRANSITION[0])) & (route_panel["t"] <= cfg.q_to_t(cfg.TRANSITION[1])),
    route_panel["t"] >= cfg.q_to_t(cfg.POST_START)
]

route_panel["period"] = np.select(periods, ["pre", "transition", "post"], default="UNASSIGNED")

treatment = pd.concat([group, confounded.rename("confounded")], axis=1)
route_group = route_panel.merge(treatment.reset_index(), on=["route", "nonstop"], how="left", validate="m:1", indicator=True)
route_group["group"] = route_group["group"].fillna("excluded")
route_group["confounded"] = route_group["confounded"].fillna(False).astype(bool)

print(route_group["_merge"].value_counts())

post = route_group["period"] == "post"
route_group["treat_post"] = (route_group["group"] == "treated") & post
assert route_group["period"].ne("UNASSIGNED").all()

route_group.to_parquet(panel_folder / "route_group.parquet", index=False)

counts = route_group.groupby(["nonstop", "group"], observed=True).agg(
    n_routes = ("route", "nunique"),
    n_route_quarters = ("route", "size"),
    total_pax = ("pax", "sum")
).reset_index()

g = route_group.groupby(["nonstop","group"], observed=True)
counts["mean_fare"] = (g["fx"].sum() / g["pax"].sum()).values

counts.to_csv(main_dir / "output" / "treatment_summary.csv", index=False)
print(counts)

route_totals = route_group.groupby(["nonstop", "group", "route"], observed=True)["pax"].sum().reset_index()
route_totals = route_totals.sort_values("pax", ascending=False)
examples = route_totals.groupby(["nonstop","group"], observed=True).head(5)
print(examples)

pre_route_totals = route_group[route_group["period"] == "pre"]
pre_route_totals = pre_route_totals.groupby(["route", "nonstop", "group"], observed=True).agg(pax_total=("pax", "sum"), dist=("dist", "mean"), 
                                                                                              fx=("fx", "sum"), HHI=("HHI", "mean"), pax_per_qtr=("pax", "mean"))
pre_route_totals = pre_route_totals.groupby(["group", "nonstop"], observed=True).agg(dist_mean=("dist", "mean"), HHI_mean=("HHI", "mean"), fx=("fx", "sum"), 
                                                                                     pax=("pax_total", "sum"), pax_per_qtr=("pax_per_qtr", "mean")).reset_index()
pre_route_totals["fare_mean"] = pre_route_totals["fx"] / pre_route_totals["pax"]
print(pre_route_totals)

conf = route_group.drop_duplicates(["route", "nonstop"])
print(conf.groupby(["nonstop", "group"], observed=True)["confounded"].agg(["sum", "mean"]))

pd.DataFrame(audit).to_csv(main_dir/"output"/"audit_step3.csv", index=False)