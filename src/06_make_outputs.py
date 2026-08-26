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

estimation_sample = route_group[route_group["group"] != "excluded"]
estimation_sample = estimation_sample[estimation_sample["confounded"] != True]


def estimation_sample_function(df, segment, group):
    df = df[df["nonstop"] == segment]

    df = df[df["group"].isin(["treated", group])].copy()

    df["post"] = (df["period"] == "post")
    df["treat_post"] = (df["post"] & (df["group"] == "treated"))

    assert df["group"].nunique() == 2
    assert df["treat_post"].any()

    return df

estimation_sample = estimation_sample_function(estimation_sample, False, "control_A")
estimation_sample["treated"] = estimation_sample["group"] == "treated"
estimation_sample["treated"] = estimation_sample["treated"].astype(int)



m = pf.feols(f"log_fare ~ i(yearq, treated, ref='{cfg.ES_REFERENCE}') | route + yearq", data=estimation_sample, vcov={"CRV1": "route"})
m_values = m.tidy().reset_index()

m_values["Coefficient"] = m_values["Coefficient"].str.extract(r"(\d{4}Q\d)")
m_values = m_values.drop(columns=["Std. Error", "t value", "Pr(>|t|)"])
m_values = m_values.set_index("Coefficient")
m_values.loc[cfg.ES_REFERENCE] = {"Estimate": np.nan, "2.5%": np.nan, "97.5%": np.nan}
m_values = m_values.sort_index()

m_values.to_csv(tables_folder / "m_values.csv")

yearqs = m_values.index
estimates = m_values["Estimate"]

lower_error = estimates - m_values["2.5%"]
upper_error = m_values["97.5%"] - estimates
asymmetric_error = [lower_error, upper_error]

plt.figure(figsize=(10, 6))
plt.errorbar(
    x=yearqs,
    y=estimates,
    yerr=asymmetric_error,
    fmt="o",
    color="blue",
    ecolor="gray",
    capsize=4,
    label="Estimate (95% CI)"
)

plt.plot(cfg.ES_REFERENCE, 0, marker="o", markerfacecolor="white",
         markeredgecolor="blue", markersize=8, label="Reference (normalized)")

events = {
        "2008Q2": "Announcement",
        "2008Q4": "Close",
        "2010Q1": "Integration"
    }

for date, event in events.items():
        x_pos = m_values.index.get_loc(date)
        plt.axvline(x=x_pos, color="red", linestyle="--", linewidth=1.5)
        plt.text(x=x_pos - 0.5, y=plt.ylim()[1] * 0.9, s=event, color="red", 
                rotation=90, verticalalignment='top')

announcement_x = m_values.index.get_loc("2008Q1")
plt.axvspan(xmin=plt.xlim()[0], xmax=announcement_x - 0.5, color='gray', alpha=0.2, label='Pre-period')

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xticks(rotation=45, ha='right')
plt.title("Event Study Estimates")
plt.xlabel("Quarter")
plt.ylabel("Effect on Log Fare")
plt.legend()
plt.tight_layout()

plt.savefig(figures_folder / "fig2_eventstudy.pdf", bbox_inches="tight")

pre = m_values.loc[:"2007Q4"]
print(f"mean pre coef: {pre["Estimate"].mean()}")
print(f"mean pre |coef|: {pre["Estimate"].abs().mean()}")
print(f"mean CI half: {((pre["97.5%"] - pre["2.5%"]) / 2).mean()}")