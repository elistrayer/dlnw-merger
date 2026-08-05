import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

main_dir = Path(__file__).parent.parent
csv_folder = main_dir / "data" / "raw"
paths = sorted(csv_folder.glob("*csv"))

audit = []
for p in paths:
    df = pd.read_csv(p, usecols=cfg.DATA_COLS, dtype=cfg.DATA_TYPES)
    df.info()

    raw_rows = len(df)
    raw_pax = df["Passengers"].sum()
    year = df["Year"].iloc[0]
    quarter = df["Quarter"].iloc[0]
    audit.append({"Filename": p, "Year": year, "Quarter": quarter, "Filter": "File Read", "Rows Before": raw_rows, "Rows After": 
                raw_rows, "Passengers Before": raw_pax, "Passengers After": raw_pax})

    df = df.dropna(subset=["MktFare", "Passengers", "Origin", "Dest", "TkCarrier", "MktCoupons", 
                        "MktGeoType", "BulkFare", "TkCarrierChange"])
    post_nan_rows = len(df)
    post_nan_pax = df["Passengers"].sum()

    assert (post_nan_rows - raw_rows) / raw_rows > -0.01, f"Dropped NaN Rows exceeded 1%: {(post_nan_rows - raw_rows) / raw_rows}%"
    audit.append({"Filename": p, "Year": year, "Quarter": quarter, "Filter": "NaN Drop", "Rows Before": raw_rows, "Rows After": 
                post_nan_rows, "Passengers Before": raw_pax, "Passengers After": post_nan_pax})

    def to_int(s, name, dt):
        assert s.notna().all(), f"NaN Values present in {name} column, cannot cast to {dt}"
        info = np.iinfo(dt)
        assert s.min() >= info.min and s.max() <= info.max, f"Overflow error for {dt} in {name} column."
        return s.astype(dt)

    for col, dt in cfg.INT_COLS.items():
        df[col] = to_int(df[col], col, dt,)

    df = df[df["BulkFare"] != 1]
    post_bulk_rows = len(df)
    post_bulk_pax = df["Passengers"].sum()
    audit.append({"Filename": p, "Year": year, "Quarter": quarter, "Filter": "Bulk Fare", "Rows Before": post_nan_rows, "Rows After": 
                post_bulk_rows, "Passengers Before": post_nan_pax, "Passengers After": post_bulk_pax})

    df = df[df["TkCarrierChange"] != 1]
    post_cchange_rows = len(df)
    post_cchange_pax = df["Passengers"].sum()
    audit.append({"Filename": p, "Year": year, "Quarter": quarter, "Filter": "Carrier Change", "Rows Before": post_bulk_rows, "Rows After": 
                post_cchange_rows, "Passengers Before": post_bulk_pax, "Passengers After": post_cchange_pax})

    df = df[df["MktGeoType"] == 2]
    post_mktgeo_rows = len(df)
    post_mktgeo_pax = df["Passengers"].sum()
    audit.append({"Filename": p, "Year": year, "Quarter": quarter, "Filter": "Geography Restriction", "Rows Before": post_cchange_rows, "Rows After": 
                post_mktgeo_rows, "Passengers Before": post_cchange_pax, "Passengers After": post_mktgeo_pax})

    if cfg.TRIM_MODE == "fixed":
        df = df[(df["MktFare"] > cfg.FARE_MIN)]
        post_faremin_rows = len(df)
        post_faremin_pax = df["Passengers"].sum()
        audit.append({"Filename": p, "Year": year, "Quarter": quarter, "Filter": "Fare Min (fixed)", "Rows Before": post_mktgeo_rows, "Rows After": 
                post_faremin_rows, "Passengers Before": post_mktgeo_pax, "Passengers After": post_faremin_pax})

        df = df[(df["MktFare"] < cfg.FARE_MAX)]
        post_faremax_rows = len(df)
        post_faremax_pax = df["Passengers"].sum()
        audit.append({"Filename": p, "Year": year, "Quarter": quarter, "Filter": "Fare Max (fixed)", "Rows Before": post_faremin_rows, "Rows After": 
                post_faremax_rows, "Passengers Before": post_faremin_pax, "Passengers After": post_faremax_pax})

    elif cfg.TRIM_MODE == "percentile":
        lower_limit = df["MktFare"].quantile(cfg.FARE_PCT_MIN)
        upper_limit = df["MktFare"].quantile(cfg.FARE_PCT_MAX)
        
        df = df[(df["MktFare"] > lower_limit)]
        post_faremin_rows = len(df)
        post_faremin_pax = df["Passengers"].sum()
        audit.append({"Filename": p, "Year": year, "Quarter": quarter, "Filter": "Fare Min (percentile)", "Rows Before": post_mktgeo_rows, "Rows After": 
                post_faremin_rows, "Passengers Before": post_mktgeo_pax, "Passengers After": post_faremin_pax})

        df = df[(df["MktFare"] < upper_limit)]
        post_faremax_rows = len(df)
        post_faremax_pax = df["Passengers"].sum()
        audit.append({"Filename": p, "Year": year, "Quarter": quarter, "Filter": "Fare Max (percentile)", "Rows Before": post_faremin_rows, "Rows After": 
                post_faremax_rows, "Passengers Before": post_faremin_pax, "Passengers After": post_faremax_pax})

    else:
        raise ValueError(f"Unknown TRIM MODE, {cfg.TRIM_MODE}")

    assert len(df) > 0, "DataFrame empty"

    o, d = df["Origin"].astype(str), df["Dest"].astype(str)
    df["route"] = np.where(o > d, o + '-' + d, d + '-' + o)
    del o, d
    df["route"] = df["route"].astype("category")
    df["nonstop"] = df["MktCoupons"] == 1
    df["fare_x_pax"] = df["MktFare"] * df["Passengers"]

    om, dm = df["OriginCityMarketID"], df["DestCityMarketID"]
    df["RouteMarketID"] = np.where(om < dm, om * cfg.ID_MULTIPLIER + dm, dm * cfg.ID_MULTIPLIER + om)

    assert df["Year"].nunique() == 1 and df["Quarter"].nunique() == 1
    year = df["Year"].iloc[0]
    quarter = df["Quarter"].iloc[0]

    g = df.groupby(["route", "TkCarrier", "nonstop"], observed=True).agg(fx = ("fare_x_pax", "sum"), pax = ("Passengers", "sum"),
                                                                        n_obs = ("Passengers", "size"), dist = ("MktDistance", "mean"),
                                                                        mktId = ("RouteMarketID", "first")).reset_index()
    g["Year"] = year
    g["Quarter"] = quarter

    assert g.pax.sum() == df.Passengers.sum(), f"Collapsed DataFrame passengers does not match original passengers: {g.pax.sum()} != {df.Passengers.sum()}"
    assert g.n_obs.sum() == len(df), f"Number of observations in collapsed DataFrame differs from original length: {g.n_obs.sum()} != {len(df)}"
    assert np.isclose(g.fx.sum(), df.fare_x_pax.sum())

    post_collapse_rows = len(g)
    post_collapse_pax = g["pax"].sum()
    audit.append({"Filename": p, "Year": year, "Quarter": quarter, "Filter": "Collapse", "Rows Before": post_faremax_rows, "Rows After": 
                post_collapse_rows, "Passengers Before": post_faremax_pax, "Passengers After": post_collapse_pax})

    parquet_path = csv_folder = main_dir / "data" / "clean" / f"{p.stem}.parquet"
    g.to_parquet(parquet_path, engine="pyarrow", index=False)


audit_path = main_dir / "output" / "audit.csv"
audit_df = pd.DataFrame(audit)
audit_df.to_csv(audit_path, index=False)
