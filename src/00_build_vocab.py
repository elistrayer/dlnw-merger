import pandas as pd
from pathlib import Path

airports = set()
carriers = set()

script_dir = Path(__file__).parent
csv_folder = script_dir.parent / "data" / "raw"
paths = sorted(csv_folder.glob("*.csv"))
assert len(paths) == 24, f"Expected 24 CSVs, only found {len(paths)}"

for p in paths:
    df = pd.read_csv(p, usecols=["Origin", "Dest", "TkCarrier"])
    airports |= set(df.Origin.dropna().unique())
    airports |= set(df.Dest.dropna().unique())
    carriers |= set(df.TkCarrier.dropna().unique())
    print(f"{p.name}: {len(airports)} airports and {len(carriers)} carriers so far.")

vocab = script_dir.parent / "vocab.py"
with open(vocab, "w") as file:
    file.write(f"# WRITTEN BY 00_Build_Vocab.py -- don't edit by hand\n")
    file.write(f"AIRPORTS = {sorted(airports)!r}\n")
    file.write(f"CARRIERS = {sorted(carriers)!r}")

print(f"Wrote {len(airports)} airports and {len(carriers)} to {vocab}")