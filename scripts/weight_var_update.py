import pathlib
import sys

import pandas as pd

from reg23_experiments.data.structs import Error


def main(p: str) -> Error | None:
    p = pathlib.Path(p)
    if not p.is_dir():
        return Error(f"{str(p)} is not a directory")

    out = p.with_name(p.name + "_updated")
    while out.is_dir():
        out = out.with_name(out.name + "_again")
    out.mkdir(exist_ok=False)

    for f in p.iterdir():
        if not f.is_file():
            continue
        if f.suffix != ".parquet":
            continue
        if not f.name.startswith("data_"):
            continue

        df = pd.read_parquet(f)
        df["weight_alpha"] = 0.0
        df["weight_alpha"] = df["weighting"].fillna(df["weight_alpha"])
        df["apply_weighting"] = df["weighting"].notna()
        df = df.drop(columns=["weighting"])

        this_out = out / f.name
        df.to_parquet(this_out)
        print(f"Wrote to {str(this_out)}")

    return None


if __name__ == "__main__":
    _err = main(sys.argv[1])
    if isinstance(_err, Error):
        print("ERROR:", _err)
        exit(1)
