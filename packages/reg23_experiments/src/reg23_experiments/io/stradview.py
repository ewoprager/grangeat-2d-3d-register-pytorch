import pathlib
import pandas as pd

import torch

from reg23_experiments.data.structs import Error

__all__ = ["extract_landmarks"]


def extract_landmarks(stradview_csv_path: pathlib.Path) -> list[tuple[str, torch.Tensor]] | Error:
    if not stradview_csv_path.is_file():
        return Error(f"'{str(stradview_csv_path)}' is not a file.")

    df = pd.read_csv(stradview_csv_path)

    first_row = df.loc[0]
    for i, row in df.iterrows():
        if row["Type"] != "Surface":
            return Error(f"All landmarks should be of type 'Surface'. Found '{row["Type"]}' for landmark {i}.")
        if row["Surface"] != first_row["Surface"]:
            return Error(f"All landmarks should be on the same surface. Found '{row["Surface"]}' for landmark {i}, "
                         f"but '{first_row["Surface"]}' for landmark 0.")

    return [  #
        (row["Landmark"], torch.tensor([  #
            float(row[col])  #
            for col in ["x (mm)", "y (mm)", "z (mm)"]  #
        ]))  #
        for _, row in df.iterrows()  #
    ]
