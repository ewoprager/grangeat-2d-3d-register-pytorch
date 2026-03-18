import argparse
from typing import Any
import pprint

import pathlib
import pandas as pd
import numpy as np

from reg23_experiments.utils import logs_setup
from reg23_experiments.analysis.helpers import get_axis_values_if_dataframe_rectangular_over_columns

KNOWN_MEASUREMENT_COLUMNS = ["distance"]


def main(data_dir: pathlib.Path):
    logger.info(f"Summary of data stored in directory '{str(data_dir)}':")
    sub_directories = [element for element in data_dir.iterdir() if element.is_dir()]
    sub_directories = sorted(sub_directories)
    for sub_dir in sub_directories:
        data_files = []
        for sub_element in sub_dir.iterdir():
            if sub_element.stem.startswith("data") and sub_element.suffix == ".parquet":
                data_files.append(sub_element)
        notes_file = sub_dir / "notes.txt"
        if notes_file.is_file():
            if not data_files:
                logger.warning(f"Notes found in directory '{sub_dir}', but no data files.")
                continue
            notes = notes_file.read_text()
        else:
            if data_files:
                logger.warning(f"Data file(s) found in directory '{sub_dir}', but no notes.")
            continue
        string = f"\nSubdirectory '{str(sub_dir.name)}':\n\tNotes: '{notes}'"
        # read in the full DataFrame from all the files
        df = pd.concat([  #
            pd.read_parquet(data_file)  #
            for data_file in data_files  #
        ])
        # get columns that contain only one unique value
        constant_cols: list[str] = [col for col in df.columns if df[col].nunique() == 1]
        # get the values of the constant columns as a dict
        constants: dict[str, Any] = {  #
            col: df[col].values[0]  #
            for col in constant_cols  #
        }
        string += f"\n\tConstant variables: \n{pprint.pformat(constants, indent=8)}"
        # get the rest of the columns
        variable_cols: list[str] = [col for col in df.columns if col not in constant_cols]
        # find any known measurement columns that are present, and remove them from `variable_cols`
        measurement_cols: list[str] = []
        for col in KNOWN_MEASUREMENT_COLUMNS:
            if col in variable_cols:
                variable_cols.remove(col)
                measurement_cols.append(col)
        if measurement_cols:
            string += f"\n\tMeasurement variables: {measurement_cols}"
        # check if the dataframe is rectangular over the remaining variable columns, and if so, get the axis values
        axis_values: list[np.ndarray] | None = get_axis_values_if_dataframe_rectangular_over_columns(  #
            df, columns=variable_cols)
        if axis_values is not None:
            axis_value_dict: dict[str, list] = {  #
                name: array.tolist()  #
                for name, array in zip(variable_cols, axis_values)  #
            }
            string += (f"\n\tDataFrame is rectangular over remaining variables; axis values: \n"
                       f"{pprint.pformat(axis_value_dict, indent=8)}")
        else:
            string += (f"\n\tDataFrame is not rectangular over remaining variables, which are: "
                       f"{variable_cols}")
        string += "\n"
        logger.info(string)


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument("-d", "--data-dir", type=str, default="data/temp/program_truncation",
                        help="Directory in which to find the data files.")
    args = parser.parse_args()

    main(data_dir=pathlib.Path(args.data_dir))
