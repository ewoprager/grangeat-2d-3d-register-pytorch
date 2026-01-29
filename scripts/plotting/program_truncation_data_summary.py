import argparse

import itertools
import pathlib
import pandas as pd

from reg23_experiments.utils import logs_setup
from reg23_experiments.analysis.helpers import dataframe_to_tensor


def main(data_dir: pathlib.Path):
    configs = []
    for element in data_dir.iterdir():
        if not element.is_dir():
            continue
        data_files = []
        for sub_element in element.iterdir():
            if sub_element.stem.startswith("data") and sub_element.suffix == ".parquet":
                data_files.append(sub_element)
        config_file = element / "config.txt"
        if config_file.is_file():
            if not data_files:
                logger.warning(f"Config found in directory '{element}', but no data files.")
                continue
            configs.append(config_file.read_text())
        else:
            if data_files:
                logger.warning(f"Data file(s) found in directory '{element}', but no config.")
            continue
        # read in the full DataFrame from all the files
        df = pd.concat([  #
            pd.read_parquet(data_file)  #
            for data_file in data_files  #
        ])
        # get columns that contain only one unique value
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]




if __name__ == "__main__":
    logger = logs_setup.setup_logger()

    main()
