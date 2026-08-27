import pathlib
import sys

import pandas as pd
import yaml

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
        df["weighting_method"] = df["apply_weighting"].map({False: "none", True: "smooth_step"})
        df = df.drop(columns=["apply_weighting"])

        this_out = out / f.name
        df.to_parquet(this_out)
        print(f"Wrote to {str(this_out)}")

    variables_file_in = p / "variables.txt"
    if variables_file_in.is_file():
        with open(variables_file_in, 'r') as file:
            variables_config = yaml.safe_load(file)

        assert isinstance(variables_config, dict)
        for section_contents in variables_config.values():
            assert isinstance(section_contents, dict)

        def map_value(value, var_name):
            if var_name == "apply_weighting":
                return "smooth_step" if value else "none"
            return value

        new_variables_config = {  #
            section_name: {  #
                ("weighting_method" if variable_name == "apply_weighting" else variable_name):  #
                    ([  #
                         map_value(e, variable_name)  #
                         for e in value  #
                     ] if isinstance(value, list) else map_value(value, variable_name))  #
                for variable_name, value in section_contents.items()  #
            }  #
            for section_name, section_contents in variables_config.items()  #
        }

        variables_file_out = out / "variables.txt"
        with open(variables_file_out, 'w') as file:
            yaml.safe_dump(new_variables_config, file, sort_keys=False)  # very important to preserve order of keys
    else:
        print(f"Warning: No variables.txt found.")

    return None


if __name__ == "__main__":
    _err = main(sys.argv[1])
    if isinstance(_err, Error):
        print("ERROR:", _err)
        exit(1)
