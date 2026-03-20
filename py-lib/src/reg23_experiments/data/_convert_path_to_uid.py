import json
import sys

import pathlib
import pydicom
import pandas as pd
from tqdm import tqdm


def dataframe_index_mapping(idx):
    return pydicom.dcmread(idx[0])["SOPInstanceUID"].value, idx[1]


def main(directory: pathlib.Path) -> None:
    assert directory.is_dir()
    for element in tqdm(directory.iterdir(), desc=f"Iterating through subdirectories of {str(directory)}"):
        if not element.is_dir():
            continue
        log_path = element / "log.jsonl"
        snapshot_path = element / "snapshot.parquet"
        assert log_path.is_file()
        assert snapshot_path.is_file()
        # convert the snapshot
        df = pd.read_parquet(snapshot_path)
        df = df.rename_axis(index={"xray_path": "xray_sop_instance_uid"})
        df.index = df.index.map(dataframe_index_mapping)
        df.to_parquet(snapshot_path)

        # convert the log
        new_log_lines = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    change = json.loads(line)
                    assert isinstance(change, dict)
                except Exception as e:
                    raise Exception(f"Error parsing line: '{line}' as JSON from log file '{str(log_path)}': {e}")
                if "xray_path" not in change:
                    continue
                xray_path = change.pop("xray_path")
                change["xray_sop_instance_uid"] = pydicom.dcmread(xray_path)["SOPInstanceUID"].value
                new_log_lines.append(json.dumps(change) + "\n")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.writelines(new_log_lines)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
