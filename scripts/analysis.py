import pathlib
import sys

import pandas as pd
import torch
from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.data.transformation_save_data import \
    TransformationSaveData
from reg23_experiments.io.image import XrayDICOM, read_dicom
from reg23_experiments.io.save_data import load_latest_save
from reg23_experiments.utils import logs_setup
from scipy.spatial.transform import Rotation as R


def get_uid(path: str | pathlib.Path) -> str:
    path = pathlib.Path(path)
    dicom: XrayDICOM = read_dicom(path)
    return dicom["uid"]


def load_ground_truth(xray_sop_instance_uid: str, saved_transformations: pd.DataFrame) -> Transformation:
    idx = (xray_sop_instance_uid, "gold_standard")
    row = saved_transformations.loc[idx]
    return Transformation.from_vector(torch.tensor([row[f"x{i}"] for i in range(6)], dtype=torch.float64))


def angle_difference(a: Transformation, b: Transformation) -> float:
    # axis-angle vectors: axis * angle (radians)
    r1 = R.from_rotvec(a.rotation.numpy())
    r2 = R.from_rotvec(b.rotation.numpy())

    # relative rotation: apply r1, then rotate back with r2
    relative = r2 * r1.inv()

    # shortest angular offset
    angle = relative.magnitude()
    return angle

    # corresponding axis  # axis = relative.as_rotvec()  # axis /= angle


def main(directory: str | pathlib.Path):
    directory = pathlib.Path(directory)
    assert directory.is_dir()

    # -----
    # Load all saved transformations; these are searched through for ground truth alignments
    res: tuple[pathlib.Path, TransformationSaveData, int] | Error = load_latest_save(  #
        TransformationSaveData,  #
        save_directory=pathlib.Path("data/app_transformation_save_data")  #
    )
    if isinstance(res, Error):
        raise RuntimeError(f"Failed to load saved transformation: {res.description}")
    _, transformation_save_data, _ = res
    saved_transformations: pd.DataFrame = transformation_save_data.get_data()
    logger.info(f"Saved transformation data:\n{saved_transformations.to_string()}")

    def get_ground_truth(filename: str) -> Transformation:
        return load_ground_truth(get_uid(directory / filename), saved_transformations)

    for a, b in [  #
        ("level_000", "up_000"),  #
        ("level_000", "down_000"),  #
        ("up_000", "down_000"),  #
    ]:
        ta, tb = get_ground_truth(a), get_ground_truth(b)
        logger.info(f"Angle between 'gold standard' transformations of images '{a}' and '{b}' is "
                    f"{angle_difference(ta, tb):.4f}")


if __name__ == "__main__":
    logger = logs_setup.setup_logger()

    main(sys.argv[1])
