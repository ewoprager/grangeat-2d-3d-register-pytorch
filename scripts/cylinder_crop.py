import argparse
import pathlib
from typing import Callable, NamedTuple

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import nrrd
import torch
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider

from reg23_experiments.data.structs import Error
from reg23_experiments.io.volume import OneSeries, SeriesDescription, Volume, find_ct_series, \
    get_input_ct_series_choice, load_ct_series
from reg23_experiments.ops.ct import convert_ct_to_mu
from reg23_experiments.utils import logs_setup


class SliderSpec(NamedTuple):
    vmin: float
    vmax: float
    callback: Callable
    slider_kwargs: dict = {}


class VolumeViewer:
    def __init__(self, volume: np.ndarray):
        """
        volume shape: (z, y, x)
        """
        self.volume = volume

        slice_rad = max(self.volume.shape[1], self.volume.shape[2])

        self.fig, self.ax = plt.subplots()

        self.slice_idx = volume.shape[0] // 2

        self._vmin = np.min(volume)
        self._vmax = np.max(volume)
        self._x_radius = 40.0
        self._y_radius = 40.0

        slider_specs: dict[str, SliderSpec] = {  #
            "Slice": SliderSpec(vmin=0, vmax=volume.shape[0] - 1,
                                slider_kwargs={"valinit": self.slice_idx, "valstep": 1}, callback=self._update),  #
            "VMin": SliderSpec(vmin=self._vmin, vmax=self._vmax, slider_kwargs={"valinit": self._vmin},
                               callback=self._update),  #
            "VMax": SliderSpec(vmin=self._vmin, vmax=self._vmax, slider_kwargs={"valinit": self._vmax},
                               callback=self._update),  #
            "XRadius": SliderSpec(vmin=20, vmax=slice_rad, slider_kwargs={"valinit": self._x_radius},
                                  callback=self._update),  #
            "YRadius": SliderSpec(vmin=20, vmax=slice_rad, slider_kwargs={"valinit": self._y_radius},
                                  callback=self._update),  #
            "DirX": SliderSpec(vmin=-20.0, vmax=20.0, slider_kwargs={"valinit": 0.0}, callback=self._update),  #
            "DirY": SliderSpec(vmin=-20.0, vmax=20.0, slider_kwargs={"valinit": 0.0}, callback=self._update),  #
            "CentreX": SliderSpec(vmin=-slice_rad, vmax=slice_rad, slider_kwargs={"valinit": 0.0},
                                  callback=self._update),  #
            "CentreY": SliderSpec(vmin=-slice_rad, vmax=slice_rad, slider_kwargs={"valinit": 0.0},
                                  callback=self._update),  #
        }

        plt.subplots_adjust(bottom=0.1 + 0.05 * len(slider_specs))

        # Display initial slice
        self.im = self.ax.imshow(self.volume[self.slice_idx], cmap="gray", origin="lower", vmin=self._vmin,
                                 vmax=self._vmax)

        self.ax.set_title(f"Slice {self.slice_idx}")

        # --- Sliders --------------------------------------------------------
        y = len(slider_specs) * 0.05
        self._sliders = {}
        for k, v in slider_specs.items():
            axes = plt.axes((0.15, y, 0.7, 0.03))
            y -= 0.05
            self._sliders[k] = Slider(axes, k, v.vmin, v.vmax, **v.slider_kwargs)
            self._sliders[k].on_changed(v.callback)

        self._centre_offset = np.zeros(2)
        self._direction = np.array([0.0, 0.0, 1.0])
        self._direction /= np.linalg.norm(self._direction)
        self._ellipse = Ellipse((0.0, 0.0), self._x_radius, self._y_radius, fill=False, edgecolor="red", linewidth=2)
        self._update_ellipse()
        self.ax.add_patch(self._ellipse)

    def _update_ellipse(self):
        self._centre_offset = np.array([self._sliders["CentreX"].val, self._sliders["CentreY"].val])
        self._direction = np.array([self._sliders["DirX"].val, self._sliders["DirY"].val, 10.0])
        self._direction /= np.linalg.norm(self._direction)
        centre = (float(self.slice_idx) - 0.5 * float(self.volume.shape[0])) * self._direction[
            0:2] + self._centre_offset
        self._ellipse.set_center((self.volume.shape[2] // 2 + centre[0], self.volume.shape[1] // 2 + centre[1]))
        self._ellipse.set_width(self._sliders["XRadius"].val)
        self._ellipse.set_height(self._sliders["YRadius"].val)
        self.fig.canvas.draw_idle()

    def _update(self, _):
        self.slice_idx = int(self._sliders["Slice"].val)

        self._vmin = self._sliders["VMin"].val
        self._vmax = self._sliders["VMax"].val
        self.im.set_clim(self._vmin, self._vmax)
        # Update displayed image
        self.im.set_data(self.volume[self.slice_idx])

        self.ax.set_title(f"Slice {self.slice_idx} | "
                          f"vmin={self._vmin:.2f} | "
                          f"vmax={self._vmax:.2f}")

        # Put any future processing here using vmin/vmax
        self._update_ellipse()

        self.fig.canvas.draw_idle()

    def apply_crop(self) -> np.ndarray:
        ret = self.volume.copy()
        for i in range(len(ret)):
            centre = (float(self.slice_idx) - 0.5 * float(self.volume.shape[0])) * self._direction[
                0:2] + self._centre_offset
            ys, xs = np.indices(self.volume.shape[1:3])
            xs = (xs - self.volume.shape[2] // 2 + centre[0]) / self._x_radius
            ys = (ys - self.volume.shape[1] // 2 + centre[1]) / self._y_radius
            mask = xs * xs + ys * ys > 1.0
            ret[i][mask] = 0.0
        return ret

    def show(self):
        plt.show()


def main(*, ct_path: pathlib.Path, output_path: pathlib.Path) -> None | Error:
    device = torch.device("cpu")

    series: dict[str, SeriesDescription | OneSeries] = find_ct_series(ct_path)
    if not series:
        return Error(f"No CT series found at path '{str(ct_path)}'.")
    if len(series) == 1:
        key = next(iter(series))
    else:
        key = get_input_ct_series_choice(series)
    volume: Volume | Error = load_ct_series(ct_path, key)
    if isinstance(volume, Error):
        return Error(f"Failed to open CT from path '{str(ct_path)}': {volume.description}")
    if volume.image_position_patient is None:
        logger.warning(f"No ImagePositionPatient found for given CT series. Assuming (0, 0, 0).")
        volume.image_position_patient = torch.zeros(3, dtype=torch.float64)
    tensor: torch.Tensor | Error = convert_ct_to_mu(volume, dtype=torch.float32)
    if isinstance(tensor, Error):
        return Error(f"Failed to convert CT from path '{str(ct_path)}' to mu: {tensor.description}")
    ct_volume = tensor.to(device=device)
    ct_spacing = volume.spacing.to(device=device, dtype=torch.float64)
    image_position_patient = volume.image_position_patient.to(device=device, dtype=torch.float64)
    logger.info(
        "CT volume loaded, size=[{} x {} x {}]; spacing=({:.3f}, {:.3f}, {:.3f}); image position patient=({:.3f}, "
        "{:.3f}, {:.3f})".format(  #
            ct_volume.size()[0], ct_volume.size()[1], ct_volume.size()[2],  #
            ct_spacing[0].item(), ct_spacing[1].item(), ct_spacing[2].item(),  #
            image_position_patient[0].item(), image_position_patient[1].item(), image_position_patient[2].item(),  #
        ))

    viewer = VolumeViewer(ct_volume.numpy())

    viewer.show()

    output = viewer.apply_crop()

    logger.info(f"Saving volume of size {output.shape}...")

    header = {  #
        "space": "left-posterior-superior",  #
        "dimension": 3,  #
        "space directions": [  #
            [0.0, 0.0, ct_spacing[2].item()],  # Z axis
            [0.0, ct_spacing[1].item(), 0.0],  # Y axis
            [ct_spacing[0].item(), 0.0, 0.0],  # X axis
        ],  #
        "space origin": image_position_patient.tolist(),  #
        "encoding": "raw"  #
    }
    nrrd.write(str(output_path), output, header=header)
    logger.info(f"Output saved to '{str(output_path)}'.")


if __name__ == "__main__":
    logger = logs_setup.setup_logger()

    _parser = argparse.ArgumentParser(description="", epilog="")
    _parser.add_argument("-c", "--ct-path", type=str, default=None, required=True,
                         help="Give a path to a .nrrd file, .nii file or directory of DICOM files containing CT data to"
                              " modify.")
    _parser.add_argument("-o", "--output-path", type=str, default=None, required=True,
                         help="Give a path at which to save the modified CT data.")
    _args = _parser.parse_args()

    if isinstance(err := main(  #
            ct_path=pathlib.Path(_args.ct_path),  #
            output_path=pathlib.Path(_args.output_path)  #
    ), Error):
        logger.error(err.description)
        exit(1)
