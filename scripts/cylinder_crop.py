import argparse
import pathlib
from typing import Callable, NamedTuple

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle
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

        self.fig, self.ax = plt.subplots()

        self.slice_idx = volume.shape[0] // 2

        self._vmin = np.min(volume)
        self._vmax = np.max(volume)
        self._radius = 40.0

        slider_specs: dict[str, SliderSpec] = {  #
            "Slice": SliderSpec(vmin=0, vmax=volume.shape[0] - 1,
                                slider_kwargs={"valinit": self.slice_idx, "valstep": 1}, callback=self._update),  #
            "VMin": SliderSpec(vmin=self._vmin, vmax=self._vmax, slider_kwargs={"valinit": self._vmin},
                               callback=self._update),  #
            "VMax": SliderSpec(vmin=self._vmin, vmax=self._vmax, slider_kwargs={"valinit": self._vmax},
                               callback=self._update),  #
            "Radius": SliderSpec(vmin=0.1, vmax=200.0, slider_kwargs={"valinit": self._radius}, callback=self._update),
            #
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
        self._direction = np.array([1.0, 2.0, 10.0])
        self._direction /= np.linalg.norm(self._direction)
        self._circle = Circle((0.0, 0.0), self._radius, fill=False, edgecolor="red", linewidth=2)
        self._update_circle()
        self.ax.add_patch(self._circle)

    def _update_circle(self):
        centre = (float(self.slice_idx) - 0.5 * float(self.volume.shape[0])) * self._direction[0:2]
        self._circle.set_center((self.volume.shape[2] // 2 + centre[0], self.volume.shape[1] // 2 + centre[1]))
        self._circle.set_radius(self._sliders["Radius"].val)
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
        self._update_circle()

        self.fig.canvas.draw_idle()

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
