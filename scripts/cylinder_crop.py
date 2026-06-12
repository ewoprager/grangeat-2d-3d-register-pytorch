import argparse
import pathlib
import pprint
from typing import Callable, NamedTuple

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider, Button
import SimpleITK as sitk
import pydicom

from reg23_experiments.data.structs import Error
from reg23_experiments.io.sitk import find_ct_series, load_ct_series, DCMSeriesInfo, save_ct_series
from reg23_experiments.utils import logs_setup
from reg23_experiments.io.command_line import get_string_required
from reg23_experiments.io.serialize import serialize_recursive


class SliderSpec(NamedTuple):
    vmin: float
    vmax: float
    callback: Callable
    slider_kwargs: dict = {}


class VolumeViewer:
    def __init__(self, input_path: pathlib.Path, output_path: pathlib.Path):
        self._input_path = input_path
        series: dict[str, DCMSeriesInfo] | Error = find_ct_series(input_path)
        if isinstance(series, Error):
            raise Exception(series)
        if not series:
            raise Exception(f"No CT series found at path '{str(input_path)}'.")
        if len(series) == 1:
            self._input_series_uid = next(iter(series))
        else:
            self._input_series_uid = get_string_required(  #
                f"Please choose one of the following CT series:\n"
                f"{"\n".join(f"{k}:\n\t{pprint.pformat(serialize_recursive(v))}\n" for k, v in series.items())}",  #
                lambda k: None if k in series else Error(f"String '{k}' does not name a series.")  #
            )
        self._input_image: sitk.Image | Error = load_ct_series(input_path, self._input_series_uid)
        if isinstance(self._input_image, Error):
            raise Exception(self._input_image)
        self._volume: np.ndarray = sitk.GetArrayFromImage(self._input_image)
        self._output_path = output_path

        slice_rad = max(self._volume.shape[1], self._volume.shape[2]) * np.max(np.array(self._input_image.GetSpacing()))

        self.fig, self.ax = plt.subplots()

        self.slice_idx = self._volume.shape[0] // 2

        self._vmin = np.min(self._volume)
        self._vmax = np.max(self._volume)
        self._x_radius = 40.0
        self._y_radius = 40.0

        slider_specs: dict[str, SliderSpec] = {  #
            "Slice": SliderSpec(vmin=0, vmax=self._volume.shape[0] - 1,
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
        spacing = self._input_image.GetSpacing()
        self.im = self.ax.imshow(self._volume[self.slice_idx], cmap="gray", origin="lower", vmin=self._vmin,
                                 vmax=self._vmax, aspect=spacing[1] / spacing[0])

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

        # Button
        button_ax = plt.axes((0.05, 0.95, 0.05, 0.05))
        self._button = Button(button_ax, "Save")
        self._button.on_clicked(self._save)

    def _set_ellipse(self, x: float, y: float, a: float, b: float):
        spacing = self._input_image.GetSpacing()
        centre = (self._volume.shape[2] // 2 + x / spacing[0], self._volume.shape[1] // 2 + y / spacing[1])
        width = a / spacing[0]
        height = b / spacing[1]
        self._ellipse.set_center(centre)
        self._ellipse.set_width(width)
        self._ellipse.set_height(height)

    def _update_ellipse(self):
        self._centre_offset = np.array([self._sliders["CentreX"].val, self._sliders["CentreY"].val])
        self._direction = np.array([self._sliders["DirX"].val, self._sliders["DirY"].val, 10.0])
        self._direction /= np.linalg.norm(self._direction)
        centre = (float(self.slice_idx) - 0.5 * float(self._volume.shape[0])) * self._direction[
            0:2] + self._centre_offset
        self._set_ellipse(centre[0], centre[1], self._sliders["XRadius"].val, self._sliders["YRadius"].val)
        self.fig.canvas.draw_idle()

    def _update(self, _):
        self.slice_idx = int(self._sliders["Slice"].val)

        self._vmin = self._sliders["VMin"].val
        self._vmax = self._sliders["VMax"].val
        self.im.set_clim(self._vmin, self._vmax)
        # Update displayed image
        self.im.set_data(self._volume[self.slice_idx])

        self.ax.set_title(f"Slice {self.slice_idx} | "
                          f"vmin={self._vmin:.2f} | "
                          f"vmax={self._vmax:.2f}")

        # Put any future processing here using vmin/vmax
        self._update_ellipse()

        self.fig.canvas.draw_idle()

    def apply_crop(self) -> np.ndarray:
        value = np.min(self._volume)
        ret = self._volume.copy()
        spacing = self._input_image.GetSpacing()
        for i in range(len(ret)):
            centre = (float(i) - 0.5 * float(self._volume.shape[0])) * self._direction[0:2] + self._centre_offset
            ys, xs = np.indices(self._volume.shape[1:3])
            ys = np.flip(ys, axis=0)
            xs = ((xs - self._volume.shape[2] // 2) * spacing[0] + centre[0]) / (0.5 * self._sliders["XRadius"].val)
            ys = ((ys - self._volume.shape[1] // 2) * spacing[1] + centre[1]) / (0.5 * self._sliders["YRadius"].val)
            mask = xs * xs + ys * ys > 1.0
            ret[i][mask] = value
        return ret

    def show(self):
        plt.show()

    def _save(self, _) -> None:
        output = self.apply_crop()

        logger.info(f"Saving volume of size {output.shape}...")

        output_image = sitk.GetImageFromArray(output)
        output_image.CopyInformation(self._input_image)
        if isinstance(err := save_ct_series(  #
                image=output_image,  #
                path=self._output_path,  #
                template_slice=pydicom.dcmread(next(iter(self._input_path.iterdir())))  #
        ), Error):
            raise Exception(err)
        logger.info(f"Output saved to '{str(self._output_path)}'.")


def main(*, input_path: pathlib.Path, output_path: pathlib.Path) -> None | Error:
    try:
        viewer = VolumeViewer(input_path, output_path)
    except Exception as e:
        return Error(f"Failed to get data from image: {str(e)}.")

    viewer.show()

    return None


if __name__ == "__main__":
    logger = logs_setup.setup_logger()

    _parser = argparse.ArgumentParser(description="", epilog="")
    _parser.add_argument("-i", "--input-path", type=str, default=None, required=True,
                         help="Give a path to a .nrrd file, .nii file or directory of DICOM files containing CT data to"
                              " modify.")
    _parser.add_argument("-o", "--output-path", type=str, default=None, required=True,
                         help="Give a path at which to save the modified CT data.")
    _args = _parser.parse_args()

    if isinstance(err := main(  #
            input_path=pathlib.Path(_args.input_path),  #
            output_path=pathlib.Path(_args.output_path)  #
    ), Error):
        logger.error(err.description)
        exit(1)
