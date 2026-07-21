import logging
import os

os.environ["QT_API"] = "PyQt6"

from magicgui import widgets

from reg23_app.context import AppContext
from reg23_app.gui.layers.ct_fiducial_layer import add_ct_fiducial_layer
from reg23_app.gui.layers.ct_layer import add_ct_layer
from reg23_app.gui.layers.debug_layer import add_debug_layer
from reg23_app.gui.layers.electrode_layer import add_electrode_layer
from reg23_app.gui.layers.fixed_image_layer import add_fixed_image_layer
from reg23_app.gui.layers.mask_layer import add_mask_layer
from reg23_app.gui.layers.moving_image_layer import add_moving_image_layer
from reg23_app.gui.layers.projected_fiducials_layer import add_projected_fiducials_layer
from reg23_app.gui.layers.xray_fiducial_layer import add_xray_fiducial_layer

__all__ = ["ImagesWidget"]

logger = logging.getLogger(__name__)


class ImagesWidget(widgets.Container):
    def __init__(self, ctx: AppContext):
        super().__init__(labels=True)
        self._ctx = ctx

        self._ctx.state.parameters.observe(self._on_xray_parameters_change, names=["xray_parameters"])
        self._on_xray_parameters_change()

    def _on_xray_parameters_change(self, *args) -> None:
        logger.debug("X-ray parameters changed; rebuilding ImagesWidget")
        self.clear()

        if self._ctx.state.parameters.ct_path is not None:
            logger.debug("CT present; adding CT buttons to ImagesWidget")
            # Slice view
            show_ct_button = widgets.PushButton(label="Show volume")
            show_ct_button.changed.connect(lambda _: self._on_show_ct_layer())
            # Fiducial points
            show_ct_fiducials_button = widgets.PushButton(label="Show fiducials")
            show_ct_fiducials_button.changed.connect(lambda _: self._on_show_ct_fiducials_layer())
            self.append(widgets.Container(widgets=[  #
                show_ct_button,  #
                show_ct_fiducials_button,  #
            ], label="CT volume:"))

        if self._ctx.state.parameters.xray_parameters:
            logger.debug("X-rays present:")
            self.append(widgets.Label(label="X-ray images:"))
            for key, value in self._ctx.state.parameters.xray_parameters.items():
                logger.debug(f"Adding buttons for X-ray '{key}' to ImagesWidget")
                # Image 2d full
                show_image_2d_full_button = widgets.PushButton(label="Show full 2d image")
                show_image_2d_full_button.changed.connect(lambda _, name=key: self._on_show_image_2d_full_layer(name))
                # Fixed image
                show_fixed_image_button = widgets.PushButton(label="Show fixed image")
                show_fixed_image_button.changed.connect(lambda _, name=key: self._on_show_fixed_image_layer(name))
                # Moving image
                show_moving_image_button = widgets.PushButton(label="Show moving image")
                show_moving_image_button.changed.connect(lambda _, name=key: self._on_show_moving_image_layer(name))
                # Mask
                show_mask_button = widgets.PushButton(label="Show mask")
                show_mask_button.changed.connect(lambda _, name=key: self._on_show_mask_layer(name))
                # Electrode points
                show_electrodes_button = widgets.PushButton(label="Show electrodes")
                show_electrodes_button.changed.connect(lambda _, name=key: self._on_show_electrode_layer(name))
                # Fiducial points
                show_xray_fiducials_button = widgets.PushButton(label="Show fiducials")
                show_xray_fiducials_button.changed.connect(lambda _, name=key: self._on_show_xray_fiducials_layer(name))
                # Projected fiducial points
                show_projected_fiducials_button = widgets.PushButton(label="Show projected fiducials")
                show_projected_fiducials_button.changed.connect(
                    lambda _, name=key: self._on_show_projected_fiducials_layer(name))
                # Debug image
                show_debug_button = widgets.PushButton(label="Show debug image")
                show_debug_button.changed.connect(lambda _, name=key: self._on_show_debug_layer(name))
                # appending
                self.append(widgets.Container(widgets=[  #
                    show_image_2d_full_button,  #
                    show_fixed_image_button,  #
                    show_moving_image_button,  #
                    show_mask_button,  #
                    show_electrodes_button,  #
                    show_xray_fiducials_button,  #
                    show_projected_fiducials_button,  #
                    show_debug_button  #
                ], label=key))

    def _on_show_ct_layer(self) -> None:
        logger.debug(f"Show ct clicked")
        add_ct_layer(ctx=self._ctx)

    def _on_show_ct_fiducials_layer(self) -> None:
        logger.debug(f"Show ct_fiducials clicked")
        add_ct_fiducial_layer(dadg=self._ctx.dadg)

    def _on_show_image_2d_full_layer(self, xray_name: str) -> None:
        logger.debug(f"Show image_2d_full for '{xray_name}' clicked")
        add_fixed_image_layer(ctx=self._ctx, dadg_key=f"{xray_name}__image_2d_full",
                              spacing_dadg_key=f"{xray_name}__image_2d_full_spacing")

    def _on_show_fixed_image_layer(self, xray_name: str) -> None:
        logger.debug(f"Show fixed_image for '{xray_name}' clicked")
        add_fixed_image_layer(ctx=self._ctx, dadg_key=f"{xray_name}__fixed_image",
                              spacing_dadg_key=f"{xray_name}__fixed_image_spacing")

    def _on_show_moving_image_layer(self, xray_name: str) -> None:
        logger.debug(f"Show moving_image for '{xray_name}' clicked")
        add_moving_image_layer(ctx=self._ctx, namespace=xray_name, spacing_dadg_key=f"{xray_name}__fixed_image_spacing")

    def _on_show_mask_layer(self, xray_name: str) -> None:
        logger.debug(f"Show mask for '{xray_name}' clicked")
        add_mask_layer(ctx=self._ctx, namespace=xray_name, spacing_dadg_key=f"{xray_name}__fixed_image_spacing")

    def _on_show_electrode_layer(self, xray_name: str) -> None:
        logger.debug(f"Show electrodes for '{xray_name}' clicked")
        add_electrode_layer(dadg=self._ctx.dadg, namespace=xray_name)

    def _on_show_xray_fiducials_layer(self, xray_name: str) -> None:
        logger.debug(f"Show xray_fiducials for '{xray_name}' clicked")
        add_xray_fiducial_layer(dadg=self._ctx.dadg, namespace=xray_name)

    def _on_show_projected_fiducials_layer(self, xray_name: str) -> None:
        logger.debug(f"Show projected_fiducials for '{xray_name}' clicked")
        add_projected_fiducials_layer(ctx=self._ctx, namespace=xray_name)

    def _on_show_debug_layer(self, xray_name: str) -> None:
        logger.debug(f"Show debug for '{xray_name}' clicked")
        add_debug_layer(ctx=self._ctx, namespace=xray_name)
