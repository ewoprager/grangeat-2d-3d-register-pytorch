import logging
import weakref
from typing import Any, Callable

import napari.layers
import skimage
import torch

from reg23_app.context import AppContext
from reg23_app.gui.viewer_singleton import viewer
from reg23_app.param_dadg_parity_manager import ParamDADGParityManager
from reg23_experiments.data.structs import Cropping, Error, Transformation
from reg23_experiments.ops.geometry import get_crop_full_depth_drr, get_crop_nonzero_drr, project_vectors

__all__ = ["add_debug_layer"]

logger = logging.getLogger(__name__)


class _DebugLayerManager:
    def __init__(self, *, ctx: AppContext, layer: napari.layers.Image, namespace: str):
        self._ctx = ctx
        self._layer: Callable[[], napari.layers.Image | None] = weakref.ref(layer)
        self._namespace = namespace

        self._keys = [  #
            "image_2d_full",  #
            "image_2d_full_spacing",  #
            "current_transformation",  #
            "ct_spacing",  #
            "ct_volumes",  #
            "translation_offset",  #
            "source_distance",  #
        ]
        for i in range(len(self._keys)):
            if self._keys[i] in ParamDADGParityManager.XRAY_SPECIFIC_DADG_KEYS:
                self._keys[i] = f"{self._namespace}__{self._keys[i]}"
        for key in self._keys:
            self._ctx.dadg.observe(key, "debug", self._update)
            self._ctx.dadg.set_evaluation_laziness(key, lazily_evaluated=False)

        self._update()

    def __del__(self):
        for key in self._keys:
            self._ctx.dadg.set_evaluation_laziness(f"{self._namespace}__{key}", lazily_evaluated=True)

    def _update(self, _=None) -> None:
        if (layer := self._layer()) is None:
            return
        # Get DADG nodes
        nodes: dict[str, Any] = {}
        for key in self._keys:
            res: Any | Error = self._ctx.dadg.get(key)
            if isinstance(res, Error):
                logger.error(f"Failed to get '{key}': {res.description}")
                return
            nodes[key.removeprefix(f"{self._namespace}__")] = res

        image = torch.zeros_like(nodes["image_2d_full"]).cpu().numpy()
        # border
        image[:, :4] = 1.0
        image[:, -4:] = 1.0
        image[:4, :] = 1.0
        image[-4:, :] = 1.0
        # debug geometry

        if True:
            device = torch.device("cpu")
            tensor_kwargs = {"device": device, "dtype": torch.float64}
            new_translation = nodes["current_transformation"].translation.cpu() + torch.cat(
                (nodes["translation_offset"].cpu(), torch.tensor([0.0], device=device, dtype=torch.float64)))
            transformation = Transformation(rotation=nodes["current_transformation"].rotation.cpu(),
                                            translation=new_translation)

            volume_half_diag: torch.Tensor = 0.5 * torch.tensor(nodes["ct_volumes"][0].size(), **tensor_kwargs).flip(
                dims=(0,)) * nodes["ct_spacing"].cpu()
            volume_vertices = torch.tensor([  #
                [1.0, 1.0, 1.0],  #
                [-1.0, 1.0, 1.0],  #
                [1.0, -1.0, 1.0],  #
                [-1.0, -1.0, 1.0],  #
                [1.0, 1.0, -1.0],  #
                [-1.0, 1.0, -1.0],  #
                [1.0, -1.0, -1.0],  #
                [-1.0, -1.0, -1.0]  #
            ], **tensor_kwargs) * volume_half_diag  # size = (8, 3)
            projected_vertices = project_vectors(volume_vertices, source_distance=nodes["source_distance"],
                                                 transformation=transformation)  # [mm], origin centered on detector;
            # size =
            # (8, 2)
            size = nodes["image_2d_full"].size()
            points = (projected_vertices / nodes["image_2d_full_spacing"].cpu() + 0.5 * torch.tensor(  #
                size, **tensor_kwargs).flip(dims=(0,))).round().to(dtype=torch.int64)

            for point in points[:4]:
                rr, cc = skimage.draw.disk(tuple(point.flip(dims=(0,))), 12, shape=image.shape)
                image[rr, cc] = 1.0
            for point in points[4:]:
                rr, cc = skimage.draw.disk(tuple(point.flip(dims=(0,))), 12, shape=image.shape)
                image[rr, cc] = 2.0
        if True:
            cropping: Cropping = get_crop_full_depth_drr(  #
                image_2d_full=nodes["image_2d_full"],  #
                source_distance=nodes["source_distance"],  #
                current_transformation=nodes["current_transformation"],  #
                ct_volumes=nodes["ct_volumes"],  #
                ct_spacing=nodes["ct_spacing"],  #
                image_2d_full_spacing=nodes["image_2d_full_spacing"],  #
                translation_offset=nodes["translation_offset"],  #
            )
            size = nodes["image_2d_full"].size()
            right = int(round(cropping.right * float(size[1])))
            left = int(round(cropping.left * float(size[1])))
            top = int(round(cropping.top * float(size[0])))
            bottom = int(round(cropping.bottom * float(size[0])))
            thickness = 5
            for i in range(thickness):
                rr, cc = skimage.draw.rectangle_perimeter(  #
                    start=(top - i, left - i),  #
                    end=(bottom + i, right + i),  #
                    shape=image.shape,  #
                )
                image[rr, cc] = 1.0
        if True:
            cropping: Cropping = get_crop_nonzero_drr(  #
                image_2d_full=nodes["image_2d_full"],  #
                source_distance=nodes["source_distance"],  #
                current_transformation=nodes["current_transformation"],  #
                ct_volumes=nodes["ct_volumes"],  #
                ct_spacing=nodes["ct_spacing"],  #
                image_2d_full_spacing=nodes["image_2d_full_spacing"],  #
                translation_offset=nodes["translation_offset"],  #
            )
            size = nodes["image_2d_full"].size()
            right = int(round(cropping.right * float(size[1])))
            left = int(round(cropping.left * float(size[1])))
            top = int(round(cropping.top * float(size[0])))
            bottom = int(round(cropping.bottom * float(size[0])))
            thickness = 5
            for i in range(thickness):
                rr, cc = skimage.draw.rectangle_perimeter(  #
                    start=(top - i, left - i),  #
                    end=(bottom + i, right + i),  #
                    shape=image.shape,  #
                )
                image[rr, cc] = 2.0

        # updating the layer
        layer.data = image
        layer.scale = nodes["image_2d_full_spacing"].cpu().flip(dims=(0,)).numpy()


def add_debug_layer(*, ctx: AppContext, namespace: str) -> napari.layers.Image | None:
    fixed_image = ctx.dadg.get(f"{namespace}__fixed_image")
    if isinstance(fixed_image, Error):
        logger.error(f"Failed to get fixed image for '{namespace}': {fixed_image.description}")
        return None
    layer: napari.layers.Image = viewer().add_image(  #
        torch.zeros_like(fixed_image).cpu().numpy(),  #
        blending="opaque",  #
        interpolation2d="nearest",  #
        name=f"{namespace}__debug"  #
    )
    layer.my_plugin = _DebugLayerManager(ctx=ctx, layer=layer, namespace=namespace)
    return layer
