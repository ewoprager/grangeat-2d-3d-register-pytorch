import torch

from backend import reg23

__all__ = []

if torch.cuda.is_available():
    __all__ += ["CUDATexture2D", "CUDATexture3D"]


    class CUDATexture2D:
        """
        Wrapper class for the Python bindings for the C++ class CUDATexture2D.
        This exposes all methods of the contained object, except 'clean_up', which it does automatically
        on deletion. This is because it doesn't seem to be possible to have the underlying object's destructor called
        automatically on deletion when using via Python bindings.
        """

        def __init__(self, tensor: torch.Tensor, address_mode_x: str, address_mode_y: str):
            self._internal = reg23.CUDATexture2DInternal(tensor, address_mode_x, address_mode_y)

        def __del__(self):
            self._internal.clean_up()

        @property
        def handle(self):
            return self._internal.handle()

        @property
        def size(self):
            return self._internal.size()


    class CUDATexture3D:
        """
        Wrapper class for the Python bindings for the C++ class CUDATexture3D.
        This exposes all methods of the contained object, except 'clean_up', which it does automatically
        on deletion. This is because it doesn't seem to be possible to have the underlying object's destructor called
        automatically on deletion when using via Python bindings.
        """

        def __init__(self, tensor: torch.Tensor, address_mode_x: str, address_mode_y: str, address_mode_z: str):
            self._internal = reg23.CUDATexture3DInternal(tensor, address_mode_x, address_mode_y, address_mode_z)

        def __del__(self):
            self._internal.clean_up()

        @property
        def handle(self):
            return self._internal.handle()

        @property
        def size(self):
            return self._internal.size()
