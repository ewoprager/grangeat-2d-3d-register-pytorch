import logging
import hashlib
from typing import Type

from reg23_experiments.data import sinogram

__all__ = ["deterministic_hash_string", "deterministic_hash_int", "deterministic_hash_type", "deterministic_hash_combo",
           "deterministic_hash_sinogram"]

logger = logging.getLogger(__name__)


def deterministic_hash_string(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()


def deterministic_hash_int(x: int) -> str:
    return hashlib.sha256(x.to_bytes(64)).hexdigest()


def deterministic_hash_type(tp: type) -> str:
    string = "{}.{}".format(tp.__module__, tp.__qualname__)
    return deterministic_hash_string(string)


def deterministic_hash_combo(*hex_digests: str) -> str:
    combined = b''.join(bytes.fromhex(h) for h in hex_digests)
    return hashlib.sha256(combined).hexdigest()


def deterministic_hash_sinogram(path: str, sinogram_type: Type[sinogram.SinogramType], sinogram_size: int,
                                downsample_factor: int) -> str:
    return deterministic_hash_combo(deterministic_hash_string(path), deterministic_hash_type(sinogram_type),
                                    deterministic_hash_int(sinogram_size), deterministic_hash_int(downsample_factor))
