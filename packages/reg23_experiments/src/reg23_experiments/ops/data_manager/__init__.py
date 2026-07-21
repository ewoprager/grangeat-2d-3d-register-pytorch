"""
.. include:: package_documentation.md
"""

from ._dadg_child import ChildDADG
from ._dadg_standalone import StandaloneDADG, StandaloneDADGSingleton
from ._data import NoNodeData, Updater
from ._directed_acyclic_data_graph import (ChildDirectedAcyclicDataGraph,
                                           DirectedAcyclicDataGraph)
from ._namespaces import capture_in_namespaces
from ._public import args_from_dadg, dadg_updater, data_manager

__all__ = ["dadg_updater", "args_from_dadg", "data_manager", "StandaloneDADGSingleton", "Updater",
           "DirectedAcyclicDataGraph", "ChildDirectedAcyclicDataGraph", "StandaloneDADG", "ChildDADG",
           "capture_in_namespaces", "NoNodeData"]
