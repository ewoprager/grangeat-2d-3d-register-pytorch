"""
.. include:: package_documentation.md
"""

from ._public import dadg_updater, args_from_dadg, data_manager
from ._data import Updater
from ._directed_acyclic_data_graph import DirectedAcyclicDataGraph, ChildDirectedAcyclicDataGraph
from ._dadg_standalone import StandaloneDADG, StandaloneDADGSingleton
from ._dadg_child import ChildDADG

__all__ = ["updaters", "dadg_updater", "args_from_dadg", "data_manager", "StandaloneDADGSingleton", "Updater",
           "DirectedAcyclicDataGraph", "ChildDirectedAcyclicDataGraph", "StandaloneDADG", "ChildDADG"]
