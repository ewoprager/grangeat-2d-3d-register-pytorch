"""
.. include:: package_documentation.md
"""

from ._public import dadg_updater, args_from_dadg, data_manager
from ._data import Updater
from ._i_directed_acyclic_data_graph import IDirectedAcyclicDataGraph, IChildDirectedAcyclicDataGraph
from ._dadg_standalone import StandaloneDADG, StandaloneDADGSingleton
from ._dadg_child import ChildDADG

__all__ = ["updaters", "dadg_updater", "args_from_dadg", "data_manager", "StandaloneDADGSingleton", "Updater",
           "IDirectedAcyclicDataGraph", "IChildDirectedAcyclicDataGraph", "StandaloneDADG", "ChildDADG"]
