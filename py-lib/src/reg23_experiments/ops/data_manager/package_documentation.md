This package provides a suite of modular tools built around a directed acyclic data graph (`DADG`) data structure for
managing data.

## Getting started

The `data_manager()` is a singleton instance of `StandaloneDADG` used to keep track of data and which data depends on
which other data. Here is an example to illustrate its purpose:

### Motivation behind the directed acyclic data graph

Let's say you have some data read from files at the start, like a CT scan and an X-ray image. Dependent on these you
have a long pipeline of images and data which depend on each other and these images, for example involving cropping,
masking, downsampling, projecting etc.... The `IDirectedAcyclicDataGraph` allows you to only define the dependencies
between the images, and will automatically perform the operations that evaluate the images when the dependencies change.

#### A simple example

You have a CT scan volume loaded from a file, and want to use it at various downsampling ratios. You could just store
the raw CT volume, and perform a downsampling whenever you want to access it, like so:

```Python
import torch

ct_volume: torch.Tensor = load_from_file("/my_ct_volume.nrrd")


def downsample_ct_volume(vol: torch.Tensor, ratio: int) -> torch.Tensor:
    return torch.nn.functional.avg_pool3d(vol, ratio)


...

use_ct_volume(downsample_ct_volume(ct_volume, 3))
...
use_ct_volume_again(downsample_ct_volume(ct_volume, 3))
...
use_ct_volume_again(downsample_ct_volume(ct_volume, 2))
```

The issue with this is that `downsample_ct_volume` could be quite expensive for a large volume, so a lot of time would
wasted if the same downsampling ratio was used multiple times in a row. This also becomes much more ugly and even less
efficient if there are more layers of dependency, e.g. if the `ct_volume` itself depended on something else (like if the
path from which the volume was loaded might change), or if another value was dependent on the downsampled volume.

Caching of values using `functools` would solve the former of these two issues, so long as the input values were not
floating-point, which would be quite a restriction.

To partially solve the former issue, and completely solve the latter, we use the `DirectedAcyclicDataGraph` as follows:

```Python
from typing import Any
import torch
from reg23_experiments.ops.data_manager import data_manager, dadg_updater

data_manager().set_multiple(ct_path="/my_ct_volume.nrrd", downsample_ratio=3)


@dadg_updater(names_returned=["ct_volume"])
def load_ct_volume(ct_path: str) -> dict[str, Any]:
    return {"ct_volume": load_from_file(ct_path)}


@dadg_updater(names_returned=["ct_volume_downsampled"])
def downsample_ct(ct_volume: torch.Tensor, downsample_ratio: int) -> dict[str, Any]:
    return {"ct_volume_downsampled": torch.nn.functional.avg_pool3d(ct_volume, downsample_ratio)}


data_manager().add_updater("load_ct_volume", load_ct_volume)
data_manager().add_updater("downsample_ct", downsample_ct)

...

use_ct_volume(data_manager().get("ct_volume_downsampled"))
...
use_ct_volume_again(data_manager().get("ct_volume_downsampled"))
...
data_manager().set("downsample_ratio", 3)
use_ct_volume_again(data_manager().get("ct_volume_downsampled"))
```

A walk through the above code:

- The first call to `ata_manager()` constructs the singleton instance of the `StandaloneDADG`.
- We set some values in the data manager using `data_manager().set_multiple(...)`. This just interprets each
  keyword argument as a new (name, value) pair to insert into the graph.
- We define two instances of the `Updater` class using the `@dadg_updater` decorator, which transforms the decorated
  function into an instance of `Updater`. This is an object which is used to define dependencies between nodes in the
  data manager graph: the arguments of the function define the variables that are depended upon, and the (key, value)
  pairs in the returned dictionary define the variables that are dependent on them. The function itself defines the
  mapping from the 'depended' to the 'dependent'.
- We insert these `Updater`s into the data manager using `data_manager().add_updater(...)`. The strings passed here are
  just arbitrary names, which we have chosen to match the updater names. Updaters can also be removed from the data,
  and this is done by the string name passed here.
- Then, to obtain the downsample CT image, we need only call `data_manager().get("ct_volume_downsampled")`. This will
  retrieve a previously calculated value if it is consistent with the dependencies, or run the appropriate `Updater`s to
  re-evaluate the variable if the dependencies have changed.
- The first call to `data_manager().get("ct_volume_downsampled")` in the code above will result in both `load_ct_volume`
  and `downsample_ct` being called for the first time to evaluate `"ct_volume"` and then `"ct_volume_downsampled"`.
- The second call will not perform any evaluation, as the dependencies `"ct_path"` and `"downsample_ratio"` have not
  changed since the value of `"ct_volume_downsampled"` was last evaluated.
- The third call will result in `downsample_ct` being called once more, as the value of `"downsample_ratio"` has changed
  since `"ct_volume_downsampled"` was last evaluated.