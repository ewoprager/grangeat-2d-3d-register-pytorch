# A PyTorch extension for fast 2D/3D radiographic image registration using Grangeat's relation

### Documentation

Documentation for the python libraries in this repo, as well as for the `reg23` library, can be found here:

[https://ewoprager.github.io/grangeat-2d-3d-register-pytorch/](https://ewoprager.github.io/grangeat-2d-3d-register-pytorch/)

The registration method using Grangeat's relation is from this paper:

Frysch R, Pfeiffer T, Rose G. A novel approach to 2D/3D registration of X-ray images using Grangeat's relation. Med
Image Anal. 2021 Jan;67:101815. doi: 10.1016/j.media.2020.101815. Epub 2020 Sep 30. PMID: 33065470.

This project is very much in the experimental stages, so none of the code is very usable yet. It is being actively
developed as part of a PhD.

# Repo contents

ToDo: update this

```text
.
├── reg23/                       # A Python package with custom C++/CUDA operators for PyTorch; see the README.md inside for more information.
├── py-lib/                      # A python library with tools for research and experiments, used by the scripts in the scripts/ directory.
│   ├── src/reg23_experiments/   # The library source code.
│   ├── README.md                # Some more detail about this library can be found here.
│   ├── Conventions.md           # Contains details of Python coding conventions, regarding style and structure.
│   └── pyproject.toml           # The project configuration file used by `uv` to setup the environment and dependencies of the reg23_experiments library
├── scripts/                     # Some python scripts that use the `reg23_experiments` module to perform experiments.
│   ├── benchmarking/            # Scripts specifically for measuring the speeds of different implementations of algorithms in the `reg23` package
│   ├── plotting/                # Scripts used for plotting data that is generated and saved by other scripts.
│   ├── app.py                   # The main script; see more details below.
│   └── ...                      # Other scripts, many of which may be somewhat out-of-date.
├── data/                        # Directory used for saving experimental data, or saved values between script runs.
│   ├── app_electrode_save_data/ # Segmented positions in X-rays in the app are saved here.
│   └── app_transformation_s.../ # Registered transformations between X-rays and CTs in the app are saved here.
├── figures/                     # Plots and images from experiments.
├── README.md                    # This file.
├── pyproject.toml               # The project configuration file used by `uv` to setup the environment and dependencies used by all Python scripts.
├── uv.lock                      # A file managed by `uv` which saves the exact installed dependency versions to install.
└── logging.conf                 # A config file used for logging with the `logging` standard Python package.
```

# Setup

The (much slower) CPU implementations should work on all platforms.

CUDA is required for the more expensive functionality to run quickly. NVCC version information:

```bash
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
```

The build can be done faster with Ninja installed. For Clion users: note that `setuptools` won't find the Ninja
installation within Clion.

On Ubuntu this would be:

```bash
sudo apt install ninja-build
```

Note: instructions here may be slightly outdated and not work on all platforms.
The [build_test.yml](.github/workflows/build_test.yml) GitHub
workflow can generally be relied upon to be up-to-date,
and to work.

### [`uv`](https://docs.astral.sh/uv/) is required

Initialise the virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install all dependencies with:

```bash
uv sync --extra cpu
```

or

```bash
uv sync --extra cuda
```

## Docker

The Dockerfiles do not work yet.

Docker images are built for linux/amd64.
Linux ARM64 is not supported due to upstream binary dependencies (e.g. triangle).
On Apple Silicon Macs, run the app natively for best results.

cmd: `docker compose --profile cpu up --build`

Docker dev notes:

- I ran `xhost +local:docker`
- I ran `xhost +SI:localuser:$(whoami)`
- I ran `export USER_UID=$(id -u)` and `export USER_GID=$(id -g)`
- I ran the following to allow me to run docker without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

# Scripts you can run

## The app

The main script is an interface based on `napari`:

```bash
uv run scripts/app.py
```

Most of the implementation of the app is contained in the `py-lib` library in
the [app](py-lib/src/reg23_experiments/app) directory.

The basic layout of the app is as follows:

- The state of the app is stored in a single `traitlets.HasTraits` struct called `AppState`, implemented
  in [state.py](py-lib/src/reg23_experiments/app/state.py).
    - Within this is an instance of a further `traitlets.HasTraits` struct called `Parameters`, implemented
      in [parameters.py](py-lib/src/reg23_experiments/experiments/parameters.py).
    - This `Parameters` struct contains all experimental/registration configuration for the app.
    - A serialised copy of this data is maintained eagerly in a cache file (at `platformdirs.user_cache_dir`).
    - When the app is started, it deserializes this value to restore the `Parameters` config.
    - The parameters can be configured in the interface within the `Params` tab on the right.
        - This widget is implemented
          in [parameters_widget.py](py-lib/src/reg23_experiments/app/gui/widgets/parameters_widget.py).
        - The parameter configuration widget within is auto-generated via reflection from the `Parameters` class. This
          is implemented in [hastraits_widget.py](py-lib/src/reg23_experiments/app/gui/widgets/hastraits_widget.py).
- Additionally within the *'Params'* tab are buttons to open X-ray images and CT volumes.
    - Only one CT volume may be loaded at a time; opening a CT volume while one is already loaded will replace the
      loaded volume.
    - Multiple X-ray images may be loaded simultaneously; opening an X-ray image will not affect previously-loaded
      images.
- The image data and derived images may be shown using the *'Images'* tab on the right.
    - This widget is implemented is [images_widget.py](py-lib/src/reg23_experiments/app/gui/widgets/images_widget.py).
    - The interface currently provides no way of rendering a CT volume other than as a DRR. As X-ray projection
      geometry is required to parametrise the projection of a DRR, DRRs can only be viewed with a loaded X-ray, and
      are available one per X-ray.
    - This widget contains a repeated set of buttons for each X-ray that is loaded.
    - Each button adds a different image layer to the `napari` image viewer.
    - Once added the layers will appear in the layers interface on the left, and the rendering parameters can be
      configured there.
    - The behaviours of the images in these layers are implemented in objects that are stored as 'plugins' to the
      `napari` layer objects. Their implementations are in the [layers](py-lib/src/reg23_experiments/app/gui/layers)
      directory.
- Registration of a loaded CT volume to a loaded X-ray image may be done via the *'Register'* tab on the right.
    - This widget is implemented
      is [register_widget.py](py-lib/src/reg23_experiments/app/gui/widgets/register_widget.py).
    - The transformation parameters are shown in two forms:
        - T: a 3-component rotation with units of Radians, and a 3-component translation with units of mm.
        - x: a 6-component vector comprised of scaled versions of the 6 values above. The scaling coefficients are
          chosen by hand to make the optimisation landscape more symmetrical.
    - **The transformation parameters may be changed by changing the values in the above widgets, or by selecting a '
      Moving Image' layer on the left (can be opened from the *'Images'* tab) and Left- or Right-click-dragging in the
      image viewer with the Ctrl key held down.**
    - Transformations for the selected X-ray can be saved and loaded. These are stored in a table in
      the [app_transformation_save_data](data/app_transformation_save_data) directory.
    - Objective function evaluation and registrations can be performed in this tab, but note that registration and
      similarity measure configuration is stored in the `Parameters` in the app state, and so it configured in the *'
      Params'* tab.
- Points can be segmented in X-ray images, for example the electrodes of a cochlear implant:
    - Click the *'Show electrodes'* button for an X-ray in the *'Images'* tab. This will add a `napari` 'Points' layer
      to the layer list on the left.
    - Click the *'Show full 2d image'* button for the same X-ray in the *'Images'* tab. This will add an 'Image' layer
      showing the full resolution X-ray image.
    - With the *'<X-ray name>__electrode_points'* layer selected in the layer list, points can be added by clicking the
      *'Add points'* button, which appears as a '+' inside a circle, at the top of the *'layer controls'* panel, and
      then clicking in the image viewer.
    - The positions of the points are in the coordinate system of the full resolution image, so when segmenting features
      in the X-ray, it is important to be using the 'full 2d image' layer not the 'fixed image' layer, which may be
      scaled differently.
    - The set of segmented points for the X-ray are saved eagerly to a table in
      the [app_electrode_save_data](data/app_electrode_save_data) directory, and automatically loaded on start.

### How are the images processed? Why does it lag when I show the moving image, but not when I open the CT file?

All image processing is done within a directed acyclic graph framework, which is implemented in
the [data_manager](py-lib/src/reg23_experiments/ops/data_manager) directory.

The main ideas are as follows:

- All parameters that are used in loading and processing the images are stored as nodes in a graph. For example these
  might be:
    - The file path of an X-ray DICOM image,
    - The file path of a CT volume,
    - The transformation at which to project a moving image,
    - Whether or not the X-ray image should be flipped, etc...
- Note that the above examples are not derived from other data, they are set by the user.
- Further nodes are added to the graph that **are** derived from other data, along with the appropriate mappings between
  the nodes. For example these might be:
    - The X-ray image itself, which is mapped from the X-ray DICOM file path simply by a function that finds the file
      and loads the contents into `torch.Tensor`,
    - The CT volume itself, which is similarly mapped from the file path to the CT volume file or directory,
    - The moving image, which is a projection through the CT volume according to the projection parameters given in the
      metadata of the X-ray DICOM, etc...
- As you can see, many mappings may be expensive to compute, so the number of times they must be run is minimised by
  only evaluating nodes **lazily**.
- When using the [data_manager](py-lib/src/reg23_experiments/ops/data_manager) in a script that doesn't contain an
  interface, all nodes can be evaluated lazily.
- When rendering an interface, the user will want to see images update live as parameters change, so such nodes are set
  to evaluate eagerly in this case.
- When opening a CT file/directory, with no related image layer on display, no nodes get updated, only a CT path is
  loaded into one.
- Opening an X-ray file will similarly not update further nodes.
- If the moving image is then shown, this node will be evaluated, which will trigger loading of the X-ray file for its
  metadata, loading of the whole CT volume, and then projection of the moving image through the CT volume, which will
  cause a small amount of lag.
- Mappings are created as functions decorated with `dadg_updater`, which can be found in [data_manager/__init
  __.py](py-lib/src/reg23_experiments/ops/data_manager/__init__.py). The mapping implementations themselves can be found
  in [updaters.py](py-lib/src/reg23_experiments/experiments/updaters.py)
  and [multi_xray_truncation_updaters.py](py-lib/src/reg23_experiments/experiments/multi_xray_truncation_updaters.py).

## Other scripts

All scripts are contained in the `scripts/` directory, and can be run with

```bash
uv run <script name> <args...>
```

To run any script directly with the python binary:

```bash
python <script name> <args...>
```

This is useful if you want to run with a debugger attached (e.g. if you have this as a run configuration in an IDE),
but note that this will not check for correctly install packages, nor initialise the build of the extension if the
source code has changed, as `uv` is not run here, so make sure to run `uv sync` beforehand if you have changed any
dependencies or the extension.

# The `reg23` PyTorch extension

The extension is contained within the [reg23](reg23) directory, with its own [README.md](reg23/README.md).

It is used by the app and other scripts via the `py-lib` library. In particular, `project-drr` is very commonly used.

## Other scripts (not maintained)

### Run Radon transform algorithms on CPU and GPU (CUDA) to compare performance:

```bash
uv run scripts/benchmaking/benchmark_radon2d.py "/path/to/x_ray.dcm"
uv run scripts/benchmaking/benchmark_radon3d.py "/path/to/ct.nrrd"
```

### Run the Grangeat-based resampling algorithms on CPU and GPU (CUDA) to compare performance:

```bash
uv run scripts/benchmaking/benchmark_resample_sinogram3d.py -h
uv run scripts/benchmaking/benchmark_resample_sinogram3d.py --no-load --no-save --sinogram-size 64 # run on synthetic data
uv run scripts/benchmaking/benchmark_resample_sinogram3d.py --ct-nrrd-path "/path/to/ct.nrrd"
```

### Run registration experiments:

```bash
uv run scripts/register.py -h
uv run scripts/register.py --no-load --no-save --sinogram-size 64 # run on synthetic data
uv run scripts/register.py --ct-nrrd-path "/path/to/ct.nrrd"
```

### Dev scripts

```bash
uv run registration/lib/dev_scripts/dev_sinogram.py --help 
```

or

```bash
python registration/lib/dev_scripts/dev_sinogram.py --help 
```

## IDE integration

All the following IDE integration advice is based on CLion 2024.3.1.1.

`reg23/CMakeLists.txt` exists exclusively to aid your IDE with syntax highlighting and error detection in the
extension .cpp and .cu source files. Configure a CMake project in your IDE to make use of this.

To help CLion follow the imports in python, right click on the following directories and select 'Mark Directory As' > '
Project Sources and Headers':

- `py-lib/src`
- `reg23/src`