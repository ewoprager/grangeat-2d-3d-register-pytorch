# Development

Setup is described in the [project README](../README.md#setup). This page covers the machinery around the code.

## Workspace

The repository is a `uv` workspace. The root `pyproject.toml` describes the environment for the scripts, and each directory in
`packages/` has its own. The `cpu` and `cuda` extras choose the PyTorch build and cannot be combined. The `dev` extra adds the
tools for tests and plotting.

`uv sync` is run twice, first with `--no-install-workspace`, because building the extension needs PyTorch. The
[justfile](../justfile) has recipes for this (`just build`, `just devbuild`), and for the truncation experiment.

## Building the extension

`packages/reg23_core/setup.py` collects the sources and chooses between a CPU and a CUDA build:

| Control                  | Effect                                                                  |
|--------------------------|-------------------------------------------------------------------------|
| (default)                | Builds with CUDA if a GPU and a CUDA toolkit are found                  |
| `FORCE_CUDA=1`           | Builds with CUDA regardless                                             |
| `--no-cuda`              | Builds for the CPU only                                                 |
| `--debug`                | No optimisation, with debug symbols and logging of texture lifetimes    |

With a GPU present, code is generated only for that GPU's compute capability. `CMakeLists.txt` exists for IDEs and is not used in
the build. Changes to the C++ and CUDA sources trigger a rebuild on the next `uv run` or `uv sync`.

## Tests

```bash
python -m pytest packages
```

See the READMEs for the [core](../packages/reg23_core/tests/README.md) and
[experiments](../packages/reg23_experiments/tests/README.md) tests. CUDA code is tested only on a machine with a GPU.

## Continuous integration

| Workflow                                                | Runs on                        | Does                                                     |
|---------------------------------------------------------|--------------------------------|----------------------------------------------------------|
| [build_test.yml](../.github/workflows/build_test.yml)   | pushes and pull requests to main | Creates a fresh lock file, builds for the CPU, runs tests |
| [docs.yml](../.github/workflows/docs.yml)               | pushes to main                 | Builds both sets of documentation, deploys to GitHub Pages |

## Documentation

| Where                                              | How it is published                                                                      |
|----------------------------------------------------|------------------------------------------------------------------------------------------|
| `packages/reg23_experiments/**/package_documentation.md` | `pdoc`, as the docstring of the package whose `__init__.py` includes it            |
| Markdown in `packages/reg23_core/src`, `mainpage.md`, `Conventions.md` | Doxygen, as pages                                                    |
| Comments in the C++ and CUDA sources               | Doxygen                                                                                  |
| `README.md` files and this directory               | Not published; read on GitHub                                                            |

To document a new subpackage, add a `package_documentation.md` and start its `__init__.py` with:

```python
"""
.. include:: package_documentation.md
"""
```

Build locally with:

```bash
cd packages/reg23_core/Docs && doxygen Doxyfile
```

```bash
cd packages/reg23_experiments && uv run --extra cpu --group docs pdoc src/reg23_experiments
```

The data graph diagrams use Mermaid, which GitHub renders. `pdoc` needs the `--mermaid` option to render them.

## Docker

`docker-compose.yml` defines CPU and CUDA versions of the app, which display through X11. The CPU image uses a build stage with
`uv` and a runtime stage with the Qt libraries. As the project README says, the images do not work yet. The CUDA file predates
the current package layout.

## Formatting

C++ and CUDA code follows `.clang-format`. Python imports are sorted with `isort`, configured in
`packages/reg23_core/pyproject.toml`. Naming and layout rules are in the two `Conventions.md` files.
