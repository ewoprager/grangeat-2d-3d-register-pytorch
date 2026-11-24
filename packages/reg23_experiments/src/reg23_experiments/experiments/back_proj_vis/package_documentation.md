A 3D view of electrodes back-projected from X-rays, beside their reference positions in the CT.

```bash
python -m reg23_experiments.experiments.back_proj_vis -p <CT directory> -x <X-ray files...> [-n <transformation name>]
```

For each X-ray, the saved electrode positions and the saved transformation (by default `gold_standard`) are used to draw a
ray from the source through each electrode, in CT space. The rays are cut off at a sphere around the electrodes. With accurate
transformations, the rays from different X-rays meet at the reference positions.

This needs `plotly`, which is part of the `dev` extra, and opens the figure in a browser. Run it from the repository root, so
that the save data can be found.
