# 2D-3D registration gold-standard dataset for the hip joint based on uncertainty modeling

From here: https://yareta.unige.ch/archives/ed4bb43a-eaaa-4e62-8763-bd5803f5cd47

Download archive to `archive_download/` for this module to work. The following important things should exist:
- `archive_download/researchdata/code/`
- `archive_download/researchdata/data/`
- `archive_download/researchdata/data_description.json`


## Transformations

The coordinate system of this dataset is similar to ours, but with the following differences:
- the axes point in the same directions, but the $y$- and $z$-axes have flipped directions/signs
- the origin is at the source, rather that in the centre of the fixed image.

As such, conversion from their transformation to out transformation involves:
1. Multiplication of the rotation matrix and translation vector by
$$
P = \begin{bmatrix} 
1 & 0 & 0 \\
0 & -1 & 0 \\
0 & 0 & -1 \\
\end{bmatrix}
$$
2. Subtraction of the source position from the translation vector.