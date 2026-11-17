# Coil Forces

Self and inter-coil forces, magnetic fields, and inductance for stellarator
coils of rectangular cross-section, using the 1D filament model of Landreman,
Hurwitz & Antonsen. Built around the HSX coil set, for studying coil
displacement.

## Requirements

```
pip install "numpy>=2.0.0" "scipy>=1.15.0" "jax>=0.9.0" "jaxlib>=0.9.0" "matplotlib>=3.9.0"
```

## Running

Run any script from the repository root:

```
python fig_reproduce.py
```

Output folders (`results/` and `results_txt/`) are created automatically.
Coil index, point count, current, and cross-section are set at the top of each
script's `main()`.

## Scripts

| File | Description | Output |
| --- | --- | --- |
| `hsx_utilities.py` | Shared library: geometry loading, the {t,p,q} frame, regularized field, force, inductance, plotting helpers | — |
| `fig_reproduce.py` | Reproduces Figures 1, 6, 7 and 8 of the paper | `results/Fig{1,6,7,8}_0.png` |
| `tpq_frame.py` | Self-force along the coil in the {t,p,q} frame | `results_txt/hsx_forces_tpq_256_0.txt` |
| `tpq_full.py` | Same, including the other 47 coils | `results_txt/hsx_forces_tpq_TOTAL_256_0.txt` |
| `SelfInterForce.py` | Self and inter-coil force in the p and q directions, plotted together | `results/Full_TPQ_Forces_0.png` |
| `plot_tpq_8_regions.py` | Length-averaged forces over eight equal arc-length regions, plus zero-bounded peak F_q regions | `results/TPQ_8_region_forces_coil_5.png`, two tables in `results_txt/` |
| `animate_B.py` | Animates the internal magnetic field around the coil length | `results/internal_field_5.gif` |
| `animate_B_full.py` | Same, including contributions from the other coils | `results/internal_field_TOTAL_0.gif` |
| `animate_forces.py` | Animates the forces around the coil length | `results/coil0.gif` |
| `animate_forces_full.py` | Same, including contributions from the other coils | `results/coil0_total_force.gif` |

## Data

| File | Description |
| --- | --- |
| `HSX_coil_geometry.txt` | Multi-turn HSX geometry; the 48 full-length 14-turn modular coils are used, the shorter auxiliary coils are skipped |
| `coils.hsx.original` | Single-filament FOCUS/makegrid format, read by `read_coils_file` |

## Defaults

150 kA total current (per-turn current x 14 turns) through a 0.13 x 0.06 m
rectangular cross-section. `plot_tpq_8_regions.py` uses 190050 A instead.
`animate_B.py` targets coil index 5 and `plot_tpq_8_regions.py` coil index 4
(its filenames count from 1); the rest target coil index 0.

## Reference

Efficient calculation of self magnetic field, self-force, and self-inductance
for electromagnetic coils. II. Rectangular cross-section — Landreman, Hurwitz &
Antonsen, [arXiv:2310.12087](https://arxiv.org/abs/2310.12087).

With advice from Mark Nornberg and Benedikt Geiger at HSX, UW-Madison.
