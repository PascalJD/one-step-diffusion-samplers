# One-Step Diffusion Samplers

This repository contains the code for the paper *One-Step Diffusion Samplers via Self-Distillation and Deterministic Flow* (OSDS).

At a high level, OSDS:
- learns a step-conditioned ODE using state-space self‑distillation, and
- calibrates density change using volume consistency,
- enabling fast sampling and robust ELBO estimates in the one/few‑step regime via a deterministic flow map.


## Environment

We provide a pinned conda environment with JAX/Flax/Optax.

```bash
conda env create -f environment.yml
conda activate osds
```

## Quick start 
Trains and evaluates OSDS end‑to‑end.

```bash
python run.py algorithm=osds target=funnel
```

## Citation

If you use this code, please cite our paper:
```
@misc{jutrasdube2025osds,
  title={One-Step Diffusion Samplers via Self-Distillation and Deterministic Flow},
  author={Pascal Jutras-Dub{\'e} and Jiaru Zhang and Ziran Wang and Ruqi Zhang},
  year={2025}
}
```

Please also cite the following work from which we based our code on:
```
@inproceedings{blessing2024elbos,
  title={{Beyond ELBOs: A Large-Scale Evaluation of Variational Methods for Sampling}}, 
  author={Denis Blessing and Xiaogang Jia and Johannes Esslinger and Francisco Vargas and Gerhard Neumann},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```