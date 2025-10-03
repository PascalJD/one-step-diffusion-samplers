# One-Step Diffusion Samplers

This repo contains code for Self‑Distilled Single‑Step Samplers (SDSS): a step‑conditioned sampler that compresses many small probability‑flow ODE updates into one (or a few) large steps, and computes ELBO via a deterministic flow.

---

## 1) Environment

We provide a pinned conda environment with JAX/Flax/Optax.

```bash
conda env create -f environment.yml
conda activate sdss
```

## 2) Quick start 
Trains and evaluates SDSS end‑to‑end.

```bash
python run.py algorithm=sdss_vp target=funnel
```