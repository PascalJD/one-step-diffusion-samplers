#!/usr/bin/env python3
import os
import re
from uuid import uuid4
from pathlib import Path
from datetime import datetime
import hashlib
import urllib.parse

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import jax
import wandb
import orbax.checkpoint as ocp

from utils.helper import flatten_dict, reset_device_memory
from utils.train_selector import get_train_fn

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PATHY_KEYS = {
    "algorithm.teacher.ckpt_uri",
    "algorithm.teacher.ckpt_path",
    "paths.resume_from",
    "paths.ckpt_dir",
    "paths.output",
}

DEFAULT_CHECKPOINT_MAX_TO_KEEP = 1
DEFAULT_CHECKPOINT_ASYNC_SAVE = False

def _shorten_value(v: str) -> str:
    v = v.strip('"\'')
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", v):
        p = urllib.parse.urlparse(v)
        host = (p.netloc.split(":")[0]) if p.netloc else ""
        tail = Path(p.path).name or Path(p.path).parent.name or ""
        parts = [p.scheme]
        if host:
            parts.append(host)
        if tail:
            parts.append(tail)
        short = "-".join(parts) if parts else "uri"
        return short
    if "/" in v or "\\" in v or v.startswith("~"):
        tail = Path(v).name or Path(v).parent.name or "path"
        return tail

    return v

def _normalize_override(item: str) -> str:
    item = item.strip()
    if "=" not in item:
        return item
    k, v = item.split("=", 1)
    k = k.strip()
    v = v.strip()
    if k in PATHY_KEYS or any(s in k for s in (".dir", ".path", "_dir", "_path", "ckpt", "ckpt_uri")):
        v_short = _shorten_value(v)
        if len(v_short) > 32:
            v_short = v_short[:24] + "-" + hashlib.md5(v_short.encode()).hexdigest()[:8]
        # Present a compact, readable token
        if k in ("algorithm.teacher.ckpt_uri", "algorithm.teacher.ckpt_path"):
            return f"teacherckpt-{v_short}"
        return f"{k}={v_short}"

    return f"{k}={v}"

def _slugify(overrides):
    if overrides is None:
        return "default"
    if isinstance(overrides, str):
        overrides = [o for o in overrides.split(",") if o]
    tokens = [_normalize_override(o) for o in overrides]
    # Keep only safe characters; replace others with '-'
    cleaned = [re.sub(r"[^0-9A-Za-z._-]+", "-", t) for t in tokens if t]
    slug = "_".join(cleaned).strip("_-.")
    # Hard cap length; append short hash if we truncate
    MAX_LEN = 180
    if len(slug) > MAX_LEN:
        h = hashlib.md5(slug.encode()).hexdigest()[:8]
        slug = slug[: (MAX_LEN - 9)].rstrip("_-.") + "-" + h
    return slug or uuid4().hex[:6]

OmegaConf.register_new_resolver("slug", _slugify, use_cache=False)


def _build_manager(
    root: Path,
    subdir: str,
    *,
    max_to_keep: int | None,
    async_save: bool,
) -> ocp.CheckpointManager:
    options = ocp.CheckpointManagerOptions(
        create=True,
        max_to_keep=max_to_keep,
        enable_async_checkpointing=bool(async_save),
    )
    return ocp.CheckpointManager(root / subdir, ocp.PyTreeCheckpointer(), options=options)


# Hydra entry-point
@hydra.main(version_base=None, config_path="configs", config_name="base_conf")
def main(cfg: DictConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # Readable slug
    overrides_raw = HydraConfig.get().overrides.task
    slug = _slugify(overrides_raw)

    # Instantiate everything the config describes
    cfg = hydra.utils.instantiate(cfg)  # resolves target.fn etc.
    target = cfg.target.fn
    train_fn = get_train_fn(cfg.algorithm.name)

    # Put W&B inside Hydra's run directory
    run_dir = Path(cfg.paths.output)
    os.environ["WANDB_DIR"] = str(run_dir)  # fallback for wandb.init

    if cfg.use_wandb:
        wandb.init(
            dir=run_dir,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{slug}_{datetime.now():%H-%M-%S}",
            group=cfg.algorithm.name,
            job_type=f"{cfg.target.name}_{target.dim}D",
            config=flatten_dict(OmegaConf.to_container(cfg, resolve=True)),
            resume=cfg.wandb.resume,
            tags=cfg.wandb.tags,
            mode=cfg.wandb.mode,
        )

    ckpt_root = Path(cfg.paths.resume_from) if cfg.paths.resume_from else Path(cfg.paths.ckpt_dir)

    # Assemble checkpoint manager(s)
    managers: dict[str, ocp.CheckpointManager] = {}

    use_checkpoint = bool(getattr(cfg, "save_model", True))
    if use_checkpoint:
        max_to_keep_cfg = getattr(cfg, "checkpoint_max_to_keep", DEFAULT_CHECKPOINT_MAX_TO_KEEP)
        max_to_keep = int(max_to_keep_cfg)
        async_save_cfg = getattr(cfg, "checkpoint_async_save", DEFAULT_CHECKPOINT_ASYNC_SAVE)
        async_save = bool(async_save_cfg)
        managers["last"] = _build_manager(
            ckpt_root,
            "last",
            max_to_keep=max_to_keep,
            async_save=async_save,
        )
    if managers:
        checkpointer = (
            managers if len(managers) > 1 else next(iter(managers.values()))
        )
    else:
        checkpointer = None

    try:
        if cfg.use_jit:
            train_fn(cfg, target, checkpointer)  # hand manager to trainer
        else:
            with jax.disable_jit():
                train_fn(cfg, target, checkpointer)
        if cfg.use_wandb:
            wandb.run.summary["error"] = None
            wandb.finish()
    except Exception as exc:
        if cfg.use_wandb:
            wandb.run.summary["error"] = str(exc)
            wandb.finish(exit_code=1)
        reset_device_memory()
        raise


if __name__ == "__main__":
    main()
