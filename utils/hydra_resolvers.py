# utils/hydra_resolvers.py
import re
import hashlib
from pathlib import Path
import urllib.parse
from omegaconf import OmegaConf

PATHY_KEYS = {
    "algorithm.teacher.ckpt_uri",
    "algorithm.teacher.ckpt_path",
    "paths.resume_from",
    "paths.ckpt_dir",
    "paths.output",
}

def _shorten_value(v: str) -> str:
    v = v.strip('"\'')

    # URI? keep scheme, host, and tail
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

    # Filesystem path?
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
    cleaned = [re.sub(r"[^0-9A-Za-z._-]+", "-", t) for t in tokens if t]
    slug = "_".join(cleaned).strip("_-.")

    MAX_LEN = 180
    if len(slug) > MAX_LEN:
        h = hashlib.md5(slug.encode()).hexdigest()[:8]
        slug = slug[: (MAX_LEN - 9)].rstrip("_-.") + "-" + h

    return slug or "default"

def register():
    # idempotent; safe to call multiple times
    OmegaConf.register_new_resolver("slug", _slugify, use_cache=False)