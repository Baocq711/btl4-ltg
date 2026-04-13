"""Checkpoint management for model persistence."""

from __future__ import annotations

import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _to_numpy(tree):
    """Convert a JAX pytree to numpy arrays for serialization."""
    return jax.tree.map(lambda x: np.array(x), tree)


def _to_jax(tree):
    """Convert numpy arrays back to JAX arrays."""
    return jax.tree.map(lambda x: jnp.array(x), tree)


def save_checkpoint(
    params,
    opt_state,
    step: int,
    path: str | Path,
) -> Path:
    """Save model checkpoint to directory.

    Creates ``path/step_XXXX/`` with params.npz, opt_state.npz, and meta.json.
    """
    path = Path(path)
    ckpt_dir = path / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Flatten and save params
    flat_params = jax.tree.leaves(params)
    param_struct = jax.tree.structure(params)
    np.savez(ckpt_dir / "params.npz", *[np.array(p) for p in flat_params])

    # Flatten and save opt_state
    flat_opt = jax.tree.leaves(opt_state)
    opt_struct = jax.tree.structure(opt_state)
    np.savez(ckpt_dir / "opt_state.npz", *[np.array(o) for o in flat_opt])

    # Save tree structures and step
    meta = {
        "step": step,
        "param_treedef": str(param_struct),
        "opt_treedef": str(opt_struct),
    }
    with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return ckpt_dir


def _load_flat_arrays(npz_path: Path) -> list[np.ndarray]:
    data = np.load(npz_path)
    return [data[k] for k in sorted(data.files, key=lambda s: int(s.replace("arr_", "")))]


def load_checkpoint(
    path: str | Path,
    param_template,
    opt_template=None,
):
    """Load checkpoint from a step directory.

    Args:
        path: Path to ``step_XXXX/`` directory.
        param_template: A pytree with same structure as saved params (for unflattening).
        opt_template: Optional pytree for optimizer state.

    Returns:
        (params, opt_state_or_None, step)
    """
    path = Path(path)
    with open(path / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    step = meta["step"]

    # Restore params
    flat_params = _load_flat_arrays(path / "params.npz")
    param_struct = jax.tree.structure(param_template)
    params = jax.tree.unflatten(param_struct, [jnp.array(a) for a in flat_params])

    # Restore opt_state if template provided
    opt_state = None
    if opt_template is not None and (path / "opt_state.npz").exists():
        flat_opt = _load_flat_arrays(path / "opt_state.npz")
        opt_struct = jax.tree.structure(opt_template)
        opt_state = jax.tree.unflatten(opt_struct, [jnp.array(a) for a in flat_opt])

    return params, opt_state, step


def list_checkpoints(directory: str | Path) -> list[Path]:
    """List checkpoint directories sorted by step number."""
    directory = Path(directory)
    if not directory.exists():
        return []
    pattern = re.compile(r"step_(\d+)")
    ckpts = []
    for child in directory.iterdir():
        if child.is_dir():
            m = pattern.match(child.name)
            if m:
                ckpts.append((int(m.group(1)), child))
    ckpts.sort(key=lambda t: t[0])
    return [c[1] for c in ckpts]


def load_latest(
    directory: str | Path,
    param_template,
    opt_template=None,
):
    """Load the latest checkpoint from a directory, or return None."""
    ckpts = list_checkpoints(directory)
    if not ckpts:
        return None
    return load_checkpoint(ckpts[-1], param_template, opt_template)
