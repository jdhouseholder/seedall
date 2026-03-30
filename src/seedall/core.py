"""
Core seeding logic for seedall.

Supports: random, numpy, torch (CPU + CUDA), tensorflow, JAX, cupy.
Each backend is optional -- missing libraries are silently skipped.
"""

from __future__ import annotations

import os
import random
import logging
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

@dataclass
class Backend:
    """Describes one RNG backend (e.g. numpy, torch)."""
    name: str
    seed_fn: Any          # callable(seed: int) -> None
    get_state_fn: Any     # callable() -> state
    set_state_fn: Any     # callable(state) -> None
    available: bool = True
    extras: Dict[str, Any] = field(default_factory=dict)


_BACKENDS: Dict[str, Backend] = {}


def _register_builtin_backends() -> None:
    """Detect and register all supported RNG backends."""

    # 1. Python stdlib random
    _BACKENDS["random"] = Backend(
        name="random",
        seed_fn=random.seed,
        get_state_fn=random.getstate,
        set_state_fn=random.setstate,
    )

    # 2. os.environ PYTHONHASHSEED (best-effort)
    def _hashseed_set_state(st: Any) -> None:
        if st is not None:
            os.environ["PYTHONHASHSEED"] = str(st)
        else:
            os.environ.pop("PYTHONHASHSEED", None)

    _BACKENDS["hashseed"] = Backend(
        name="hashseed",
        seed_fn=lambda s: os.environ.__setitem__("PYTHONHASHSEED", str(s)),
        get_state_fn=lambda: os.environ.get("PYTHONHASHSEED"),
        set_state_fn=_hashseed_set_state,
    )

    # 3. NumPy
    try:
        import numpy as np

        _BACKENDS["numpy"] = Backend(
            name="numpy",
            seed_fn=lambda s: np.random.seed(s),
            get_state_fn=np.random.get_state,
            set_state_fn=np.random.set_state,
        )
    except ImportError:
        logger.debug("numpy not found -- skipping")

    # 4. PyTorch
    try:
        import torch

        def _torch_seed(s: int) -> None:
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(s)
                torch.cuda.manual_seed_all(s)

        def _torch_get_state() -> dict:
            state = {"cpu": torch.random.get_rng_state()}
            if torch.cuda.is_available():
                state["cuda"] = [
                    torch.cuda.get_rng_state(i)
                    for i in range(torch.cuda.device_count())
                ]
            return state

        def _torch_set_state(state: dict) -> None:
            torch.random.set_rng_state(state["cpu"])
            if "cuda" in state and torch.cuda.is_available():
                for i, s in enumerate(state["cuda"]):
                    torch.cuda.set_rng_state(s, i)

        _BACKENDS["torch"] = Backend(
            name="torch",
            seed_fn=_torch_seed,
            get_state_fn=_torch_get_state,
            set_state_fn=_torch_set_state,
        )
    except ImportError:
        logger.debug("torch not found -- skipping")

    # 5. TensorFlow
    # NOTE: TensorFlow does not expose a global RNG get/set state API.
    # Seeding works, but get_states()/set_states() are no-ops for this backend.
    try:
        import tensorflow as tf

        _BACKENDS["tensorflow"] = Backend(
            name="tensorflow",
            seed_fn=lambda s: tf.random.set_seed(s),
            get_state_fn=lambda: None,
            set_state_fn=lambda st: None,
        )
    except ImportError:
        logger.debug("tensorflow not found -- skipping")

    # 6. JAX
    try:
        import jax

        # JAX uses explicit PRNG keys rather than global state, so we store
        # a "default key" that users can retrieve via seedall.get_states().
        _jax_state: Dict[str, Any] = {"key": None}

        def _jax_seed(s: int) -> None:
            _jax_state["key"] = jax.random.PRNGKey(s)

        _BACKENDS["jax"] = Backend(
            name="jax",
            seed_fn=_jax_seed,
            get_state_fn=lambda: _jax_state.copy(),
            set_state_fn=lambda st: _jax_state.update(st),
            extras={"get_key": lambda: _jax_state["key"]},
        )
    except ImportError:
        logger.debug("jax not found -- skipping")

    # 7. CuPy
    try:
        import cupy as cp

        _BACKENDS["cupy"] = Backend(
            name="cupy",
            seed_fn=lambda s: cp.random.seed(s),
            get_state_fn=cp.random.get_random_state,
            set_state_fn=lambda st: cp.random.set_random_state(st),
        )
    except ImportError:
        logger.debug("cupy not found -- skipping")


# Run registration on import
_register_builtin_backends()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def seed(
    value: int,
    *,
    backends: Optional[List[str]] = None,
    deterministic: bool = False,
    warn_missing: bool = False,
) -> Dict[str, bool]:
    """
    Seed all (or selected) RNG backends for reproducibility.

    Parameters
    ----------
    value : int
        The seed value (must be a non-negative integer).
    backends : list[str], optional
        Subset of backend names to seed. ``None`` means all available.
    deterministic : bool
        If True, also enable PyTorch deterministic mode and disable
        cudnn benchmarking for maximum reproducibility (at a speed cost).
    warn_missing : bool
        If True, emit a warning for each requested backend that is not
        installed.

    Returns
    -------
    dict[str, bool]
        Mapping of backend name -> whether it was successfully seeded.

    Raises
    ------
    TypeError
        If *value* is not an integer.
    ValueError
        If *value* is negative.
    """
    if not isinstance(value, int):
        raise TypeError(f"seed value must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"seed value must be non-negative, got {value}")

    results: Dict[str, bool] = {}
    targets = backends or list(_BACKENDS.keys())

    with _lock:
        for name in targets:
            if name not in _BACKENDS:
                if warn_missing:
                    warnings.warn(f"seedall: backend '{name}' is not available")
                results[name] = False
                continue

            backend = _BACKENDS[name]
            try:
                backend.seed_fn(value)
                results[name] = True
                logger.info("Seeded %s with %d", name, value)
            except Exception as exc:
                logger.warning("Failed to seed %s: %s", name, exc)
                results[name] = False

    # PyTorch deterministic extras
    if deterministic:
        _set_torch_deterministic(True)

    return results


def available() -> List[str]:
    """Return the names of all detected RNG backends."""
    return list(_BACKENDS.keys())


def get_states(backends: Optional[List[str]] = None) -> Dict[str, Any]:
    """Snapshot the current RNG state for each backend."""
    targets = backends or list(_BACKENDS.keys())
    with _lock:
        return {
            name: _BACKENDS[name].get_state_fn()
            for name in targets
            if name in _BACKENDS
        }


def set_states(states: Dict[str, Any]) -> None:
    """Restore RNG states from a previous ``get_states()`` snapshot."""
    with _lock:
        for name, state in states.items():
            if name in _BACKENDS:
                _BACKENDS[name].set_state_fn(state)


@contextmanager
def temp_seed(
    value: int, *, deterministic: bool = False
) -> Generator[None, None, None]:
    """
    Context manager that seeds all RNGs on entry and restores their
    previous states on exit.

    Example
    -------
    >>> with seedall.temp_seed(0):
    ...     x = np.random.rand()   # reproducible
    >>> y = np.random.rand()       # back to original sequence
    """
    old_states = get_states()
    old_deterministic = _get_torch_deterministic()
    seed(value, deterministic=deterministic)
    try:
        yield
    finally:
        set_states(old_states)
        if deterministic:
            _set_torch_deterministic(old_deterministic)


class SeedContext:
    """
    Reusable seeding context -- call ``.enter()`` / ``.exit()`` manually
    when a context manager isn't convenient (e.g. in test setUp/tearDown),
    or use as a regular ``with`` statement.
    """

    def __init__(self, value: int, *, deterministic: bool = False):
        self.value = value
        self.deterministic = deterministic
        self._saved_states: Optional[Dict[str, Any]] = None
        self._saved_det: Optional[bool] = None
        self._active = False

    def enter(self) -> None:
        if self._active:
            raise RuntimeError("SeedContext.enter() called while already active")
        self._saved_states = get_states()
        self._saved_det = _get_torch_deterministic()
        self._active = True
        seed(self.value, deterministic=self.deterministic)

    def exit(self) -> None:
        if not self._active:
            raise RuntimeError("SeedContext.exit() called without a matching enter()")
        if self._saved_states is not None:
            set_states(self._saved_states)
        if self.deterministic and self._saved_det is not None:
            _set_torch_deterministic(self._saved_det)
        self._active = False

    def __enter__(self) -> SeedContext:
        self.enter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.exit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_torch_deterministic(enabled: bool) -> None:
    try:
        import torch
        torch.use_deterministic_algorithms(enabled)
        torch.backends.cudnn.deterministic = enabled
        torch.backends.cudnn.benchmark = not enabled
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("Could not set torch deterministic mode: %s", exc)


def _get_torch_deterministic() -> bool:
    try:
        import torch
        return torch.are_deterministic_algorithms_enabled()
    except (ImportError, AttributeError):
        return False
