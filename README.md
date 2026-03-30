# seedall

Seed **all** common RNGs in one call for reproducible experiments.

```python
import seedall

seedall.seed(42)  # seeds random, numpy, torch, tensorflow, jax, cupy — whatever is installed
```

## Supported backends

| Backend    | What gets seeded                                         |
|------------|----------------------------------------------------------|
| `random`   | Python stdlib `random`                                   |
| `hashseed` | `PYTHONHASHSEED` env var                                 |
| `numpy`    | `np.random.seed()`                                       |
| `torch`    | `torch.manual_seed()` + `cuda.manual_seed_all()`        |
| `tensorflow` | `tf.random.set_seed()`                                |
| `jax`      | Creates a `jax.random.PRNGKey` (retrieve via states API) |
| `cupy`     | `cp.random.seed()`                                       |

Missing libraries are silently skipped — install only what you need.

## API

### `seedall.seed(value, *, backends=None, deterministic=False, warn_missing=False)`

Seed all (or selected) backends. Returns `dict[str, bool]` showing what was seeded.

```python
# Seed everything
seedall.seed(42)

# Seed only specific backends
seedall.seed(42, backends=["numpy", "torch"])

# Also enable PyTorch deterministic mode (slower but fully reproducible)
seedall.seed(42, deterministic=True)
```

### `seedall.temp_seed(value, *, deterministic=False)`

Context manager — seeds on entry, restores previous RNG states on exit.

```python
with seedall.temp_seed(0):
    x = np.random.rand(100)   # reproducible
y = np.random.rand(100)       # back to original sequence
```

### `seedall.available()`

List detected backends:

```python
>>> seedall.available()
['random', 'hashseed', 'numpy', 'torch']
```

### `seedall.get_states()` / `seedall.set_states(states)`

Snapshot and restore RNG states manually:

```python
states = seedall.get_states()
# ... do stuff ...
seedall.set_states(states)  # rewind
```

### `seedall.SeedContext(value)`

Class-based alternative when a context manager isn't convenient:

```python
ctx = seedall.SeedContext(42)
ctx.enter()   # seed
# ... run experiment ...
ctx.exit()    # restore
```

## Install

```bash
pip install seedall                # core (stdlib random only)
pip install seedall[numpy]         # + numpy
pip install seedall[torch]         # + pytorch
pip install seedall[all]           # everything
```

## License

MIT
