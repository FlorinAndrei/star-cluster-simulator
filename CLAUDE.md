# Star Cluster Simulator - Development Guide

## Project Overview

A real-time N-body gravity simulation using JAX for GPU-accelerated physics and Vispy for 3D visualization. Simulates realistic star clusters with proper stellar mass distributions (Kroupa IMF), mass-luminosity relations, and optional central black holes.

## Architecture

```
star_cluster/
├── config.py           # Central configuration constants
├── initialization/
│   └── generators.py   # Cluster generation (IMF, stellar properties, virial velocities)
├── physics/
│   ├── engine.py       # Physics process main loop
│   ├── gravity.py      # Gravitational acceleration calculations
│   └── integrator.py   # Leapfrog integration, energy calculations
├── state/
│   ├── shared.py       # SharedState class for inter-process communication
│   └── persistence.py  # HDF5 save/load functionality
└── visualization/
    ├── renderer.py     # Vispy-based 3D visualization (runs in main process)
    └── colors.py       # Blackbody color calculations, brightness mapping
```

## Key Technical Constraints

### JAX JIT Compilation

- Functions decorated with `@jit` or `@partial(jit, ...)` are compiled by XLA
- **Do not use Python `int()`, `float()`, or `bool()` on JAX arrays inside JIT functions** - this causes `ConcretizationTypeError`
- Return JAX arrays from JIT functions, convert to Python types outside
- Static arguments (passed via `static_argnames`) must be hashable constants

Example of correct pattern:
```python
# In integrator.py
@jit
def compute_unbound_count(...) -> jnp.ndarray:
    ...
    return jnp.sum(total_energy > 0)  # Returns JAX array, not int

# In engine.py (outside JIT)
unbound = int(compute_unbound_count(...))  # Convert to int here
```

### Multiprocessing Architecture

- **Physics runs in a separate process** (required for JIT compilation isolation)
- **Visualization runs in main process** (required for OpenGL)
- Communication via shared memory (`SharedState` class)
- Control flags use `multiprocessing.Value` with proper locking
- Short shared memory names required for macOS compatibility (31 char limit)

### Vispy/OpenGL Considerations

- Markers with overlapping positions need `depth_test=False` to prevent z-fighting
- DPI scaling varies by platform - use `canvas.dpi / 96.0` for UI element sizing
- Text positioning must account for DPI scaling

## Coding Conventions

- Configuration constants belong in `config.py`
- Help text for CLI and in-app overlay uses shared `HELP_CONTENT` constant
- Black holes are indicated by `temperature == 0`
- Particle arrays include black hole (if present) at index 0
- Use `num_particles` (not `num_stars`) when array sizes matter

## Common Pitfalls

1. **Black hole initialization order**: Black hole must be added BEFORE computing virial velocities, otherwise stars get incorrect velocities and exhibit "spray fountain" behavior

2. **Array size mismatch**: When black hole is present, `num_particles = num_stars + 1`. The `SharedState` must be sized for `num_particles`

3. **Energy calculations**: Per-star energy for unbound detection differs from total system energy - must sum potential contributions from all other stars

4. **Trajectory visualization**: Use `set_gl_state(depth_test=False)` for trajectory markers to prevent partial visibility

## Running the Simulation

```bash
python main.py                          # 1000 stars, globular profile
python main.py -n 5000 --profile young  # 5000 stars, young cluster
python main.py --black-hole 5000        # Add 5000 M☉ central black hole
python main.py --load saved.h5          # Load saved state
```

## Testing Changes

- Run with small star count (100-500) for quick iteration
- Watch for energy drift (displayed as percentage) - should stay small
- Monitor unbound count - initial ~10% is normal from virial initialization
- Check console for error messages from physics process
