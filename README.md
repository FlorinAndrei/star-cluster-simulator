# Star Cluster Simulator

Accurate physical simulation of a star cluster, using JAX and (where available) GPU acceleration.

Demo on YouTube: https://www.youtube.com/watch?v=1qAy3-AHfuo

## Features

The emphasis is on physical accuracy and realism.

- Multiprocessing: one process runs the GPU-accelerated math via JAX; the other process does the visualization
- Shared memory: the physics engine and the graphics engine see the same data in shared memory
- Accurate physical simulation: N-body simulator using the symplectic Leapfrog integrator
- The time step is adjustable
- Realistic cluster geometry: distribution of stars in the cluster is calculated via the Plummer model
- Realistic star distributions: the masses are generated via the Kroupa IMF (initial mass function) for young clusters; luminosity includes the main sequence among other distributions (this depends on the --profile option)
- Physically accurate star measurements: radius is computed from mass via a main sequence model; temperature is computed from radius and luminosity via Stefan-Boltzmann
- Clusters are at virial equilibrium
- Visualization shows realistic colors for the stars; RGB is computed from temperature via the Tanner Helland formula; brightness on screen is computed from stellar magnitude
- 32 bit precision used by default, 64 bit may be used instead if requested
- Simulation state can be saved, and reloaded later from a file
- Individual star trajectories can be captured on demand
- Optionally, the cluster may include a black hole at the center, more massive than the stars (but not supermassive)

## Install

Create a virtual environment and assume it:

```
python3.13 -m venv .venv
source .venv/bin/activate
```

Install basic packages:

```
pip install -r requirements.txt
```

Install JAX:

```
# CPU (Windows, macOS)
pip install jax

# NVIDIA GPU (Linux, including WSL)
pip install "jax[cuda13]"

# Metal (macOS)
# see this issue: https://github.com/jax-ml/jax/issues/34109
# and use only --precision 32 on this platform
pip install jax==0.5.0 jaxlib==0.5.0 jax-metal==0.1.1
```

For other JAX install options, see https://github.com/jax-ml/jax

Test JAX after installing it:

```
python -c 'import jax; print(jax.numpy.arange(10))'
```

## Options

In the terminal, use `python main.py --help` to see all the CLI options. In the GUI, press the H key to see all the GUI options.

Command-line options:

```
-h, --help            show this help message and exit
--load, -l FILE       Load simulation state from HDF5 file
--num-stars, -n N     Number of stars (default: 1000, ignored if loading)
--seed, -s SEED       Random seed for reproducibility (ignored if loading)
--profile, -pf TYPE   Cluster type: young (full IMF), globular (old, like M13), intermediate (1-3 Gyr). Default: globular.
--steps-per-update, -t N
                    Physics steps between screen updates (default: 4)
--precision, -p BITS  Floating point precision: 64 or 32 bits (default: 32).
--timestep, -dt DT    Integration timestep (default: 1e-05).
--brightness-floor, -b FLOOR
                    Minimum brightness for dimmest stars, 0.0-1.0. Use ~0.05 for realistic appearance, ~0.25 for video-game
                    style where all stars are clearly visible (default: 0.05).
--black-hole, -bh MASS
                    Add a central black hole with specified mass (solar masses). Typical IMBH: 1000-10000. If not specified, no
                    black hole is added.
```

GUI controls:

```
Mouse drag: Rotate view
Scroll wheel: Zoom in/out
Right-click: Track star
Q: Quit
P: Pause/Resume physics
S: Save simulation state
R: Reset camera/tracking
F: Toggle fullscreen
C: Toggle star colors
T: Toggle trajectory visibility
X: Clear trajectory data
H: Toggle this help
```

A few options discussed in detail:

### --profile

python main.py --profile young        # Default, full IMF
python main.py --profile globular     # Old cluster like M13
python main.py --profile intermediate # 1-3 Gyr cluster

Profile Characteristics

| Profile      | Age        | Main Sequence            | Giants    | Special Stars     |
|--------------|------------|--------------------------|-----------|-------------------|
| young        | <100 Myr   | 0.08-100 M☉ (full)       | None      | O/B stars present |
| globular     | ~10-12 Gyr | 0.08-0.85 M☉ (truncated) | RGB (10%) | HB stars (5%)     |
| intermediate | 1-3 Gyr    | 0.08-2.0 M☉              | RGB (8%)  | Red clump (2%)    |

Visual Differences

Young cluster:
- Many dim red stars, some bright blue giants
- Temperature range: ~2000-40000K

Globular cluster (M13-like):
- Many dim red main sequence stars
- Prominent red giants (cool, bright)
- Blue horizontal branch stars (hot, medium brightness)
- No young blue stars
- Temperature range: ~2000-25000K

Intermediate cluster:
- Mostly dim red stars
- Some red giants
- No very hot blue stars
- Temperature range: ~2000-8700K

### --precision

By default, the app runs with --precision 32, where all major variables are 32 bit. For a more accurate simulation, use --precision 64, but the app will run much more slowly.

## Star Trajectories

You may notice the stars that seem to roughly "orbit" the dense core of the cluster do not have closed trajectories. This behavior is realistic. The simulator is not broken.

In a two-body system (planet orbiting a star), orbits are closed ellipses because the gravitational potential is Keplerian (point mass, 1/r potential).

In an N-body system like a star cluster, the gravitational potential comes from a distributed mass. As a star moves through the cluster, the shell theorem varies the amount of mass that pulls it inwards. It may also go through close encounters with other stars. For all these reasons, the trajectory can be complex, and open.

The simulator also correctly shows cluster evaporation. Some stars gain enough energy and leave the cluster, never to return. Watch the Unbound metric in the top-left corner of the screen, it counts the stars that do this.
