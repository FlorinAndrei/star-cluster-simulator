#!/usr/bin/env python3
"""
Star Cluster Simulator

A real-time N-body gravity simulation using JAX for physics calculations
and Vispy for 3D visualization.

Usage:
    python main.py                    # Start new simulation with 1000 stars
    python main.py --num-stars 5000   # Start with 5000 stars
    python main.py --load FILE.h5     # Load saved simulation state
"""

import argparse
import sys
from pathlib import Path
from multiprocessing import Process, Value
from ctypes import c_bool, c_double

from star_cluster.config import (
    NUM_STARS,
    STEPS_PER_UPDATE,
    PRECISION,
    TIMESTEP,
    CLUSTER_PROFILE,
    BRIGHTNESS_FLOOR,
    HELP_CONTENT,
)
from star_cluster.state.shared import SharedState
from star_cluster.state.persistence import load_state
from star_cluster.initialization.generators import initialize_cluster
from star_cluster.physics.engine import run_physics_engine

# Note: visualization.renderer is imported lazily in main() to avoid
# loading graphics libraries when only --help is requested


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Star Cluster Simulator - N-body gravity simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_CONTENT,
    )
    parser.add_argument(
        '--load',
        '-l',
        type=str,
        metavar='FILE',
        help='Load simulation state from HDF5 file',
    )
    parser.add_argument(
        '--num-stars',
        '-n',
        type=int,
        default=NUM_STARS,
        metavar='N',
        help=f'Number of stars (default: {NUM_STARS}, ignored if loading)',
    )
    parser.add_argument(
        '--seed',
        '-s',
        type=int,
        default=None,
        metavar='SEED',
        help='Random seed for reproducibility (ignored if loading)',
    )
    parser.add_argument(
        '--profile',
        '-pf',
        type=str,
        choices=['young', 'globular', 'intermediate'],
        default=CLUSTER_PROFILE,
        metavar='TYPE',
        help=f'Cluster type: young (full IMF), globular (old, like M13), '
        f'intermediate (1-3 Gyr). Default: {CLUSTER_PROFILE}.',
    )
    parser.add_argument(
        '--steps-per-update',
        '-t',
        type=int,
        default=STEPS_PER_UPDATE,
        metavar='N',
        help=f'Physics steps between screen updates (default: {STEPS_PER_UPDATE})',
    )
    parser.add_argument(
        '--precision',
        '-p',
        type=int,
        choices=[64, 32],
        default=PRECISION,
        metavar='BITS',
        help=f'Floating point precision: 64 or 32 bits (default: {PRECISION}).',
    )
    parser.add_argument(
        '--timestep',
        '-dt',
        type=float,
        default=TIMESTEP,
        metavar='DT',
        help=f'Integration timestep (default: {TIMESTEP}).',
    )
    parser.add_argument(
        '--brightness-floor',
        '-b',
        type=float,
        default=BRIGHTNESS_FLOOR,
        metavar='FLOOR',
        help=f'Minimum brightness for dimmest stars, 0.0-1.0. '
        f'Use ~0.05 for realistic appearance, ~0.25 for video-game style '
        f'where all stars are clearly visible (default: {BRIGHTNESS_FLOOR}).',
    )
    parser.add_argument(
        '--black-hole',
        '-bh',
        type=float,
        default=None,
        metavar='MASS',
        help='Add a central black hole with specified mass (solar masses). '
        'Typical IMBH: 1000-10000. If not specified, no black hole is added.',
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    print("Use --help for command-line options.")
    print("Press H in the visualization window to toggle on-screen help.")
    args = parse_args()

    # Initialize or load simulation data
    initial_energy = 0.0  # Will be computed by physics engine if 0
    if args.load:
        # Load from file
        load_path = Path(args.load)
        if not load_path.exists():
            print(f"Error: File not found: {load_path}")
            sys.exit(1)

        print(f"Loading simulation from: {load_path}")
        star_data, sim_time, num_stars, initial_energy, camera_state = load_state(
            load_path
        )
        print(f"Loaded {num_stars} stars, simulation time: {sim_time:.3f}")

        positions = star_data['positions']
        velocities = star_data['velocities']
        masses = star_data['masses']
        magnitudes = star_data['magnitudes']
        temperatures = star_data['temperatures']
    else:
        # Generate new cluster
        num_stars = args.num_stars
        print(f"Initializing {args.profile} cluster with {num_stars} stars...")

        positions, velocities, masses, magnitudes, temperatures = initialize_cluster(
            num_stars,
            seed=args.seed,
            profile=args.profile,
            black_hole_mass=args.black_hole,
        )
        sim_time = 0.0
        camera_state = None

        print("Cluster initialized.")

    # Create shared memory (use actual particle count which includes black hole if present)
    num_particles = len(positions)
    print("Setting up shared memory...")
    shared_state = SharedState(num_particles, create=True)

    # Copy initial data to shared memory
    shared_state.copy_from(positions, velocities, masses, magnitudes, temperatures)
    shared_state.sim_time = sim_time
    shared_state.initial_energy = initial_energy

    # Create shared control values for the physics process
    running = shared_state._running
    paused = shared_state._paused
    save_requested = shared_state._save_requested
    sim_time_val = shared_state._sim_time
    initial_energy_val = shared_state._initial_energy
    current_energy_val = shared_state._current_energy
    steps_per_second_val = shared_state._steps_per_second
    unbound_count_val = shared_state._unbound_count

    # Start physics process
    print("Starting physics engine...")
    physics_process = Process(
        target=run_physics_engine,
        args=(
            num_particles,
            shared_state.name_prefix,
            running,
            paused,
            save_requested,
            sim_time_val,
            initial_energy_val,
            current_energy_val,
            steps_per_second_val,
            unbound_count_val,
            args.steps_per_update,
            args.precision,
            args.timestep,
        ),
    )
    physics_process.start()

    # Run visualization in main process (required for OpenGL)
    # Import here to avoid loading graphics libraries for --help
    from star_cluster.visualization.renderer import run_visualization

    print("Starting visualization...")
    try:
        run_visualization(shared_state, camera_state, args.brightness_floor)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        shared_state.running = False
    finally:
        # Clean up
        print("Shutting down...")

        # Signal physics to stop
        shared_state.running = False

        # Wait for physics process
        physics_process.join(timeout=2.0)
        if physics_process.is_alive():
            print("Force terminating physics process...")
            physics_process.terminate()
            physics_process.join()

        # Clean up shared memory
        shared_state.cleanup()
        print("Done.")


if __name__ == '__main__':
    main()
