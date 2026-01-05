"""Physics engine process for N-body simulation."""

import os
import time
from ctypes import c_bool, c_double
from multiprocessing import Value

import numpy as np

# Configure JAX
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp

# Precision dtype mapping
PRECISION_DTYPES = {
    64: jnp.float64,
    32: jnp.float32,
}

from ..config import SOFTENING, STEPS_PER_UPDATE, TIMESTEP, G, PRECISION
from ..state.shared import SharedState
from .gravity import get_device_info
from .integrator import (
    compute_center_of_mass,
    compute_total_energy,
    compute_unbound_count,
    integrate_multiple_steps,
)


def run_physics_engine(
    num_stars: int,
    name_prefix: str,
    running: Value,
    paused: Value,
    save_requested: Value,
    sim_time: Value,
    initial_energy: Value,
    current_energy: Value,
    steps_per_second: Value,
    unbound_count: Value,
    steps_per_update: int = STEPS_PER_UPDATE,
    precision: int = PRECISION,
    timestep: float = TIMESTEP,
):
    """
    Main physics engine loop running in a separate process.

    This function attaches to shared memory created by the main process,
    runs the N-body simulation, and updates shared arrays periodically.

    Args:
        num_stars: Number of stars in the simulation
        name_prefix: Shared memory name prefix
        running: Shared flag indicating if simulation should continue
        paused: Shared flag indicating if physics is paused
        save_requested: Shared flag for save requests (not handled here)
        sim_time: Shared simulation time value
        steps_per_update: Physics steps between shared memory updates
        precision: Floating point precision (64 or 32 bits)
        timestep: Integration timestep
    """
    # Configure JAX precision - must be done before any JAX operations
    jax.config.update("jax_enable_x64", precision == 64)
    dtype = PRECISION_DTYPES[precision]

    print(f"Physics engine starting (precision: {precision}-bit)...")
    print(get_device_info())

    # Attach to existing shared memory
    shared_state = SharedState(num_stars, create=False, name_prefix=name_prefix)
    shared_state.set_control_values(
        running,
        paused,
        save_requested,
        sim_time,
        initial_energy,
        current_energy,
        steps_per_second,
        unbound_count,
    )

    # Copy initial data from shared memory to JAX arrays
    positions = jnp.array(shared_state.positions, dtype=dtype)
    velocities = jnp.array(shared_state.velocities, dtype=dtype)
    masses = jnp.array(shared_state.masses, dtype=dtype)

    # Warm up JIT compilation with a test step
    print("Compiling physics kernels...")
    _ = integrate_multiple_steps(
        positions, velocities, masses, steps_per_update, timestep, SOFTENING, G
    )
    jax.block_until_ready(_)

    # Also compile energy, center of mass, and unbound count functions
    _ = compute_total_energy(positions, velocities, masses, SOFTENING, G)
    jax.block_until_ready(_)
    _ = compute_center_of_mass(positions, masses)
    jax.block_until_ready(_)
    _ = compute_unbound_count(positions, velocities, masses, SOFTENING, G)
    jax.block_until_ready(_)
    print("Physics engine ready.")

    # Compute initial energy if not already set (i.e., not loaded from file)
    if shared_state.initial_energy == 0.0:
        init_energy = compute_total_energy(positions, velocities, masses, SOFTENING, G)
        jax.block_until_ready(init_energy)
        shared_state.initial_energy = float(init_energy)
        print(f"Initial total energy: {shared_state.initial_energy:.6f}")

    current_time = shared_state.sim_time
    step_count = 0
    last_stats_time = time.time()

    print(f"Running {steps_per_update} simulation steps before each state update")

    try:
        while shared_state.running:
            if shared_state.paused:
                # When paused, just sleep briefly and check flags
                time.sleep(0.01)
                last_stats_time = time.time()  # Reset to avoid spike after unpause
                step_count = 0
                continue

            # Perform multiple integration steps
            positions, velocities = integrate_multiple_steps(
                positions, velocities, masses, steps_per_update, timestep, SOFTENING, G
            )

            # Block until GPU computation is complete
            jax.block_until_ready(positions)
            jax.block_until_ready(velocities)

            # Update simulation time and step count
            current_time += steps_per_update * timestep
            shared_state.sim_time = current_time
            step_count += steps_per_update

            # Compute current energy, center of mass, and unbound count
            energy = compute_total_energy(positions, velocities, masses, SOFTENING, G)
            com = compute_center_of_mass(positions, masses)
            unbound = compute_unbound_count(positions, velocities, masses, SOFTENING, G)
            jax.block_until_ready(energy)
            jax.block_until_ready(com)
            jax.block_until_ready(unbound)

            shared_state.current_energy = float(energy)
            np.copyto(shared_state.center_of_mass, np.asarray(com))
            shared_state.unbound_count = int(unbound)

            # Update steps per second
            now = time.time()
            elapsed = now - last_stats_time
            if elapsed >= 0.5:  # Update rate every 0.5 seconds
                shared_state.steps_per_second = step_count / elapsed
                step_count = 0
                last_stats_time = now

            # Update shared memory with new positions and velocities
            # Convert JAX arrays to numpy for shared memory
            np.copyto(shared_state.positions, np.asarray(positions))
            np.copyto(shared_state.velocities, np.asarray(velocities))

    except Exception as e:
        print(f"Physics engine error: {e}")
        raise
    finally:
        # Clean up shared memory attachment
        shared_state.close()
        print("Physics engine stopped.")
