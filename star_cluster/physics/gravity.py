"""JAX-accelerated gravitational force calculations."""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

from ..config import G, SOFTENING


@partial(jit, static_argnames=['softening', 'g'])
def compute_pairwise_acceleration(
    pos_i: jnp.ndarray,
    pos_j: jnp.ndarray,
    mass_j: jnp.ndarray,
    softening: float = SOFTENING,
    g: float = G,
) -> jnp.ndarray:
    """
    Compute gravitational acceleration on particle i due to particle j.

    a_i = -G * m_j * (r_i - r_j) / (|r_i - r_j|^2 + eps^2)^(3/2)

    Args:
        pos_i: Position of particle i (3,)
        pos_j: Position of particle j (3,)
        mass_j: Mass of particle j (scalar)
        softening: Softening parameter
        g: Gravitational constant

    Returns:
        Acceleration vector (3,)
    """
    r_ij = pos_i - pos_j
    r_squared = jnp.sum(r_ij**2)
    r_soft_squared = r_squared + softening**2
    r_soft_cubed = r_soft_squared**1.5

    # Avoid division by zero when i == j (r_ij = 0)
    # The softening handles this, but we use where for numerical stability
    acceleration = jnp.where(
        r_squared > 0, -g * mass_j * r_ij / r_soft_cubed, jnp.zeros(3)
    )

    return acceleration


@partial(jit, static_argnames=['softening', 'g'])
def compute_acceleration_on_particle(
    pos_i: jnp.ndarray,
    all_positions: jnp.ndarray,
    all_masses: jnp.ndarray,
    softening: float = SOFTENING,
    g: float = G,
) -> jnp.ndarray:
    """
    Compute total gravitational acceleration on a single particle from all others.

    Args:
        pos_i: Position of particle i (3,)
        all_positions: Positions of all particles (N, 3)
        all_masses: Masses of all particles (N,)
        softening: Softening parameter
        g: Gravitational constant

    Returns:
        Total acceleration vector (3,)
    """
    # Vectorize over all j particles
    accelerations = vmap(
        lambda pos_j, mass_j: compute_pairwise_acceleration(
            pos_i, pos_j, mass_j, softening, g
        )
    )(all_positions, all_masses)

    # Sum contributions from all particles
    return jnp.sum(accelerations, axis=0)


@partial(jit, static_argnames=['softening', 'g'])
def compute_all_accelerations(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    softening: float = SOFTENING,
    g: float = G,
) -> jnp.ndarray:
    """
    Compute gravitational accelerations for all particles.

    This is the main function used by the integrator. It uses nested vmap
    for efficient parallelization on GPU.

    Args:
        positions: Positions of all particles (N, 3)
        masses: Masses of all particles (N,)
        softening: Softening parameter
        g: Gravitational constant

    Returns:
        Accelerations for all particles (N, 3)
    """
    # Vectorize over all i particles
    accelerations = vmap(
        lambda pos_i: compute_acceleration_on_particle(
            pos_i, positions, masses, softening, g
        )
    )(positions)

    return accelerations


def get_device_info() -> str:
    """Get information about JAX devices being used."""
    devices = jax.devices()
    device_strs = [f"{d.platform}:{d.device_kind}" for d in devices]
    return f"JAX devices: {device_strs}"
