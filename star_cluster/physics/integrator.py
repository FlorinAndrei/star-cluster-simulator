"""Leapfrog integrator for N-body simulation using JAX."""

import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from typing import Tuple

from .gravity import compute_all_accelerations
from ..config import TIMESTEP, STEPS_PER_UPDATE, SOFTENING, G


@partial(jit, static_argnames=['dt', 'softening', 'g'])
def leapfrog_step(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    masses: jnp.ndarray,
    dt: float = TIMESTEP,
    softening: float = SOFTENING,
    g: float = G,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform a single Leapfrog (Kick-Drift-Kick) integration step.

    The Leapfrog integrator is symplectic, meaning it conserves energy
    well over long integration times.

    https://en.wikipedia.org/wiki/Symplectic_integrator
    https://en.wikipedia.org/wiki/Leapfrog_integration

    KDK scheme:
    1. v(t + dt/2) = v(t) + a(t) * dt/2        (kick)
    2. x(t + dt) = x(t) + v(t + dt/2) * dt     (drift)
    3. v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2  (kick)

    Args:
        positions: Current positions (N, 3)
        velocities: Current velocities (N, 3)
        masses: Particle masses (N,)
        dt: Timestep
        softening: Gravitational softening
        g: Gravitational constant

    Returns:
        Tuple of (new_positions, new_velocities)
    """
    # Compute accelerations at current positions
    accelerations = compute_all_accelerations(positions, masses, softening, g)

    # First kick: half step in velocity
    velocities_half = velocities + accelerations * (dt / 2.0)

    # Drift: full step in position
    new_positions = positions + velocities_half * dt

    # Compute accelerations at new positions
    new_accelerations = compute_all_accelerations(new_positions, masses, softening, g)

    # Second kick: complete the velocity step
    new_velocities = velocities_half + new_accelerations * (dt / 2.0)

    return new_positions, new_velocities


@partial(jit, static_argnames=['num_steps', 'dt', 'softening', 'g'])
def integrate_multiple_steps(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    masses: jnp.ndarray,
    num_steps: int = STEPS_PER_UPDATE,
    dt: float = TIMESTEP,
    softening: float = SOFTENING,
    g: float = G,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform multiple integration steps efficiently using lax.scan.

    This function fuses multiple timesteps into a single compiled kernel,
    reducing overhead from repeated kernel launches.

    Args:
        positions: Initial positions (N, 3)
        velocities: Initial velocities (N, 3)
        masses: Particle masses (N,)
        num_steps: Number of integration steps to perform
        dt: Timestep
        softening: Gravitational softening
        g: Gravitational constant

    Returns:
        Tuple of (final_positions, final_velocities)
    """

    def scan_fn(carry, _):
        pos, vel = carry
        new_pos, new_vel = leapfrog_step(pos, vel, masses, dt, softening, g)
        return (new_pos, new_vel), None

    initial_state = (positions, velocities)
    (final_positions, final_velocities), _ = lax.scan(
        scan_fn, initial_state, None, length=num_steps
    )

    return final_positions, final_velocities


@jit
def compute_kinetic_energy(velocities: jnp.ndarray, masses: jnp.ndarray) -> jnp.ndarray:
    """Compute total kinetic energy: K = 0.5 * sum(m * v^2)."""
    v_squared = jnp.sum(velocities**2, axis=1)
    return 0.5 * jnp.sum(masses * v_squared)


@partial(jit, static_argnames=['softening', 'g'])
def compute_potential_energy(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    softening: float = SOFTENING,
    g: float = G,
) -> jnp.ndarray:
    """
    Compute total gravitational potential energy using JAX vectorization.

    U = -G * sum_{i<j} m_i * m_j / sqrt(|r_i - r_j|^2 + eps^2)
    """
    n = positions.shape[0]

    # Compute all pairwise distances
    # diff[i, j] = positions[i] - positions[j]
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    r_squared = jnp.sum(diff**2, axis=2)  # (N, N)
    r_soft = jnp.sqrt(r_squared + softening**2)  # (N, N)

    # Compute pairwise potential contributions: -G * m_i * m_j / r_ij
    mass_products = masses[:, None] * masses[None, :]  # (N, N)
    pairwise_potential = -g * mass_products / r_soft  # (N, N)

    # Zero out diagonal (self-interaction) and take upper triangle only
    # to avoid double counting
    mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    potential = jnp.sum(jnp.where(mask, pairwise_potential, 0.0))

    return potential


@partial(jit, static_argnames=['softening', 'g'])
def compute_total_energy(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    masses: jnp.ndarray,
    softening: float = SOFTENING,
    g: float = G,
) -> jnp.ndarray:
    """Compute total energy (kinetic + potential)."""
    K = compute_kinetic_energy(velocities, masses)
    U = compute_potential_energy(positions, masses, softening, g)
    return K + U


@jit
def compute_center_of_mass(positions: jnp.ndarray, masses: jnp.ndarray) -> jnp.ndarray:
    """Compute center of mass position: sum(m_i * r_i) / sum(m_i)."""
    total_mass = jnp.sum(masses)
    weighted_positions = jnp.sum(masses[:, None] * positions, axis=0)
    return weighted_positions / total_mass


@partial(jit, static_argnames=['softening', 'g'])
def compute_unbound_count(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    masses: jnp.ndarray,
    softening: float = SOFTENING,
    g: float = G,
) -> jnp.ndarray:
    """
    Count stars with positive total energy (unbound/escaping).

    A star is unbound when its kinetic energy exceeds the magnitude of
    its gravitational potential energy from all other stars.

    Args:
        positions: Star positions (N, 3)
        velocities: Star velocities (N, 3)
        masses: Star masses (N,)
        softening: Gravitational softening
        g: Gravitational constant

    Returns:
        Number of unbound stars
    """
    n = positions.shape[0]

    # Per-star kinetic energy: K_i = 0.5 * m_i * |v_i|^2
    v_squared = jnp.sum(velocities**2, axis=1)  # (N,)
    kinetic = 0.5 * masses * v_squared  # (N,)

    # Per-star potential energy: U_i = -G * m_i * sum_j(m_j / r_ij)
    # Compute all pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    r_squared = jnp.sum(diff**2, axis=2)  # (N, N)
    r_soft = jnp.sqrt(r_squared + softening**2)  # (N, N)

    # Avoid division by zero on diagonal (will be masked out)
    r_safe = jnp.where(r_soft > 0, r_soft, 1.0)

    # Potential contribution from each pair: -G * m_i * m_j / r_ij
    # For star i, sum over all j != i
    mass_products = masses[:, None] * masses[None, :]  # (N, N)
    pairwise_potential = -g * mass_products / r_safe  # (N, N)

    # Zero out diagonal (self-interaction)
    diagonal_mask = jnp.eye(n, dtype=bool)
    pairwise_potential = jnp.where(diagonal_mask, 0.0, pairwise_potential)

    # Per-star potential energy (sum over all other stars)
    potential = jnp.sum(pairwise_potential, axis=1)  # (N,)

    # Total energy per star
    total_energy = kinetic + potential  # (N,)

    # Count stars with positive energy (unbound)
    unbound = jnp.sum(total_energy > 0)

    return unbound
