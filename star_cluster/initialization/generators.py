"""Star cluster initialization functions."""

import numpy as np
from typing import Tuple, Optional
from ..config import (
    NUM_STARS,
    PLUMMER_SCALE,
    MASS_MIN,
    MASS_MAX,
    CLUSTER_PROFILE,
    SOLAR_TEMPERATURE,
    SOLAR_MAGNITUDE,
    G,
    SOFTENING,
)


def generate_plummer_positions(
    num_stars: int = NUM_STARS,
    scale: float = PLUMMER_SCALE,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate star positions following a Plummer sphere distribution.

    https://en.wikipedia.org/wiki/Plummer_model

    The Plummer model has density profile: rho(r) = (3M / 4*pi*a^3) * (1 + r^2/a^2)^(-5/2)

    Args:
        num_stars: Number of stars to generate
        scale: Plummer scale radius
        rng: Random number generator (created if None)

    Returns:
        positions: Array of shape (num_stars, 3) with star positions
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample radii using inverse transform sampling
    # The cumulative mass fraction is M(r)/M_total = r^3 / (r^2 + a^2)^(3/2)
    # Inverting: r = a / sqrt(u^(-2/3) - 1) where u is uniform [0,1)
    u = rng.uniform(0.001, 1.0, num_stars)  # Avoid u=0 to prevent infinity
    radii = scale / np.sqrt(u ** (-2.0 / 3.0) - 1.0)

    # Generate random directions (uniform on sphere)
    cos_theta = rng.uniform(-1.0, 1.0, num_stars)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = rng.uniform(0.0, 2.0 * np.pi, num_stars)

    # Convert to Cartesian coordinates
    positions = np.zeros((num_stars, 3), dtype=np.float64)
    positions[:, 0] = radii * sin_theta * np.cos(phi)
    positions[:, 1] = radii * sin_theta * np.sin(phi)
    positions[:, 2] = radii * cos_theta

    return positions


def generate_masses_kroupa(
    num_stars: int = NUM_STARS,
    mass_min: float = MASS_MIN,
    mass_max: float = MASS_MAX,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate star masses following the Kroupa (2001) Initial Mass Function.

    https://en.wikipedia.org/wiki/Initial_mass_function

    The Kroupa IMF is a broken power law:
        ξ(M) ∝ M^(-α) where:
        α = 1.3 for 0.08 < M < 0.50 M☉
        α = 2.3 for M > 0.50 M☉

    Args:
        num_stars: Number of stars
        mass_min: Minimum mass (solar masses), default 0.08
        mass_max: Maximum mass (solar masses), default 100.0
        rng: Random number generator

    Returns:
        masses: Array of shape (num_stars,) with star masses in solar masses
    """
    if rng is None:
        rng = np.random.default_rng()

    # Kroupa IMF break points and slopes
    m_break = 0.5  # Break between low-mass and high-mass slopes
    alpha_low = 1.3  # Slope for 0.08 < M < 0.50
    alpha_high = 2.3  # Slope for M > 0.50

    # Ensure mass_min >= 0.08 (hydrogen burning limit)
    mass_min = max(mass_min, 0.08)

    # Compute segment boundaries
    m1 = mass_min
    m2 = min(m_break, mass_max)
    m3 = mass_max

    # For power law ξ(M) ∝ M^(-α), the integral from m_a to m_b is:
    # ∫ M^(-α) dM = [M^(1-α) / (1-α)] from m_a to m_b
    def power_law_integral(m_a, m_b, alpha):
        if m_a >= m_b:
            return 0.0
        exp = 1.0 - alpha
        return (m_b**exp - m_a**exp) / exp

    # Compute weights for each segment (ensuring continuity at break)
    # At the break, we need: k_low * m_break^(-alpha_low) = k_high * m_break^(-alpha_high)
    # So k_high / k_low = m_break^(alpha_high - alpha_low)
    continuity_factor = m_break ** (alpha_high - alpha_low)

    # Integrals for each segment
    integral_low = power_law_integral(m1, m2, alpha_low) if m1 < m_break else 0.0
    integral_high = (
        power_law_integral(max(m_break, m1), m3, alpha_high) * continuity_factor
        if m3 > m_break
        else 0.0
    )

    total_weight = integral_low + integral_high

    # Probability of being in low-mass segment
    p_low = integral_low / total_weight if total_weight > 0 else 0.0

    # Sample which segment each star belongs to
    in_low_segment = rng.random(num_stars) < p_low

    # Sample from power law using inverse transform
    # For ξ(M) ∝ M^(-α), CDF^(-1)(u) = (m_min^(1-α) + u*(m_max^(1-α) - m_min^(1-α)))^(1/(1-α))
    def sample_power_law(u, m_a, m_b, alpha):
        exp = 1.0 - alpha
        return (m_a**exp + u * (m_b**exp - m_a**exp)) ** (1.0 / exp)

    masses = np.empty(num_stars, dtype=np.float64)

    # Sample low-mass stars
    n_low = np.sum(in_low_segment)
    if n_low > 0 and m1 < m_break:
        u_low = rng.random(n_low)
        masses[in_low_segment] = sample_power_law(u_low, m1, m2, alpha_low)

    # Sample high-mass stars
    n_high = num_stars - n_low
    if n_high > 0 and m3 > m_break:
        u_high = rng.random(n_high)
        m_start = max(m_break, m1)
        masses[~in_low_segment] = sample_power_law(u_high, m_start, m3, alpha_high)

    return masses


def compute_luminosity(masses: np.ndarray) -> np.ndarray:
    """
    Compute stellar luminosity from mass using main sequence relations.

    https://en.wikipedia.org/wiki/Mass%E2%80%93luminosity_relation

    Uses piecewise mass-luminosity relation (in solar units):
        M < 0.43:  L ∝ M^2.3
        0.43-2:    L ∝ M^4
        2-55:      L ∝ M^3.5
        M > 55:    L ∝ M^1

    Coefficients are chosen to ensure continuity at boundaries.

    Args:
        masses: Star masses in solar masses (N,)

    Returns:
        luminosities: Star luminosities in solar luminosities (N,)
    """
    luminosities = np.empty_like(masses)

    # Define mass boundaries
    m1, m2, m3 = 0.43, 2.0, 55.0

    # Compute luminosity for each regime with continuity
    # L = M^α, with coefficients to ensure continuity

    # For M < 0.43: L = 0.23 * M^2.3
    mask1 = masses < m1
    luminosities[mask1] = 0.23 * masses[mask1] ** 2.3

    # For 0.43 <= M < 2: L = M^4
    # At M=0.43: L = 0.23 * 0.43^2.3 ≈ 0.033, and 0.43^4 ≈ 0.034 (close enough)
    mask2 = (masses >= m1) & (masses < m2)
    luminosities[mask2] = masses[mask2] ** 4.0

    # For 2 <= M < 55: L = 1.4 * M^3.5
    # At M=2: 2^4 = 16, and 1.4 * 2^3.5 ≈ 15.8 (continuous)
    mask3 = (masses >= m2) & (masses < m3)
    luminosities[mask3] = 1.4 * masses[mask3] ** 3.5

    # For M >= 55: L = 32000 * M
    # At M=55: 1.4 * 55^3.5 ≈ 1.76e6, and 32000 * 55 = 1.76e6 (continuous)
    mask4 = masses >= m3
    luminosities[mask4] = 32000.0 * masses[mask4]

    return luminosities


def compute_radius(masses: np.ndarray) -> np.ndarray:
    """
    Compute stellar radius from mass using main sequence relation.

    Uses approximate relation: R/R☉ ≈ (M/M☉)^0.8 for M > 1
    and R/R☉ ≈ (M/M☉)^0.57 for M < 1 (flatter for low-mass stars)

    This is very empirical.

    Args:
        masses: Star masses in solar masses (N,)

    Returns:
        radii: Star radii in solar radii (N,)
    """
    radii = np.empty_like(masses)

    # Low mass stars (M < 1): R ∝ M^0.57
    mask_low = masses < 1.0
    radii[mask_low] = masses[mask_low] ** 0.57

    # High mass stars (M >= 1): R ∝ M^0.8
    mask_high = masses >= 1.0
    radii[mask_high] = masses[mask_high] ** 0.8

    return radii


def compute_temperature(luminosities: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """
    Compute stellar effective temperature from luminosity and radius.

    Uses Stefan-Boltzmann law: L = 4πR²σT⁴
    In solar units: T/T☉ = (L/L☉)^0.25 / (R/R☉)^0.5

    Args:
        luminosities: Star luminosities in solar luminosities (N,)
        radii: Star radii in solar radii (N,)

    Returns:
        temperatures: Star effective temperatures in Kelvin (N,)
    """
    # T/T_sun = (L/L_sun)^0.25 / (R/R_sun)^0.5
    t_ratio = (luminosities**0.25) / (radii**0.5)
    return SOLAR_TEMPERATURE * t_ratio


def compute_magnitude(luminosities: np.ndarray) -> np.ndarray:
    """
    Compute absolute visual magnitude from luminosity.

    Uses: M_abs = M_sun - 2.5 * log10(L/L_sun)
    Where M_sun ≈ 4.83 (absolute visual magnitude of the Sun)

    Args:
        luminosities: Star luminosities in solar luminosities (N,)

    Returns:
        magnitudes: Absolute visual magnitudes (N,)
    """
    return SOLAR_MAGNITUDE - 2.5 * np.log10(luminosities)


def compute_potential_energy(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float = SOFTENING,
    g: float = G,
    batch_size: int = 1000,
) -> float:
    """
    Compute total gravitational potential energy of the system.

    U = -G * sum_{i<j} m_i * m_j / sqrt(|r_i - r_j|^2 + eps^2)

    Uses batched computation to avoid O(N^2) memory usage.

    Args:
        positions: Star positions (N, 3)
        masses: Star masses (N,)
        softening: Softening parameter
        g: Gravitational constant
        batch_size: Number of stars to process per batch

    Returns:
        Total potential energy (negative for bound system)
    """
    n = len(masses)
    softening_sq = softening**2
    potential = 0.0

    # Process in batches to avoid creating full N×N matrices
    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)

        # For each batch of i values, compute interactions with j > i only
        # to avoid double counting
        pos_i = positions[i_start:i_end, :]  # (batch, 3)
        mass_i = masses[i_start:i_end]  # (batch,)

        # For each i in the batch, we need j > i
        # We'll compute against all j > i_start, then mask out j <= i
        j_start = i_start  # Start from same index to get upper triangle
        pos_j = positions[j_start:, :]  # (n - j_start, 3)
        mass_j = masses[j_start:]  # (n - j_start,)

        # Compute displacements: dx[i_idx, j_idx] = pos_i[i_idx] - pos_j[j_idx]
        dx = pos_i[:, np.newaxis, :] - pos_j[np.newaxis, :, :]  # (batch, n-j_start, 3)

        # Compute softened distances
        r2 = np.sum(dx**2, axis=2) + softening_sq  # (batch, n-j_start)
        r = np.sqrt(r2)

        # Compute mass products
        mass_products = (
            mass_i[:, np.newaxis] * mass_j[np.newaxis, :]
        )  # (batch, n-j_start)

        # Compute potential contributions
        potential_contrib = -g * mass_products / r  # (batch, n-j_start)

        # Create mask for upper triangle (j > i)
        # i_idx goes from 0 to batch_size-1, corresponding to i = i_start to i_end-1
        # j_idx goes from 0 to n-j_start-1, corresponding to j = j_start to n-1
        # We want j > i, i.e., (j_start + j_idx) > (i_start + i_idx)
        # Since j_start == i_start: j_idx > i_idx
        batch_len = i_end - i_start
        j_len = n - j_start
        i_indices = np.arange(batch_len)[:, np.newaxis]
        j_indices = np.arange(j_len)[np.newaxis, :]
        mask = j_indices > i_indices  # (batch, n-j_start)

        # Sum only upper triangle contributions
        potential += np.sum(potential_contrib * mask)

    return potential


def generate_virial_velocities(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float = SOFTENING,
    g: float = G,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate velocities satisfying virial equilibrium.

    For virial equilibrium: 2K + U = 0, where K is kinetic energy and U is potential.
    This means K = -U/2 for a bound system.

    https://en.wikipedia.org/wiki/Virial_mass

    Args:
        positions: Star positions (N, 3)
        masses: Star masses (N,)
        softening: Gravitational softening
        g: Gravitational constant
        rng: Random number generator

    Returns:
        velocities: Array of shape (N, 3) with star velocities
    """
    if rng is None:
        rng = np.random.default_rng()

    num_stars = len(masses)

    # Compute potential energy
    U = compute_potential_energy(positions, masses, softening, g)

    # Target kinetic energy for virial equilibrium
    K_target = -U / 2.0

    # Generate random velocity directions (isotropic)
    # Using Gaussian distribution for each component gives isotropic directions
    velocities = rng.standard_normal((num_stars, 3))

    # Normalize to unit vectors
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    velocities = velocities / (speeds + 1e-10)

    # Assign velocity magnitudes proportional to sqrt(mass) for equipartition
    # v_i = v_0 / sqrt(m_i), then scale to match target K
    v_magnitudes = 1.0 / np.sqrt(masses)

    # Current kinetic energy with these velocities
    # K = 0.5 * sum(m_i * v_i^2) = 0.5 * sum(m_i * (v_0/sqrt(m_i))^2) = 0.5 * N * v_0^2
    # We need to find the scaling factor

    # Apply initial magnitudes
    velocities = velocities * v_magnitudes[:, np.newaxis]

    # Compute current kinetic energy
    K_current = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)

    # Scale velocities to match target kinetic energy
    if K_current > 0:
        scale = np.sqrt(K_target / K_current)
        velocities *= scale

    # Remove any net momentum (center of mass velocity)
    total_mass = np.sum(masses)
    momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    cm_velocity = momentum / total_mass
    velocities -= cm_velocity

    return velocities


def generate_young_cluster_stars(
    num_stars: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate stellar properties for a young cluster (age < 100 Myr).

    All stars are on the main sequence with full Kroupa IMF.

    Args:
        num_stars: Number of stars
        rng: Random number generator

    Returns:
        Tuple of (masses, luminosities, temperatures, magnitudes)
    """
    print("\tGenerating masses (Kroupa IMF, full range)...")
    masses = generate_masses_kroupa(num_stars, rng=rng)

    print("\tComputing stellar properties from mass...")
    luminosities = compute_luminosity(masses)
    radii = compute_radius(masses)
    temperatures = compute_temperature(luminosities, radii)
    magnitudes = compute_magnitude(luminosities)

    return masses, luminosities, temperatures, magnitudes


def generate_globular_cluster_stars(
    num_stars: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate stellar properties for an old globular cluster (age ~10-12 Gyr).

    Population breakdown (approximate for old metal-poor cluster like M13):
    - ~85% Low-mass main sequence (0.08-0.85 M☉)
    - ~10% Red Giant Branch (RGB) stars
    - ~5% Horizontal Branch (HB) stars

    Args:
        num_stars: Number of stars
        rng: Random number generator

    Returns:
        Tuple of (masses, luminosities, temperatures, magnitudes)
    """
    # Population fractions
    f_ms = 0.85  # Main sequence
    f_rgb = 0.10  # Red Giant Branch
    f_hb = 0.05  # Horizontal Branch

    n_ms = int(f_ms * num_stars)
    n_rgb = int(f_rgb * num_stars)
    n_hb = num_stars - n_ms - n_rgb

    print(f"\tGenerating {n_ms} main sequence stars...")
    print(f"\tGenerating {n_rgb} red giant branch stars...")
    print(f"\tGenerating {n_hb} horizontal branch stars...")

    # Arrays to hold all stellar properties
    masses = np.empty(num_stars, dtype=np.float64)
    luminosities = np.empty(num_stars, dtype=np.float64)
    temperatures = np.empty(num_stars, dtype=np.float64)

    idx = 0

    # Main sequence stars (truncated at turnoff ~0.85 M☉)
    if n_ms > 0:
        ms_masses = generate_masses_kroupa(n_ms, mass_min=0.08, mass_max=0.85, rng=rng)
        ms_lum = compute_luminosity(ms_masses)
        ms_radii = compute_radius(ms_masses)
        ms_temp = compute_temperature(ms_lum, ms_radii)

        masses[idx : idx + n_ms] = ms_masses
        luminosities[idx : idx + n_ms] = ms_lum
        temperatures[idx : idx + n_ms] = ms_temp
        idx += n_ms

    # Red Giant Branch stars
    # RGB stars: mass ~0.8 M☉ (original ~0.85, lost some mass)
    # Luminosity: 10-1000 L☉ (wide range along RGB)
    # Temperature: 3500-5000 K (cool giants)
    if n_rgb > 0:
        rgb_masses = np.full(n_rgb, 0.8)  # Approximately constant
        # Luminosity follows roughly log-uniform distribution along RGB
        rgb_lum = 10 ** rng.uniform(1.0, 3.0, n_rgb)  # 10 to 1000 L☉
        # Temperature inversely correlated with luminosity on RGB
        # Brighter RGB stars are cooler (approaching RGB tip)
        rgb_temp = 5000 - 1500 * (np.log10(rgb_lum) - 1) / 2  # ~5000K to ~3500K

        masses[idx : idx + n_rgb] = rgb_masses
        luminosities[idx : idx + n_rgb] = rgb_lum
        temperatures[idx : idx + n_rgb] = rgb_temp
        idx += n_rgb

    # Horizontal Branch stars
    # HB stars: mass ~0.5-0.7 M☉ (significant mass loss)
    # Luminosity: ~50 L☉ (fairly constant, characteristic of HB)
    # Temperature: 5000-25000 K (spans blue to red HB)
    if n_hb > 0:
        hb_masses = rng.uniform(0.5, 0.7, n_hb)
        hb_lum = rng.uniform(40, 60, n_hb)  # ~50 L☉ with some scatter
        # Temperature bimodal: some blue (hot), some red (cool)
        # For simplicity, uniform distribution spanning the HB
        hb_temp = rng.uniform(5000, 25000, n_hb)

        masses[idx : idx + n_hb] = hb_masses
        luminosities[idx : idx + n_hb] = hb_lum
        temperatures[idx : idx + n_hb] = hb_temp

    # Compute magnitudes from luminosities
    magnitudes = compute_magnitude(luminosities)

    # Shuffle to mix populations
    shuffle_idx = rng.permutation(num_stars)
    masses = masses[shuffle_idx]
    luminosities = luminosities[shuffle_idx]
    temperatures = temperatures[shuffle_idx]
    magnitudes = magnitudes[shuffle_idx]

    return masses, luminosities, temperatures, magnitudes


def generate_intermediate_cluster_stars(
    num_stars: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate stellar properties for an intermediate-age cluster (1-3 Gyr).

    Population breakdown:
    - ~90% Main sequence (0.08-2.0 M☉, turnoff at ~2 M☉)
    - ~8% Red giants (evolved from 2-3 M☉ progenitors)
    - ~2% Core helium burning / red clump stars

    Args:
        num_stars: Number of stars
        rng: Random number generator

    Returns:
        Tuple of (masses, luminosities, temperatures, magnitudes)
    """
    # Population fractions
    f_ms = 0.90
    f_rgb = 0.08
    f_rc = 0.02  # Red clump

    n_ms = int(f_ms * num_stars)
    n_rgb = int(f_rgb * num_stars)
    n_rc = num_stars - n_ms - n_rgb

    print(f"\tGenerating {n_ms} main sequence stars...")
    print(f"\tGenerating {n_rgb} red giant stars...")
    print(f"\tGenerating {n_rc} red clump stars...")

    masses = np.empty(num_stars, dtype=np.float64)
    luminosities = np.empty(num_stars, dtype=np.float64)
    temperatures = np.empty(num_stars, dtype=np.float64)

    idx = 0

    # Main sequence stars (turnoff at ~2 M☉ for ~1-3 Gyr age)
    if n_ms > 0:
        ms_masses = generate_masses_kroupa(n_ms, mass_min=0.08, mass_max=2.0, rng=rng)
        ms_lum = compute_luminosity(ms_masses)
        ms_radii = compute_radius(ms_masses)
        ms_temp = compute_temperature(ms_lum, ms_radii)

        masses[idx : idx + n_ms] = ms_masses
        luminosities[idx : idx + n_ms] = ms_lum
        temperatures[idx : idx + n_ms] = ms_temp
        idx += n_ms

    # Red giants (evolved from ~2-3 M☉ progenitors)
    # Current mass ~2 M☉, luminosity 50-500 L☉, temp 4000-5000K
    if n_rgb > 0:
        rgb_masses = rng.uniform(1.8, 2.2, n_rgb)
        rgb_lum = 10 ** rng.uniform(1.7, 2.7, n_rgb)  # 50-500 L☉
        rgb_temp = 5000 - 1000 * (np.log10(rgb_lum) - 1.7) / 1.0

        masses[idx : idx + n_rgb] = rgb_masses
        luminosities[idx : idx + n_rgb] = rgb_lum
        temperatures[idx : idx + n_rgb] = rgb_temp
        idx += n_rgb

    # Red clump stars (core helium burning, like HB but for higher mass)
    # Mass ~2 M☉, luminosity ~50-100 L☉, temp ~5000K
    if n_rc > 0:
        rc_masses = rng.uniform(1.8, 2.2, n_rc)
        rc_lum = rng.uniform(50, 100, n_rc)
        rc_temp = rng.uniform(4800, 5200, n_rc)

        masses[idx : idx + n_rc] = rc_masses
        luminosities[idx : idx + n_rc] = rc_lum
        temperatures[idx : idx + n_rc] = rc_temp

    # Compute magnitudes from luminosities
    magnitudes = compute_magnitude(luminosities)

    # Shuffle to mix populations
    shuffle_idx = rng.permutation(num_stars)
    masses = masses[shuffle_idx]
    luminosities = luminosities[shuffle_idx]
    temperatures = temperatures[shuffle_idx]
    magnitudes = magnitudes[shuffle_idx]

    return masses, luminosities, temperatures, magnitudes


def initialize_cluster(
    num_stars: int = NUM_STARS,
    seed: Optional[int] = None,
    profile: str = CLUSTER_PROFILE,
    black_hole_mass: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize a complete star cluster with all properties.

    Supports different cluster profiles:
    - "young": Full Kroupa IMF, all main sequence (age < 100 Myr)
    - "globular": Old cluster like M13 (age ~10-12 Gyr) with RGB and HB
    - "intermediate": Middle-aged cluster (age 1-3 Gyr) with some giants

    Args:
        num_stars: Number of stars in the cluster
        seed: Random seed for reproducibility
        profile: Cluster type ("young", "globular", or "intermediate")
        black_hole_mass: Optional mass for a central black hole (solar masses)

    Returns:
        Tuple of (positions, velocities, masses, magnitudes, temperatures)
    """
    rng = np.random.default_rng(seed)

    print("\tGenerating positions (Plummer sphere)...")
    positions = generate_plummer_positions(num_stars, rng=rng)

    # Generate stellar properties based on profile
    if profile == "young":
        masses, luminosities, temperatures, magnitudes = generate_young_cluster_stars(
            num_stars, rng
        )
    elif profile == "globular":
        masses, luminosities, temperatures, magnitudes = (
            generate_globular_cluster_stars(num_stars, rng)
        )
    elif profile == "intermediate":
        masses, luminosities, temperatures, magnitudes = (
            generate_intermediate_cluster_stars(num_stars, rng)
        )
    else:
        raise ValueError(f"Unknown cluster profile: {profile}")

    # Add central black hole BEFORE computing virial velocities so the
    # gravitational potential includes the black hole's contribution
    if black_hole_mass is not None:
        print(f"\tAdding central black hole ({black_hole_mass:.0f} M☉)...")
        # Black hole at origin
        bh_position = np.array([[0.0, 0.0, 0.0]])
        bh_mass = np.array([black_hole_mass])
        # Temperature 0 marks it as a black hole (handled specially in visualization)
        bh_temperature = np.array([0.0])
        # Very bright magnitude so it's visible
        bh_magnitude = np.array([-10.0])

        # Prepend black hole to position and mass arrays
        positions = np.vstack([bh_position, positions])
        masses = np.concatenate([bh_mass, masses])
        temperatures = np.concatenate([bh_temperature, temperatures])
        magnitudes = np.concatenate([bh_magnitude, magnitudes])

    print("\tGenerating virial velocities...")
    velocities = generate_virial_velocities(positions, masses, rng=rng)

    # Set black hole velocity to zero (it should stay at center)
    if black_hole_mass is not None:
        velocities[0] = 0.0

    # Print some statistics
    star_masses = masses[1:] if black_hole_mass else masses
    print(f"\t  Mass range: {star_masses.min():.3f} - {star_masses.max():.3f} M☉")
    print(
        f"\t  Temperature range: {temperatures[temperatures > 0].min():.0f} - {temperatures.max():.0f} K"
    )
    print(f"\t  Magnitude range: {magnitudes.min():.1f} - {magnitudes.max():.1f}")

    return positions, velocities, masses, magnitudes, temperatures
