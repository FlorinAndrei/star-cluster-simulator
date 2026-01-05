"""Color and brightness calculations for star visualization."""

import numpy as np
from typing import Tuple


def _kelvin_to_rgb(temp: float) -> Tuple[float, float, float]:
    """
    Convert color temperature in Kelvin to RGB using Tanner Helland's algorithm.

    https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html
    Valid range: ~1000K to ~40000K

    Args:
        temp: Temperature in Kelvin

    Returns:
        Tuple of (R, G, B) values in 0-1 range
    """
    # Clamp temperature to valid range
    temp = max(1000, min(40000, temp))

    # Scale temperature to 0-100 range used by algorithm
    temp = temp / 100.0

    # Calculate red
    if temp <= 66:
        r = 255
    else:
        r = temp - 60
        r = 329.698727446 * (r**-0.1332047592)
        r = max(0, min(255, r))

    # Calculate green
    if temp <= 66:
        g = temp
        g = 99.4708025861 * np.log(g) - 161.1195681661
        g = max(0, min(255, g))
    else:
        g = temp - 60
        g = 288.1221695283 * (g**-0.0755148492)
        g = max(0, min(255, g))

    # Calculate blue
    if temp >= 66:
        b = 255
    elif temp <= 19:
        b = 0
    else:
        b = temp - 10
        b = 138.5177312231 * np.log(b) - 305.0447927307
        b = max(0, min(255, b))

    return (r / 255.0, g / 255.0, b / 255.0)


def generate_blackbody_lut(
    temp_min: float = 1000, temp_max: float = 40000, step: float = 100
) -> Tuple[np.ndarray, float, float, float]:
    """
    Generate a lookup table for blackbody colors.

    Args:
        temp_min: Minimum temperature in Kelvin
        temp_max: Maximum temperature in Kelvin
        step: Temperature step size in Kelvin

    Returns:
        Tuple of (lut, temp_min, temp_max, step) where lut is shape (N, 3)
    """
    num_entries = int((temp_max - temp_min) / step) + 1
    lut = np.zeros((num_entries, 3), dtype=np.float32)

    for i in range(num_entries):
        temp = temp_min + i * step
        lut[i] = _kelvin_to_rgb(temp)

    return lut, temp_min, temp_max, step


def temperatures_to_rgb(
    temperatures: np.ndarray, lut: np.ndarray, temp_min: float, step: float
) -> np.ndarray:
    """
    Convert an array of temperatures to RGB colors using the lookup table.

    Args:
        temperatures: Array of temperatures in Kelvin
        lut: Lookup table from generate_blackbody_lut
        temp_min: Minimum temperature of LUT
        step: Temperature step of LUT

    Returns:
        Array of shape (N, 3) with RGB values in 0-1 range
    """
    # Compute indices into LUT
    indices = (temperatures - temp_min) / step
    indices = np.clip(indices, 0, len(lut) - 1).astype(int)

    return lut[indices]


def magnitudes_to_brightness(
    magnitudes: np.ndarray,
    mag_min: float,
    mag_max: float,
    gamma: float = 0.5,
    floor: float = 0.25,
) -> np.ndarray:
    """
    Convert magnitudes to brightness values using a power curve.

    Brighter stars have lower (more negative) magnitudes.

    Args:
        magnitudes: Array of absolute magnitudes
        mag_min: Minimum magnitude (brightest)
        mag_max: Maximum magnitude (dimmest)
        gamma: Power curve exponent (<1 compresses dynamic range)
        floor: Minimum brightness for dimmest stars

    Returns:
        Array of brightness values in [floor, 1.0] range
    """
    # Normalize: 0 = dimmest, 1 = brightest
    normalized = (mag_max - magnitudes) / (mag_max - mag_min)
    normalized = np.clip(normalized, 0, 1)

    # Apply power curve and scale to [floor, 1.0]
    brightness = floor + (1.0 - floor) * (normalized**gamma)

    return brightness


# Pre-generate the default LUT at module load time
_DEFAULT_LUT, _LUT_TEMP_MIN, _LUT_TEMP_MAX, _LUT_STEP = generate_blackbody_lut()


def compute_star_colors(
    temperatures: np.ndarray,
    magnitudes: np.ndarray,
    mag_min: float,
    mag_max: float,
    gamma: float = 0.5,
    brightness_floor: float = 0.25,
) -> np.ndarray:
    """
    Compute RGBA colors for stars based on temperature and magnitude.

    Args:
        temperatures: Array of star temperatures in Kelvin
        magnitudes: Array of star absolute magnitudes
        mag_min: Minimum magnitude (brightest)
        mag_max: Maximum magnitude (dimmest)
        gamma: Brightness power curve exponent
        brightness_floor: Minimum brightness for dimmest stars

    Returns:
        Array of shape (N, 4) with RGBA values in 0-1 range
    """
    # Get base colors from temperature
    rgb = temperatures_to_rgb(temperatures, _DEFAULT_LUT, _LUT_TEMP_MIN, _LUT_STEP)

    # Get brightness from magnitude
    brightness = magnitudes_to_brightness(
        magnitudes, mag_min, mag_max, gamma, brightness_floor
    )

    # Apply brightness to RGB
    rgb = rgb * brightness[:, np.newaxis]

    # Create RGBA array with alpha = 1.0
    rgba = np.ones((len(temperatures), 4), dtype=np.float32)
    rgba[:, :3] = rgb

    # Handle black holes (temperature == 0): render as bright white
    black_hole_mask = temperatures == 0
    if np.any(black_hole_mask):
        rgba[black_hole_mask, :3] = 1.0  # Pure white

    return rgba
