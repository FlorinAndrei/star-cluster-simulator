"""HDF5-based state persistence for simulation save/load."""

import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from ..config import SAVE_DIRECTORY


def generate_save_filename() -> str:
    """Generate a save filename with timestamp in YYYYMMDD-HHMMSS format."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"simulation_{timestamp}.h5"


def ensure_save_directory(base_path: Optional[Path] = None) -> Path:
    """Ensure the save directory exists and return its path."""
    if base_path is None:
        save_dir = Path(SAVE_DIRECTORY)
    else:
        save_dir = base_path / SAVE_DIRECTORY

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def save_state(
    filepath: Path,
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    magnitudes: np.ndarray,
    temperatures: np.ndarray,
    sim_time: float,
    initial_energy: float,
    camera_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save simulation state to an HDF5 file.

    Args:
        filepath: Path to save the file
        positions: Star positions (N, 3)
        velocities: Star velocities (N, 3)
        masses: Star masses (N,)
        magnitudes: Star absolute magnitudes (N,)
        temperatures: Star temperatures (N,)
        sim_time: Current simulation time
        initial_energy: Initial total energy of the system
        camera_state: Optional camera state dictionary

    Returns:
        Path to the saved file as a string
    """
    num_stars = len(masses)

    with h5py.File(filepath, 'w') as f:
        # Create groups
        stars_grp = f.create_group('stars')
        sim_grp = f.create_group('simulation')

        # Save star data
        stars_grp.create_dataset('positions', data=positions, dtype='float64')
        stars_grp.create_dataset('velocities', data=velocities, dtype='float64')
        stars_grp.create_dataset('masses', data=masses, dtype='float64')
        stars_grp.create_dataset('magnitudes', data=magnitudes, dtype='float64')
        stars_grp.create_dataset('temperatures', data=temperatures, dtype='float64')

        # Save simulation metadata
        sim_grp.create_dataset('time', data=sim_time, dtype='float64')
        sim_grp.create_dataset('num_stars', data=num_stars, dtype='int64')
        sim_grp.create_dataset('initial_energy', data=initial_energy, dtype='float64')

        # Save camera state if provided
        if camera_state is not None:
            camera_grp = f.create_group('camera')
            camera_grp.create_dataset(
                'center', data=camera_state.get('center', [0, 0, 0]), dtype='float64'
            )
            camera_grp.create_dataset(
                'azimuth', data=camera_state.get('azimuth', 0.0), dtype='float64'
            )
            camera_grp.create_dataset(
                'elevation', data=camera_state.get('elevation', 30.0), dtype='float64'
            )
            camera_grp.create_dataset(
                'distance', data=camera_state.get('distance', 10.0), dtype='float64'
            )

        # Add metadata attributes
        f.attrs['version'] = '1.0'
        f.attrs['created'] = datetime.now().isoformat()

    return str(filepath)


def load_state(
    filepath: Path,
) -> Tuple[Dict[str, np.ndarray], float, int, float, Optional[Dict[str, Any]]]:
    """
    Load simulation state from an HDF5 file.

    Args:
        filepath: Path to the HDF5 file

    Returns:
        Tuple of:
            - Dictionary with star arrays (positions, velocities, masses, magnitudes, temperatures)
            - Simulation time
            - Number of stars
            - Initial energy
            - Camera state dictionary (or None if not saved)
    """
    with h5py.File(filepath, 'r') as f:
        # Load star data
        star_data = {
            'positions': f['stars/positions'][:],
            'velocities': f['stars/velocities'][:],
            'masses': f['stars/masses'][:],
            'magnitudes': f['stars/magnitudes'][:],
            'temperatures': f['stars/temperatures'][:],
        }

        # Load simulation metadata
        sim_time = float(f['simulation/time'][()])
        num_stars = int(f['simulation/num_stars'][()])

        # Load initial energy (default to 0 for older save files)
        initial_energy = 0.0
        if 'initial_energy' in f['simulation']:
            initial_energy = float(f['simulation/initial_energy'][()])

        # Load camera state if present
        camera_state = None
        if 'camera' in f:
            camera_state = {
                'center': f['camera/center'][:].tolist(),
                'azimuth': float(f['camera/azimuth'][()]),
                'elevation': float(f['camera/elevation'][()]),
                'distance': float(f['camera/distance'][()]),
            }

    return star_data, sim_time, num_stars, initial_energy, camera_state


def save_simulation_state(
    shared_state,
    camera_state: Optional[Dict[str, Any]] = None,
    base_path: Optional[Path] = None,
) -> str:
    """
    Save the current simulation state from shared memory.

    Args:
        shared_state: SharedState object with current simulation data
        camera_state: Optional camera state dictionary
        base_path: Base path for save directory (uses cwd if None)

    Returns:
        Path to the saved file
    """
    save_dir = ensure_save_directory(base_path)
    filename = generate_save_filename()
    filepath = save_dir / filename

    # Get snapshot of current state
    snapshot = shared_state.get_snapshot()

    saved_path = save_state(
        filepath,
        snapshot['positions'],
        snapshot['velocities'],
        snapshot['masses'],
        snapshot['magnitudes'],
        snapshot['temperatures'],
        snapshot['sim_time'],
        snapshot['initial_energy'],
        camera_state,
    )

    return saved_path
