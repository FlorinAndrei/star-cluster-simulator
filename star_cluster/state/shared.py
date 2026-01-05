"""Shared memory management for inter-process communication."""

import numpy as np
from multiprocessing import shared_memory, Value
from ctypes import c_bool, c_double
from typing import Optional
import uuid


class SharedState:
    """Manages shared memory arrays and control flags between processes."""

    def __init__(
        self, num_stars: int, create: bool = True, name_prefix: Optional[str] = None
    ):
        """
        Initialize shared memory state.

        Args:
            num_stars: Number of stars in the simulation
            create: If True, create new shared memory; if False, attach to existing
            name_prefix: Prefix for shared memory names (generated if None)
        """
        self.num_stars = num_stars
        self.create = create
        # Use short prefix for macOS compatibility (31 char limit for shm names)
        self.name_prefix = name_prefix or f"sc_{uuid.uuid4().hex[:4]}"

        # Define array shapes and sizes (short names for macOS compatibility)
        self._array_specs = {
            'pos': (num_stars, 3),
            'vel': (num_stars, 3),
            'mass': (num_stars,),
            'mag': (num_stars,),
            'temp': (num_stars,),
            'com': (3,),  # Center of mass (x, y, z)
        }

        # Shared memory blocks
        self._shm_blocks: dict[str, shared_memory.SharedMemory] = {}
        self._arrays: dict[str, np.ndarray] = {}

        # Control flags using multiprocessing Value (process-safe)
        if create:
            self._running = Value(c_bool, True)
            self._paused = Value(c_bool, False)
            self._save_requested = Value(c_bool, False)
            self._sim_time = Value(c_double, 0.0)
            self._initial_energy = Value(c_double, 0.0)
            self._current_energy = Value(c_double, 0.0)
            self._steps_per_second = Value(c_double, 0.0)
            self._unbound_count = Value(c_double, 0.0)
        else:
            self._running = None
            self._paused = None
            self._save_requested = None
            self._sim_time = None
            self._initial_energy = None
            self._current_energy = None
            self._steps_per_second = None
            self._unbound_count = None

        # Initialize shared memory blocks
        self._init_shared_memory()

    def _init_shared_memory(self):
        """Initialize or attach to shared memory blocks."""
        for name, shape in self._array_specs.items():
            shm_name = f"{self.name_prefix}_{name}"
            size = int(np.prod(shape)) * np.float64().nbytes

            if self.create:
                shm = shared_memory.SharedMemory(name=shm_name, create=True, size=size)
            else:
                shm = shared_memory.SharedMemory(name=shm_name, create=False)

            self._shm_blocks[name] = shm
            self._arrays[name] = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)

            if self.create:
                self._arrays[name][:] = 0.0

    @property
    def positions(self) -> np.ndarray:
        """Star positions array (N, 3)."""
        return self._arrays['pos']

    @property
    def velocities(self) -> np.ndarray:
        """Star velocities array (N, 3)."""
        return self._arrays['vel']

    @property
    def masses(self) -> np.ndarray:
        """Star masses array (N,)."""
        return self._arrays['mass']

    @property
    def magnitudes(self) -> np.ndarray:
        """Star absolute magnitudes array (N,)."""
        return self._arrays['mag']

    @property
    def temperatures(self) -> np.ndarray:
        """Star temperatures array (N,)."""
        return self._arrays['temp']

    @property
    def center_of_mass(self) -> np.ndarray:
        """Center of mass position (3,)."""
        return self._arrays['com']

    @property
    def running(self) -> bool:
        """Whether the simulation should continue running."""
        return self._running.value

    @running.setter
    def running(self, value: bool):
        self._running.value = value

    @property
    def paused(self) -> bool:
        """Whether the physics simulation is paused."""
        return self._paused.value

    @paused.setter
    def paused(self, value: bool):
        self._paused.value = value

    @property
    def save_requested(self) -> bool:
        """Whether a save has been requested."""
        return self._save_requested.value

    @save_requested.setter
    def save_requested(self, value: bool):
        self._save_requested.value = value

    @property
    def sim_time(self) -> float:
        """Current simulation time."""
        return self._sim_time.value

    @sim_time.setter
    def sim_time(self, value: float):
        self._sim_time.value = value

    @property
    def initial_energy(self) -> float:
        """Initial total energy of the system."""
        return self._initial_energy.value

    @initial_energy.setter
    def initial_energy(self, value: float):
        self._initial_energy.value = value

    @property
    def current_energy(self) -> float:
        """Current total energy of the system."""
        return self._current_energy.value

    @current_energy.setter
    def current_energy(self, value: float):
        self._current_energy.value = value

    @property
    def steps_per_second(self) -> float:
        """Physics integration steps per second."""
        return self._steps_per_second.value

    @steps_per_second.setter
    def steps_per_second(self, value: float):
        self._steps_per_second.value = value

    @property
    def unbound_count(self) -> int:
        """Number of unbound (escaping) stars."""
        return int(self._unbound_count.value)

    @unbound_count.setter
    def unbound_count(self, value: int):
        self._unbound_count.value = float(value)

    def set_control_values(
        self,
        running: Value,
        paused: Value,
        save_requested: Value,
        sim_time: Value,
        initial_energy: Value,
        current_energy: Value,
        steps_per_second: Value,
        unbound_count: Value,
    ):
        """Set control value references (used when attaching to existing state)."""
        self._running = running
        self._paused = paused
        self._save_requested = save_requested
        self._sim_time = sim_time
        self._initial_energy = initial_energy
        self._current_energy = current_energy
        self._steps_per_second = steps_per_second
        self._unbound_count = unbound_count

    def copy_from(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        magnitudes: np.ndarray,
        temperatures: np.ndarray,
    ):
        """Copy initial data into shared memory arrays."""
        np.copyto(self.positions, positions)
        np.copyto(self.velocities, velocities)
        np.copyto(self.masses, masses)
        np.copyto(self.magnitudes, magnitudes)
        np.copyto(self.temperatures, temperatures)

    def get_snapshot(self) -> dict:
        """Get a copy of all current state data."""
        return {
            'positions': self.positions.copy(),
            'velocities': self.velocities.copy(),
            'masses': self.masses.copy(),
            'magnitudes': self.magnitudes.copy(),
            'temperatures': self.temperatures.copy(),
            'sim_time': self.sim_time,
            'num_stars': self.num_stars,
            'initial_energy': self.initial_energy,
        }

    def cleanup(self):
        """Close and unlink shared memory blocks."""
        for name, shm in self._shm_blocks.items():
            shm.close()
            if self.create:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
        self._shm_blocks.clear()
        self._arrays.clear()

    def close(self):
        """Close shared memory blocks without unlinking (for child processes)."""
        for shm in self._shm_blocks.values():
            shm.close()
        self._shm_blocks.clear()
        self._arrays.clear()
