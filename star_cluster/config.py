"""Configuration constants for the star cluster simulation."""

# Simulation parameters
NUM_STARS = 1000
SOFTENING = 0.1  # Gravitational softening parameter to prevent singularities
TIMESTEP = 1e-5  # Integration timestep
STEPS_PER_UPDATE = 4  # Physics steps between shared memory updates
G = 1.0  # Gravitational constant (simulation units)
PRECISION = 32  # Floating point precision: 64 or 32 bits

# Initialization parameters
PLUMMER_SCALE = 1.0  # Plummer sphere scale radius

# Stellar mass range (solar masses) for Kroupa IMF
MASS_MIN = 0.08  # Hydrogen burning limit
MASS_MAX = 100.0  # Massive star upper limit

# Default cluster profile: "young", "globular", or "intermediate"
CLUSTER_PROFILE = "globular"

# Solar reference values (for stellar property calculations)
SOLAR_TEMPERATURE = 5778.0  # Kelvin
SOLAR_MAGNITUDE = 4.83  # Absolute visual magnitude

# Visualization parameters
CAMERA_DISTANCE = 10.0  # Initial camera distance
CAMERA_FOV = 60.0  # Field of view in degrees
STAR_SIZE = 5  # Star marker size in pixels
BRIGHTNESS_FLOOR = 0.05  # Minimum brightness for dimmest stars (0.0-1.0)

# File paths
SAVE_DIRECTORY = "saved_states"

# Help text (used in both CLI --help and in-app H key overlay)
HELP_CONTENT = (
    "--- Controls ---\n"
    "Mouse drag: Rotate view\n"
    "Scroll wheel: Zoom in/out\n"
    "Right-click: Track star\n"
    "Q: Quit\n"
    "P: Pause/Resume physics\n"
    "S: Save simulation state\n"
    "R: Reset camera/tracking\n"
    "F: Toggle fullscreen\n"
    "C: Toggle star colors\n"
    "T: Toggle trajectory visibility\n"
    "X: Clear trajectory data\n"
    "H: Toggle this help"
)
