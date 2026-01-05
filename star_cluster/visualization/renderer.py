"""Vispy-based real-time 3D visualization for star cluster simulation."""

import time
import numpy as np

# Set Vispy backend
# Use glfw on macOS for better OpenGL compatibility, pyglet elsewhere
import platform
import vispy

if platform.system() == 'Darwin':
    vispy.use('glfw', gl='gl2')
    import glfw

    _USE_GLFW = True
else:
    vispy.use('pyglet')
    glfw = None
    _USE_GLFW = False

from vispy import app, scene
from vispy.scene import visuals
from typing import Optional, Dict, Any

from ..state.shared import SharedState
from ..state.persistence import save_simulation_state
from ..config import (
    BRIGHTNESS_FLOOR,
    CAMERA_DISTANCE,
    CAMERA_FOV,
    HELP_CONTENT,
    STAR_SIZE,
)
from .colors import compute_star_colors


class StarClusterVisualizer:
    """Real-time 3D visualization of star cluster using Vispy."""

    def __init__(
        self,
        shared_state: SharedState,
        initial_camera: Optional[Dict[str, Any]] = None,
        brightness_floor: float = BRIGHTNESS_FLOOR,
    ):
        """
        Initialize the visualizer.

        Args:
            shared_state: SharedState object for reading star positions
            initial_camera: Optional initial camera state from loaded file
            brightness_floor: Minimum brightness for dimmest stars (0.0-1.0)
        """
        self.shared_state = shared_state
        self.initial_camera = initial_camera
        self.brightness_floor = brightness_floor

        # Store initial camera settings for reset
        self._default_camera = {
            'center': (0, 0, 0),
            'azimuth': 45.0,
            'elevation': 30.0,
            'distance': CAMERA_DISTANCE,
        }

        # Determine initial window size (half of primary screen dimensions)
        try:
            screens = app.screens()
            if screens:
                screen = screens[0]
                screen_w = screen['geometry']['width']
                screen_h = screen['geometry']['height']
                window_size = (screen_w // 2, screen_h // 2)
            else:
                window_size = (1200, 800)
        except Exception:
            window_size = (1200, 800)

        # Create the canvas with vsync for display-synchronized updates
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            title='Star Cluster Simulator',
            size=window_size,
            show=True,
            bgcolor='black',
            vsync=True,
        )

        # Print display scaling diagnostics
        print(f"Display diagnostics:")
        print(f"  Logical size: {self.canvas.size}")
        print(f"  Physical size: {self.canvas.physical_size}")
        print(f"  Pixel scale: {self.canvas.pixel_scale}")
        print(f"  DPI: {self.canvas.dpi}")

        # Set up the view
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(
            fov=CAMERA_FOV, distance=CAMERA_DISTANCE, elevation=30, azimuth=45
        )

        # Apply initial camera state if provided
        if initial_camera is not None:
            self._apply_camera_state(initial_camera)
        elif self.initial_camera is not None:
            self._apply_camera_state(self.initial_camera)

        # Create scatter plot for stars
        self.scatter = visuals.Markers()
        self.view.add(self.scatter)

        # Compute star colors from temperature and magnitude (computed once)
        # Use actual magnitude range from data for proper normalization
        mag_min = self.shared_state.magnitudes.min()
        mag_max = self.shared_state.magnitudes.max()
        self._star_colors = compute_star_colors(
            self.shared_state.temperatures,
            self.shared_state.magnitudes,
            mag_min,
            mag_max,
            brightness_floor=self.brightness_floor,
        )

        # Detect black hole(s) by temperature == 0
        self._black_hole_mask = self.shared_state.temperatures == 0
        self._has_black_hole = np.any(self._black_hole_mask)

        # Make black holes invisible in main scatter (they'll be rendered separately)
        if self._has_black_hole:
            self._star_colors[self._black_hole_mask, 3] = 0.0  # Set alpha to 0

        # Create separate visual for black hole(s) with hollow circle symbol
        self.black_hole_visual = visuals.Markers()
        self.view.add(self.black_hole_visual)
        self.black_hole_visual.visible = self._has_black_hole

        # Compute star sizes (standard size for all stars now)
        self._star_sizes = np.full(
            self.shared_state.num_stars, STAR_SIZE, dtype=np.float32
        )

        # Ring marker for highlighting the star closest to mouse cursor
        self.highlight_ring = visuals.Markers()
        self.view.add(self.highlight_ring)
        self.highlight_ring.visible = False
        self._highlighted_star_idx = None
        self._last_mouse_pos = None  # Track mouse position for per-frame updates

        # Text overlay for star info (shown when hovering over a star)
        self.star_info_text = scene.visuals.Text(
            text='',
            color='yellow',
            anchor_x='left',
            anchor_y='top',
            font_size=12,
            parent=self.canvas.scene,
        )
        self.star_info_text.visible = False

        # Ring marker for the star being tracked by camera (distinct from hover highlight)
        self.tracking_ring = visuals.Markers()
        self.view.add(self.tracking_ring)
        self.tracking_ring.visible = False
        self._tracked_star_idx = (
            None  # Star being tracked by camera (None = not tracking)
        )

        # Trajectory visualization for tracked stars
        self.trajectory_markers = visuals.Markers()
        self.trajectory_markers.set_gl_state(depth_test=False)
        self.view.add(self.trajectory_markers)
        self._trajectory_positions = []  # List of (x, y, z) positions
        self._trajectory_colors = []  # List of RGBA colors for each position
        self._trajectory_visible = True

        # Camera animation state
        self._animating = False
        self._anim_start_time = 0.0
        self._anim_duration = 0.5  # seconds
        self._anim_start_center = np.array([0.0, 0.0, 0.0])
        self._anim_start_azimuth = 0.0
        self._anim_start_elevation = 0.0
        self._anim_start_distance = 0.0
        self._anim_target_azimuth = None  # None means don't animate this property
        self._anim_target_elevation = None
        self._anim_target_distance = None

        # Scale UI elements based on DPI (96 is baseline)
        dpi_scale = self.canvas.dpi / 96.0

        # Create text overlay for stats (attached to canvas, not 3D view)
        self.stats_text = scene.visuals.Text(
            text='',
            color='white',
            anchor_x='left',
            anchor_y='bottom',
            font_size=12,
            parent=self.canvas.scene,
        )
        # Position in top-left corner (in pixel coordinates)
        self.stats_text.pos = (10, 10)

        # Text overlay for tracked star info (below stats, top-left)
        self.tracked_star_text = scene.visuals.Text(
            text='',
            color='cyan',
            anchor_x='left',
            anchor_y='bottom',
            font_size=12,
            parent=self.canvas.scene,
        )
        # Y position scales with DPI to stay below stats text
        self.tracked_star_text.pos = (10, int(95 * dpi_scale))
        self.tracked_star_text.visible = False

        # Create help overlay with background (centered, initially hidden)
        help_width = int(360 * dpi_scale)
        help_height = int(400 * dpi_scale)
        # Center on canvas
        canvas_center_x = self.canvas.size[0] / 2
        canvas_center_y = self.canvas.size[1] / 2
        # Text offset from center (scales with DPI)
        text_offset_x = int(130 * dpi_scale)

        # Semi-transparent background rectangle
        self.help_bg = scene.visuals.Rectangle(
            center=(canvas_center_x, canvas_center_y),
            width=help_width,
            height=help_height,
            color=(0, 0, 0, 0.9),
            parent=self.canvas.scene,
        )
        self.help_bg.visible = False
        # Help text on top of background
        self.help_text = scene.visuals.Text(
            text=HELP_CONTENT,
            color='white',
            anchor_x='left',
            anchor_y='center',
            font_size=14,
            parent=self.canvas.scene,
        )
        # Position left-aligned within the background
        self.help_text.pos = (canvas_center_x - text_offset_x, canvas_center_y)
        self._help_visible = False
        self.help_text.visible = False

        # Color mode toggle (True = colored, False = white)
        self._color_mode = True

        # Track fullscreen state locally (more reliable than querying canvas)
        self._is_fullscreen = False

        # Initial update
        self._update_scatter()
        self._update_stats()

        # Set up timer for updates (interval=0 runs as fast as vsync allows)
        self.timer = app.Timer(interval=0, connect=self._on_timer, start=True)

        # Connect keyboard and mouse events
        self.canvas.events.key_press.connect(self._on_key_press)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self.canvas.events.mouse_press.connect(self._on_mouse_press)

        # Cursor auto-hide state for fullscreen (using wall-clock time)
        self._cursor_visible = True
        self._cursor_last_move = time.time()
        self._cursor_hide_delay = 1.0  # seconds of inactivity before hiding

    def _apply_camera_state(self, camera_state: Dict[str, Any]):
        """Apply camera state from dictionary."""
        if 'center' in camera_state:
            self.view.camera.center = tuple(camera_state['center'])
        if 'azimuth' in camera_state:
            self.view.camera.azimuth = camera_state['azimuth']
        if 'elevation' in camera_state:
            self.view.camera.elevation = camera_state['elevation']
        if 'distance' in camera_state:
            self.view.camera.distance = camera_state['distance']

    def _get_camera_state(self) -> Dict[str, Any]:
        """Get current camera state as dictionary."""
        # If tracking a star, save the default center instead of the tracked
        # star's position (tracking is transient and shouldn't be persisted)
        if self._tracked_star_idx is not None:
            center = list(self._default_camera['center'])
        else:
            center = list(self.view.camera.center)
        return {
            'center': center,
            'azimuth': float(self.view.camera.azimuth),
            'elevation': float(self.view.camera.elevation),
            'distance': float(self.view.camera.distance),
        }

    def _start_camera_animation(
        self, target_azimuth=None, target_elevation=None, target_distance=None
    ):
        """Start a smooth camera transition to the current target.

        Args:
            target_azimuth: Target azimuth angle (None to not animate)
            target_elevation: Target elevation angle (None to not animate)
            target_distance: Target distance (None to not animate)
        """
        self._animating = True
        self._anim_start_time = time.time()
        self._anim_start_center = np.array(self.view.camera.center)
        self._anim_start_azimuth = self.view.camera.azimuth
        self._anim_start_elevation = self.view.camera.elevation
        self._anim_start_distance = self.view.camera.distance
        self._anim_target_azimuth = target_azimuth
        self._anim_target_elevation = target_elevation
        self._anim_target_distance = target_distance

    def _reset_camera(self):
        """Reset camera to initial/default position and stop tracking."""
        self._tracked_star_idx = None  # Stop tracking
        self.tracking_ring.visible = False
        self.tracked_star_text.visible = False
        # Get target camera properties
        if self.initial_camera is not None:
            target_azimuth = self.initial_camera.get(
                'azimuth', self._default_camera['azimuth']
            )
            target_elevation = self.initial_camera.get(
                'elevation', self._default_camera['elevation']
            )
            target_distance = self.initial_camera.get(
                'distance', self._default_camera['distance']
            )
        else:
            target_azimuth = self._default_camera['azimuth']
            target_elevation = self._default_camera['elevation']
            target_distance = self._default_camera['distance']
        # Start animation toward default/initial state
        self._start_camera_animation(target_azimuth, target_elevation, target_distance)

    def _update_scatter(self):
        """Update star positions from shared memory."""
        positions = self.shared_state.positions.copy()

        # Set marker properties (colored or white based on mode)
        if self._color_mode:
            face_color = self._star_colors
        else:
            # White mode, but keep black hole invisible (alpha=0)
            white_colors = np.ones_like(self._star_colors)
            if self._has_black_hole:
                white_colors[self._black_hole_mask, 3] = 0.0
            face_color = white_colors

        self.scatter.set_data(
            positions,
            edge_width=0,
            face_color=face_color,
            size=self._star_sizes,
        )

        # Update black hole visual (hollow ring)
        if self._has_black_hole:
            bh_positions = positions[self._black_hole_mask]
            self.black_hole_visual.set_data(
                pos=bh_positions,
                symbol='ring',
                size=STAR_SIZE * 2.5,
                edge_width=2,
                edge_color='white',
                face_color=(0, 0, 0, 0),  # Transparent center
            )

    def _update_stats(self):
        """Update the stats text overlay."""
        status = "PAUSED" if self.shared_state.paused else "RUNNING"
        sim_time = self.shared_state.sim_time
        num_stars = self.shared_state.num_stars
        steps_per_sec = self.shared_state.steps_per_second
        initial_energy = self.shared_state.initial_energy
        current_energy = self.shared_state.current_energy
        com = self.shared_state.center_of_mass
        unbound = self.shared_state.unbound_count

        # Compute energy deviation
        if initial_energy != 0.0:
            energy_deviation = (
                (current_energy - initial_energy) / abs(initial_energy) * 100
            )
            energy_str = f"Energy deviation: {energy_deviation:+.6f}%"
        else:
            energy_str = "Energy deviation: ---"

        stats = (
            f"Stars: {num_stars}  |  Time: {sim_time:.3f}  |  {status}\n"
            f"Steps/s: {steps_per_sec:.0f}  |  {energy_str}\n"
            f"Unbound: {unbound}  |  CoM: ({com[0]:+.3f}, {com[1]:+.3f}, {com[2]:+.3f})"
        )
        self.stats_text.text = stats

    def _update_highlight(self, mouse_pos):
        """Update the highlight ring to show the star closest to the mouse."""
        positions = self.shared_state.positions
        canvas_size = self.canvas.size
        n = len(positions)

        # Project stars to screen coordinates using full view-projection pipeline
        try:
            camera = self.view.camera

            # Step 1: View matrix (world to camera space)
            # camera.transform maps camera space -> world space, so we need inverse
            view_matrix = np.linalg.inv(camera.transform.matrix)

            # Step 2: Projection matrix (camera space to clip space)
            # Transposed for row-vector multiplication: point @ matrix
            fov_rad = np.radians(camera.fov)
            aspect = canvas_size[0] / canvas_size[1]
            f = 1.0 / np.tan(fov_rad / 2)
            near, far = 0.01, 1000.0

            projection = np.array(
                [
                    [f / aspect, 0, 0, 0],
                    [0, f, 0, 0],
                    [0, 0, (far + near) / (near - far), -1],
                    [0, 0, 2 * far * near / (near - far), 0],
                ]
            )

            # Step 3: Combined view-projection matrix
            vp_matrix = view_matrix @ projection

            # Step 4: Transform positions to clip space
            pos_4d = np.column_stack([positions, np.ones(n)])
            clip = pos_4d @ vp_matrix

            # Step 5: Perspective divide to get NDC
            w = clip[:, 3:4]
            w = np.where(np.abs(w) < 1e-10, 1e-10, w)
            ndc = clip[:, :3] / w

            # Step 6: Viewport transform: NDC (-1,1) to screen pixels
            screen_x = (ndc[:, 0] + 1) / 2 * canvas_size[0]
            screen_y = (1 - ndc[:, 1]) / 2 * canvas_size[1]

            # Compute 2D distance from each star to mouse position
            dx = screen_x - mouse_pos[0]
            dy = screen_y - mouse_pos[1]
            distances = np.sqrt(dx * dx + dy * dy)

            # Find closest star
            closest_idx = np.argmin(distances)
            min_dist = distances[closest_idx]

            # Highlight if within 50 pixel threshold
            if min_dist < 50:
                self._highlighted_star_idx = closest_idx
                self.highlight_ring.set_data(
                    pos=positions[closest_idx : closest_idx + 1],
                    size=STAR_SIZE * 2.5,
                    edge_width=2,
                    edge_color="yellow",
                    face_color=(0, 0, 0, 0),
                )
                self.highlight_ring.visible = True

                # Show star info near the highlighted star
                idx = closest_idx
                pos = positions[idx]
                vel = self.shared_state.velocities[idx]
                mass = self.shared_state.masses[idx]
                temp = self.shared_state.temperatures[idx]
                mag = self.shared_state.magnitudes[idx]

                info_text = (
                    f"Star #{idx}\n"
                    f"Mass: {mass:.3f}\n"
                    f"Temp: {temp:.0f} K\n"
                    f"Mag: {mag:.2f}\n"
                    f"Pos: ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})\n"
                    f"Vel: ({vel[0]:+.2f}, {vel[1]:+.2f}, {vel[2]:+.2f})"
                )
                self.star_info_text.text = info_text
                # Position text offset from star (20 pixels right and down)
                text_x = screen_x[idx] + 20
                text_y = screen_y[idx] + 20
                self.star_info_text.pos = (text_x, text_y)
                self.star_info_text.visible = True
            else:
                self._highlighted_star_idx = None
                self.highlight_ring.visible = False
                self.star_info_text.visible = False
        except Exception:
            # If transform fails, silently disable highlighting
            self._highlighted_star_idx = None
            self.highlight_ring.visible = False
            self.star_info_text.visible = False

    def _set_cursor_visible(self, visible: bool):
        """Set cursor visibility (for fullscreen mode)."""
        if self._cursor_visible == visible:
            return
        self._cursor_visible = visible
        try:
            if hasattr(self.canvas.native, 'set_mouse_visible'):
                # pyglet backend
                self.canvas.native.set_mouse_visible(visible)
            elif _USE_GLFW:
                # glfw backend - window handle is in canvas.native._id
                cursor_mode = glfw.CURSOR_NORMAL if visible else glfw.CURSOR_HIDDEN
                glfw.set_input_mode(self.canvas.native._id, glfw.CURSOR, cursor_mode)
        except Exception:
            pass

    def _on_mouse_move(self, event):
        """Handle mouse movement - show cursor in fullscreen, track position."""
        if self._is_fullscreen:
            self._cursor_last_move = time.time()
            if not self._cursor_visible:
                self._set_cursor_visible(True)

        # Store mouse position for per-frame highlight updates
        if event.pos is not None:
            self._last_mouse_pos = event.pos

    def _on_mouse_press(self, event):
        """Handle mouse press - right-click to track highlighted star."""
        if event.button == 2:  # Right mouse button
            if self._highlighted_star_idx is not None:
                self._tracked_star_idx = self._highlighted_star_idx
                self._start_camera_animation()
                print(f"Tracking star {self._tracked_star_idx}")

    def _on_timer(self, event):
        """Timer callback for updating visualization."""
        if not self.shared_state.running:
            self.close()
            return

        self._update_scatter()
        self._update_stats()

        # Determine target camera center
        positions = self.shared_state.positions
        if self._tracked_star_idx is not None and self._tracked_star_idx < len(
            positions
        ):
            idx = self._tracked_star_idx
            target_center = positions[idx]
            # Update tracking ring position
            self.tracking_ring.set_data(
                pos=positions[idx : idx + 1],
                size=STAR_SIZE * 3,
                edge_width=2,
                edge_color="cyan",
                face_color=(0, 0, 0, 0),
            )
            self.tracking_ring.visible = True

            # Update tracked star info text
            pos = positions[idx]
            vel = self.shared_state.velocities[idx]
            mass = self.shared_state.masses[idx]
            temp = self.shared_state.temperatures[idx]
            mag = self.shared_state.magnitudes[idx]

            info_text = (
                f"Tracking Star #{idx}\n"
                f"Mass: {mass:.3f}  Temp: {temp:.0f} K  Mag: {mag:.2f}\n"
                f"Pos: ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})\n"
                f"Vel: ({vel[0]:+.2f}, {vel[1]:+.2f}, {vel[2]:+.2f})"
            )
            self.tracked_star_text.text = info_text
            self.tracked_star_text.visible = True

            # Accumulate trajectory position with dimmed star color
            self._trajectory_positions.append(pos.copy())
            star_color = self._star_colors[idx].copy()
            star_color[:3] *= 0.5  # Dim the RGB channels
            self._trajectory_colors.append(star_color)

            # Update trajectory markers visual
            if self._trajectory_positions and self._trajectory_visible:
                self.trajectory_markers.set_data(
                    pos=np.array(self._trajectory_positions),
                    face_color=np.array(self._trajectory_colors),
                    symbol='disc',
                    size=2,
                    edge_width=0,
                )
                self.trajectory_markers.visible = True
        else:
            # Not tracking - target is default/initial center
            if self.initial_camera is not None and 'center' in self.initial_camera:
                target_center = np.array(self.initial_camera['center'])
            else:
                target_center = np.array(self._default_camera['center'])
            self.tracking_ring.visible = False
            self.tracked_star_text.visible = False

        # Update camera (with animation if active)
        if self._animating:
            elapsed = time.time() - self._anim_start_time
            t = min(1.0, elapsed / self._anim_duration)
            # Smoothstep easing: slow start, fast middle, slow end
            t = t * t * (3 - 2 * t)
            # Interpolate center
            new_center = self._anim_start_center + t * (
                target_center - self._anim_start_center
            )
            self.view.camera.center = tuple(new_center)
            # Interpolate other properties if targets are set
            if self._anim_target_azimuth is not None:
                new_azimuth = self._anim_start_azimuth + t * (
                    self._anim_target_azimuth - self._anim_start_azimuth
                )
                self.view.camera.azimuth = new_azimuth
            if self._anim_target_elevation is not None:
                new_elevation = self._anim_start_elevation + t * (
                    self._anim_target_elevation - self._anim_start_elevation
                )
                self.view.camera.elevation = new_elevation
            if self._anim_target_distance is not None:
                new_distance = self._anim_start_distance + t * (
                    self._anim_target_distance - self._anim_start_distance
                )
                self.view.camera.distance = new_distance
            # End animation when complete
            if t >= 1.0:
                self._animating = False
        elif self._tracked_star_idx is not None:
            # Live tracking (no animation) - follow the star directly
            self.view.camera.center = tuple(target_center)

        # Update highlight every frame (stars are moving)
        if self._last_mouse_pos is not None:
            self._update_highlight(self._last_mouse_pos)

        self.canvas.update()

        # Auto-hide cursor in fullscreen after idle period
        if self._is_fullscreen and self._cursor_visible:
            if time.time() - self._cursor_last_move >= self._cursor_hide_delay:
                self._set_cursor_visible(False)

    def _on_key_press(self, event):
        """Handle keyboard input."""
        if event.key == 'Q':
            # Quit - just set the flag, timer will handle the close
            print("Quit requested...")
            self.shared_state.running = False

        elif event.key == 'P':
            # Toggle pause
            self.shared_state.paused = not self.shared_state.paused
            status = "PAUSED" if self.shared_state.paused else "RUNNING"
            print(f"Physics: {status}")

        elif event.key == 'S':
            # Save state
            print("Saving simulation state...")
            camera_state = self._get_camera_state()
            try:
                filepath = save_simulation_state(self.shared_state, camera_state)
                print(f"Saved to: {filepath}")
            except Exception as e:
                print(f"Save failed: {e}")

        elif event.key == 'R':
            # Reset camera
            self._reset_camera()
            print("Camera reset")

        elif event.key == 'F':
            # Toggle fullscreen using local state tracking
            self._is_fullscreen = not self._is_fullscreen

            if _USE_GLFW:
                # glfw backend (macOS) - use maximize/restore instead of fullscreen
                # The actual glfw window handle is in canvas.native._id
                window = self.canvas.native._id
                if self._is_fullscreen:
                    glfw.maximize_window(window)
                    print(
                        "Fullscreen not supported on this platform, window maximized instead"
                    )
                else:
                    glfw.restore_window(window)
                    print("Window restored")
            else:
                # pyglet backend - use native fullscreen
                self.canvas.fullscreen = self._is_fullscreen

            # Move cursor to center to avoid triggering edge UI elements
            try:
                w, h = self.canvas.size
                if hasattr(self.canvas.native, 'set_mouse_position'):
                    self.canvas.native.set_mouse_position(w // 2, h // 2)
                elif _USE_GLFW:
                    glfw.set_cursor_pos(self.canvas.native._id, w // 2, h // 2)
            except Exception:
                pass
            if self._is_fullscreen:
                # Entering fullscreen/maximized: hide cursor immediately
                self._cursor_last_move = time.time() - self._cursor_hide_delay
                self._set_cursor_visible(False)
            else:
                # Exiting fullscreen/maximized: show cursor
                self._set_cursor_visible(True)

        elif event.key == 'C':
            # Toggle color mode
            self._color_mode = not self._color_mode

        elif event.key == 'H':
            # Toggle help overlay
            self._help_visible = not self._help_visible
            self.help_bg.visible = self._help_visible
            self.help_text.visible = self._help_visible

        elif event.key == 'T':
            # Toggle trajectory visibility
            self._trajectory_visible = not self._trajectory_visible
            if self._trajectory_positions:
                self.trajectory_markers.visible = self._trajectory_visible
            status = "visible" if self._trajectory_visible else "hidden"
            print(f"Trajectory: {status}")

        elif event.key == 'X':
            # Clear trajectory data
            self._trajectory_positions.clear()
            self._trajectory_colors.clear()
            self.trajectory_markers.visible = False
            print("Trajectory cleared")

    def close(self):
        """Close the visualizer."""
        self.timer.stop()
        self.canvas.close()
        app.quit()

    def run(self):
        """Run the visualization event loop."""
        app.run()


def run_visualization(
    shared_state: SharedState,
    initial_camera: Optional[Dict[str, Any]] = None,
    brightness_floor: float = BRIGHTNESS_FLOOR,
):
    """
    Run the visualization (main entry point).

    Args:
        shared_state: SharedState object for reading simulation data
        initial_camera: Optional initial camera state from loaded file
        brightness_floor: Minimum brightness for dimmest stars (0.0-1.0)
    """
    visualizer = StarClusterVisualizer(shared_state, initial_camera, brightness_floor)
    visualizer.run()
