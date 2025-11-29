# ruff: noqa: N806
import re

import click
import numpy as np
import serial
import serial.tools.list_ports
from vispy import app, scene
from vispy.scene import visuals

from archimedes.spatial import euler_to_dcm


def _get_usb_device():
    ports = serial.tools.list_ports.comports()

    for port in ports:
        # Filter by USB vendor/product ID or description
        if port.vid is not None:  # Has USB vendor ID (indicates USB device)
            print(f"Found USB device: {port.device} - {port.description}")
            return port.device

    raise FileNotFoundError("No USB device found")


def parse_imu_data(line):
    """
    Parse the format: "Roll: X  Pitch: Y  Yaw: Z\r\n"
    where X, Y, Z are integers in millidegrees.

    Returns: (rotation_matrix, roll_deg, pitch_deg, yaw_deg)
    """
    try:
        # Parse using regex to extract the three integers
        match = re.search(
            r"Roll:\s*(-?\d+)\s+Pitch:\s*(-?\d+)\s+Yaw:\s*(-?\d+)", line.decode("utf-8")
        )

        if match:
            # Convert millidegrees to degrees
            roll_deg = int(match.group(1)) / 1000.0
            pitch_deg = int(match.group(2)) / 1000.0
            yaw_deg = int(match.group(3)) / 1000.0

            # Convert to radians for DCM calculation
            roll_rad = np.radians(roll_deg)
            pitch_rad = np.radians(pitch_deg)
            yaw_rad = np.radians(yaw_deg)

            # Get rotation matrix
            R = euler_to_dcm([roll_rad, pitch_rad, yaw_rad]).T

            return R, roll_deg, pitch_deg, yaw_deg
        else:
            return None, None, None, None

    except Exception as e:
        print(f"Parse error: {e}")
        return None, None, None, None


class CoordinateFrame:
    def __init__(self, parent, arrow_length=1.0):
        self.arrow_length = arrow_length
        self.arrows = []
        self.labels = []
        colors = [
            (1, 0, 0, 1),  # Red for X
            (0, 1, 0, 1),  # Green for Y
            (0, 0, 1, 1),
        ]  # Blue for Z

        label_texts = ["X", "Y", "Z"]
        label_colors = ["red", "green", "blue"]

        # Initial directions
        self.directions = [
            np.array([arrow_length, 0, 0]),
            np.array([0, arrow_length, 0]),  # y-axis (right)
            np.array([0, 0, arrow_length]),  # z-axis (down)
        ]

        for i, (color, direction, text, text_color) in enumerate(
            zip(colors, self.directions, label_texts, label_colors)
        ):
            # Create arrow as cylinder (shaft) + cone (head)
            arrow = visuals.Arrow(
                pos=np.array([[0, 0, 0], direction]),
                color=color,
                arrow_size=300,
                width=3.0,
                arrow_type="stealth",
                parent=parent,
            )
            self.arrows.append(arrow)

            # Create text label at the end of each arrow
            label = scene.visuals.Text(
                text,
                pos=direction * 1.15,
                color=text_color,
                font_size=80,
                parent=parent,
            )
            self.labels.append(label)

    def update(self, rotation_matrix):
        """Update arrow orientations based on rotation matrix."""
        for arrow, label, base_dir in zip(self.arrows, self.labels, self.directions):
            rotated_dir = rotation_matrix @ base_dir

            # Flip y- and z-axes for forward-right-down convention
            rotated_dir[1:3] = -rotated_dir[1:3]

            arrow.set_data(pos=np.array([[0, 0, 0], rotated_dir]))

            # Update label position to follow arrow tip
            label.pos = rotated_dir * 1.15


@click.command()
@click.option(
    "--port", default=None, help="Serial port (e.g., /dev/tty.usbmodem141303)"
)
@click.option("--save", default=None, help="Filename to save data as CSV")
def main(port=None, save=None):
    if port is None:
        # Determine the USB port by looking for /dev/tty.usbmodem.*
        port = _get_usb_device()

    with serial.Serial(port, 115200, timeout=5) as ser:
        # Create canvas
        canvas = scene.SceneCanvas(
            keys="interactive", size=(1200, 800), show=True, bgcolor="white"
        )
        view = canvas.central_widget.add_view()
        view.camera = "turntable"
        view.camera.fov = 45
        view.camera.distance = 8

        # Create arrow visuals for X, Y, Z axes
        arrow_length = 2.0
        coord_frame = CoordinateFrame(view.scene, arrow_length=arrow_length)

        # Add text display for RPY angles (in screen space)
        rpy_display = scene.visuals.Text(
            "Roll: 0.0°  Pitch: 0.0°  Yaw: 0.0°",
            pos=(canvas.size[0] // 2, 50),
            color="black",
            font_size=16,
            parent=canvas.scene,
        )

        # Timer callback for serial reading
        def update(event):
            """Called at to read serial data and update visualization."""
            try:
                # Read ALL available data and only use the most recent
                # This clears out any backlog in the serial buffer
                latest_line = None
                if ser.in_waiting > 0:
                    while ser.in_waiting > 0:
                        latest_line = ser.readline()

                else:
                    latest_line = ser.readline()

                if latest_line:
                    R, roll, pitch, yaw = parse_imu_data(latest_line)

                    if R is not None:
                        # Update coordinate frame orientation
                        coord_frame.update(R)

                        # Update RPY text display
                        rpy_display.text = (
                            f"Roll: {roll:6.1f}°  "
                            f"Pitch: {pitch:6.1f}°  "
                            f"Yaw: {yaw:6.1f}°"
                        )

                        # Force canvas update
                        canvas.update()
            except Exception as e:
                print(f"Error: {e}")

        # Create timer for 100 Hz updates (10ms interval)
        app.Timer(interval=0.01, connect=update, start=True)

        # Optional: add keyboard shortcuts
        @canvas.events.key_press.connect
        def on_key_press(event):
            if event.key == "r":
                view.camera.reset()
            elif event.key == "q":
                app.quit()

        print("Starting VisPy IMU visualization...")
        print("Press 'r' to reset camera, 'q' to quit")

        # Run the application
        app.run()


if __name__ == "__main__":
    main()
