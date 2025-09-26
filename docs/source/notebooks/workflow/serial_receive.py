import click
import matplotlib.pyplot as plt
import numpy as np
import serial
import serial.tools.list_ports

dt = 1e-4


def _get_usb_device():
    ports = serial.tools.list_ports.comports()

    for port in ports:
        # Filter by USB vendor/product ID or description
        if port.vid is not None:  # Has USB vendor ID (indicates USB device)
            print(f"Found USB device: {port.device} - {port.description}")
            return port.device

    raise FileNotFoundError("No USB device found")


def _wait_for_start_message(ser):
    """Wait for and parse the START message."""
    while True:
        line = ser.readline().decode().strip()
        if line.startswith("START"):
            sample_count, pwm_count, sample_rate, loop_cycles = map(
                int, line.split(",")[1:]
            )
            print(f"Receiving {sample_count} samples at sample rate {sample_rate}")
            if loop_cycles > 0:
                # Divide by 180 MHz to get execution time in µs
                print(f"Control loop time: {loop_cycles / 180:.4f} µs")
            return sample_count, pwm_count, sample_rate


def _collect_raw_samples(ser, sample_count):
    """Collect raw data samples from serial connection."""
    data = []
    samples_received = 0

    while samples_received < sample_count:
        line = ser.readline().decode().strip()
        if line == "END":
            break

        try:
            values = [int(x) for x in line.split(",")]
            data.append(values)
            samples_received += 1

            if samples_received % 100 == 0:
                print(f"Received {samples_received}/{sample_count} samples")

        except ValueError:
            continue

    return data


def _receive_single_stream(ser):
    """Receive and parse a single data stream from serial connection."""

    # Parse START message
    start_info = _wait_for_start_message(ser)
    sample_count, pwm_count, sample_rate = start_info

    # Collect raw data
    raw_data = _collect_raw_samples(ser, sample_count)

    # Convert to numpy array and apply unit conversions
    return _process_raw_data(raw_data, sample_rate, pwm_count)


def _process_raw_data(raw_data, sample_rate, pwm_count):
    """Convert raw data to proper units and format."""
    data = np.array(raw_data, dtype=float)

    # Convert sample count to timesteps
    data[:, 0] = sample_rate * dt * data[:, 0]

    # PWM duty cycle to [0-1]
    data[:, 1] /= pwm_count

    # Unit conversions: mV -> V, mA -> A, mdeg -> deg
    data[:, 2:] *= 1e-3

    return data


@click.command()
@click.option(
    "--port", default=None, help="Serial port (e.g., /dev/tty.usbmodem141303)"
)
@click.option("--save", default=None, help="Filename to save data as CSV")
def main(port=None, save=None):
    if port is None:
        # Determine the USB port by looking for /dev/tty.usbmodem.*
        port = _get_usb_device()

    with serial.Serial(port, baudrate=115200, timeout=5) as ser:
        print("Waiting for experiment data...")
        data = _receive_single_stream(ser)

    # Save to CSV
    if save:
        filename = f"data/{save}"
        header = "t [s]\t\tu [-]\t\tv [V]\t\ti [A]\t\tpos [deg]"
        np.savetxt(
            filename, data, delimiter="\t", fmt="%.6f", header=header, comments=""
        )

    _fig, ax = plt.subplots(4, 1, figsize=(7, 6), sharex=True)
    ax[0].plot(data[:, 0], 100 * data[:, 1])
    ax[0].grid()
    ax[0].set_ylabel(r"PWM duty [%]")

    data[:, 2] = np.where(data[:, 2] > 1e6, -1, data[:, 2])  # Remove erroneous spikes

    ax[1].plot(data[:, 0], data[:, 2])
    ax[1].grid()
    ax[1].set_ylabel(r"Motor voltage [V]")
    # ax[0].set_ylim([0, 15])

    ax[2].plot(data[:, 0], data[:, 3])
    ax[2].grid()
    ax[2].set_ylabel(r"Motor current [A]")

    ax[3].plot(data[:, 0], data[:, 4])
    ax[3].grid()
    ax[3].set_ylabel(r"Position [deg]")

    ax[-1].set_xlabel("Time [s]")

    plt.show()


if __name__ == "__main__":
    main()
