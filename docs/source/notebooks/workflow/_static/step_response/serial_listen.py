import serial
import csv
import numpy as np
import matplotlib.pyplot as plt

dt = 1e-4

def _recieve_single_stream(ser):
    # Wait for START message
    while True:
        line = ser.readline().decode().strip()
        if line.startswith('START'):
            parts = line.split(',')
            sample_count = int(parts[1])
            num_channels = int(parts[2])
            pwm_count = int(parts[3])
            sample_rate = int(parts[4])
            print(f"Receiving {sample_count} samples, {num_channels} channels, sample rate {sample_rate}")
            break
    
    # Collect data
    data = []
    samples_received = 0
    
    while samples_received < sample_count:
        line = ser.readline().decode().strip()
        if line == 'END':
            break
        
        try:
            values = [int(x) for x in line.split(',')]
            data.append(values)
            samples_received += 1
            
            if samples_received % 100 == 0:
                print(f"Received {samples_received}/{sample_count} samples")
                
        except ValueError:
            continue

    data = np.array(data, dtype=float)

    # Convert sample count to timesteps
    data[:, 0] = sample_rate * dt * data[:, 0]

    # pwm_duty -> [0-1]
    data[:, 1] /= pwm_count

    # mV -> volts, mA -> A, mdeg -> deg
    data[:, 2:] *= 1e-3

    return data

def receive_data(port='/dev/tty.usbmodem141303', baudrate=115200):
    """Receive and parse data from STM32"""
    
    # Step response data (200 ms)
    with serial.Serial(port, baudrate, timeout=5) as ser:
        print("Waiting for experiment data...")
        step_data = _recieve_single_stream(ser)

    # Ramp response (2s)
    with serial.Serial(port, baudrate, timeout=5) as ser:
        print("Waiting for experiment data...")
        ramp_data = _recieve_single_stream(ser)

    # ramp_data = None
    
    return step_data, ramp_data


def plot_response(data):
    fig, ax = plt.subplots(4, 1, figsize=(7, 6), sharex=True)
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


def save_data(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['t (s)', 'v (mV)', 'i (mA)', 'pos (mdeg)'])
        writer.writerows(data)


if __name__ == "__main__":
    step_data, ramp_data = receive_data()

    driver_id = 3

    # Save to CSV
    # timestamp = time.strftime("%Y%m%d_%H%M%S")
    # save_data(step_data, f"../data/step_data_{driver_id:02d}.csv")
    # save_data(ramp_data, f"../data/ramp_data_{driver_id:02d}.csv")

    plot_response(step_data)
    plot_response(ramp_data)
    
    