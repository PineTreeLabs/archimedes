# Construct an IIR filter and write a `compile` stepping it forward
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import archimedes as arc

# 1. Design a Butterworth lowpass filter (offline step with SciPy)
fs = 1000  # Sample frequency in Hz
cutoff = 100  # Cutoff frequency in Hz
order = 4  # Filter order
b, a = signal.butter(order, cutoff / (fs / 2), "low")


# 2. Create a compiled function that implements the IIR filter
@arc.compile
def iir_filter_step(x, state, b_coef, a_coef):
    """
    Apply one step of an IIR filter using direct form II transposed.

    Parameters:
    x: float, current input sample
    state: array, filter state
    b_coef: array, filter numerator coefficients
    a_coef: array, filter denominator coefficients (a[0] should be 1.0)

    Returns:
    y: float, filtered output sample
    state_new: array, updated filter state
    """
    # Calculate output: y = b[0]*x + state[0]
    y = b_coef[0] * x + state[0]

    # Update state using direct form II transposed structure
    state_new = np.zeros_like(state)

    for i in range(len(state) - 1):
        state_new[i] = state[i + 1] + b_coef[i + 1] * x - a_coef[i + 1] * y

    # Handle the last state element
    if len(state) > 0:
        i = len(state) - 1
        if i + 1 < len(b_coef):
            state_new[i] = b_coef[i + 1] * x - a_coef[i + 1] * y
        else:
            state_new[i] = -a_coef[i + 1] * y

    return y, state_new


# Function to filter an entire signal using our IIR filter
def filter_signal(signal_data, b_coef, a_coef):
    # Initialize state array (size depends on filter order)
    n = max(len(a_coef), len(b_coef)) - 1
    state = np.zeros(n)
    output = np.zeros_like(signal_data)

    # Apply the filter step by step
    for i in range(len(signal_data)):
        output[i], state = iir_filter_step(signal_data[i], state, b_coef, a_coef)

    return output


# Create a test signal with mixed frequencies
t = np.linspace(0, 1, fs)
signal_data = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 150 * t)

# Apply our filter
filtered = filter_signal(signal_data, b, a)

# Compare with SciPy's lfilter (should be identical)
filtered_scipy = signal.lfilter(b, a, signal_data)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, signal_data, "b-", alpha=0.5, label="Original Signal")
plt.plot(t, filtered, "r-", label="Archimedes Filtered")
plt.plot(t, filtered_scipy, "g--", label="SciPy Filtered")
plt.legend()
plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("IIR Filter Comparison")
plt.tight_layout()
plt.show()
