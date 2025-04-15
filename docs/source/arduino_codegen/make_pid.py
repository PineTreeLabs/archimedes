# ruff: noqa: N802, N803, N806, N815, N816

import numpy as np

import archimedes as arc


def pid(x, e, Kp=1.0, Ki=0.0, Kd=0.0, Ts=0.01, N=10.0):
    """Discrete-time PID controller with filtered derivative

    Args:
        x: Flattened state vector [e_prev, e_int, de_filt]
        e: Current error value
        Kp: Proportional gain
        Ki: Integral gain
        Kd: Derivative gain
        Ts: Sampling time in seconds
        N: Filter coefficient (larger means less filtering)

    Returns:
        (x_new, u): Updated state vector and control output
    """
    e_prev, e_int, e_dot = x

    # Calculate bilinear filter coefficients
    b = [2 * N, -2 * N]
    a = [(2 + N * Ts), (-2 + N * Ts)]

    # Update integral term using trapezoidal rule
    e_int = e_int + (e + e_prev) * Ts / 2.0

    # Apply bilinear IIR filter
    e_dot = (b[0] * e + b[1] * e_prev - a[1] * e_dot) / a[0]

    # PID control output (parallel form)
    u = Kp * e + Ki * e_int + Kd * e_dot

    # Updated state
    x_new = np.hstack([e, e_int, e_dot])

    return x_new, u


# Create template args for the function
# These are used to infer shape/dtype, but also used as the initial
# values in the generated code
x = np.array([1.0, 2.0, 3.0])
e = 0.5
template_args = (x, e)

# Keyword arguments for the function - will also generate named variables
# in the code
Ts = 0.01  # Sample time, seconds
kwargs = {
    "Kp": 1.0,
    "Ki": 0.1,
    "Kd": 0.01,
    "Ts": Ts,
    "N": 1 / Ts,
}

x, u = pid(x, e, **kwargs)
print(f"Updated state: {x}, Control output: {u}")

c_driver_config = {
    "output_path": "main.c",
    # Optional: descriptions add comments to the generated code
    "input_descriptions": {
        "x": "State vector",
        "e": "Error signal (scalar)",
        "Kp": "Proportional gain",
        "Ki": "Integral gain",
        "Kd": "Derivative gain",
        "Ts": "Sampling time in seconds",
        "N": "Filter coefficient",
    },
    "output_descriptions": {
        "x_new": "Updated state",
        "u": "Output vector (scalar)",
    },
}

# Specifying the return names is optional, but will create meaningful names
# in the generated code for the output variables.
arc.codegen(
    pid,
    "pid.c",
    template_args,
    return_names=("x_new", "u"),
    kwargs=kwargs,
    float_type=np.float32,
    int_type=np.int32,
    driver="c",
    driver_config=c_driver_config,
)


# Redo for arduino
arduino_driver_config = {
    "output_path": "arduino_codegen.ino",
    "sample_rate": Ts,
    # Optional: descriptions add comments to the generated code
    "input_descriptions": {
        "x": "State vector",
        "e": "Error signal (scalar)",
        "Kp": "Proportional gain",
        "Ki": "Integral gain",
        "Kd": "Derivative gain",
        "Ts": "Sampling time in seconds",
        "N": "Filter coefficient",
    },
    "output_descriptions": {
        "x_new": "Updated state",
        "u": "Output vector (scalar)",
    },
}

arc.codegen(
    pid,
    "pid.c",
    template_args,
    return_names=("x_new", "u"),
    kwargs=kwargs,
    float_type=np.float32,
    int_type=np.int32,
    driver="arduino",
    driver_config=arduino_driver_config,
)
