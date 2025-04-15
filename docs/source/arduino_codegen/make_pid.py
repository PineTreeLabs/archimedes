import numpy as np
import archimedes as arc
from archimedes import struct



# These will be declared as "static", hardcoding their values
# in the generated C code.  We could also leave some of these
# out to have them be arguments to the generated function
static_args = {
    "Kp": 1.0,
    "Ki": 0.1,
    "Kd": 0.01,
    "Ts": 0.01,
    "N": 100.0,
}


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
x = np.zeros(3)
e = 0.0
template_args = (x, e)

# Compile the function with specified static arguments and return names
pid = arc.compile(
    pid,
    # static_argnames=tuple(static_args.keys()),
    return_names=("x_new", "u"),
)

template_config = {
    "output_path": "main.c",
    "input_descriptions": {
        "x": "State vector",
        "e": "Error signal (scalar)",
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
    kwargs=static_args,
    header=True,
    float_type=np.float32,
    int_type=np.int32,
    template="c",
    template_config=template_config,
)

x = np.array([1.0, 2.0, 3.0])
e = 0.5
x, u = pid(x, e, **static_args)
print(f"Updated state: {x}, Control output: {u}")


# Generate C driver code

# Example context for rendering main.c
context = {
    'driver_name': 'main',
    'function_name': 'pid',
    'float_type': 'float',
    'int_type': 'int',
    'inputs': [
        {
            'type': 'float', 
            'name': 'x', 
            'dims': '3', 
            'initial_value': '{1.0, 2.0, 3.0}', 
            'description': 'State vector',
            'is_addr': False,
        },
        {
            'type': 'float', 
            'name': 'e', 
            'dims': None, 
            'initial_value': '0.5', 
            'description': 'Error signal (scalar)',
            'is_addr': True,
        }
    ],
    'outputs': [
        {
            'type': 'float', 
            'name': 'x_new', 
            'dims': '3', 
            'description': 'Updated state',
            'is_addr': False,
        },
        {
            'type': 'float', 
            'name': 'u', 
            'dims': None, 
            'description': 'Output vector (scalar)',
            'is_addr': True,
        }
    ],
}

# # Render the template
# from archimedes._core._codegen._codegen import _render_c_driver
# _render_c_driver(context, "main.c")