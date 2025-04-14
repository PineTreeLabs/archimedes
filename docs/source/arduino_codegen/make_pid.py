import numpy as np
import archimedes as arc
from archimedes import struct


@arc.compile
def test_func(t, x, u):
    y, x_prev = x
    y = y + x_prev + 0.5 * u
    return y, (y, x_prev)


template_args = (0.0, np.zeros((2,)), np.array(0.0))

t = 1.0
x = np.array([2.0, 3.0])
u = 4.0
print(test_func(t, x, u))

arc.codegen(test_func, "test_func.c", template_args, header=True)



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


static_argnames = tuple(static_args.keys())

# Create template args for the function
x = np.zeros(3)
e = 0.0
template_args = (x, e)

arc.codegen(
    pid,
    "sketch/pid.c",
    template_args,
    kwargs=static_args,
    static_argnames=static_argnames,
    header=True,
    float_type=np.float32,
    int_type=np.int32,
)

x = np.array([1.0, 2.0, 3.0])
e = 0.5
x, u = pid(x, e, **static_args)
print(f"Updated state: {x}, Control output: {u}")