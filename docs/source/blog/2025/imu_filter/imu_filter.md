---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: archimedes
---

# Complementary IMU Filter

**_An end-to-end example of simple sensor fusion with Archimedes_**

Jared Callaham • 25 Nov 2025

:::{note}
This is the first post in a series that will document the process of building a high-performance quadrotor drone from scratch, highlighting how Archimedes can be used to "close the loop" with simulation, HIL testing, and deployment. [Subscribe](https://jaredcallaham.substack.com/embed) to the mailing list to see the progress!
:::

---

<!-- This post is supposed to:

- Demonstrate a clean development/deployment workflow
- Serve as a reference point for more advanced projects in the future

-->

One of the most basic tasks in a flight control system is _attitude estimation_ - how is the vehicle oriented in the world?
The difficulty with this is that it's not directly measurable, and available sensors have noise, bias, and drift.
The solution is _sensor fusion_, combining data from multiple sources to get a better estimate than from any one sensor.

Here we'll walk through a workflow for developing one of the simplest sensor fusion algorithms: a _complementary IMU filter_, with the goal of showing an end-to-end example of developing an algorithm in Python, deploying it, and getting streaming data back into Python.
In all honesty, for an algorithm this simple you're probably better off just writing the C code yourself (in this case I spent much more time writing the sensor driver and communication layer than the filter itself).
On the other hand, it can be easier to develop, test, maintain, and reuse Python code, especially for more complicated algorithms and physics models.
What we'll see here is a replicable _workflow_ that can scale to more complex systems and algorithms.

The primary goal of this post is to show a relatively simple and self-contained example of the "write in Python, deploy in C" workflow. In this particular case the logic is simple enough that it would realistically be more efficient to just write the whole thing in C, but the general process can be replicated for much more complicated algorithms. A second goal of the post is to introduce the "drone build" project described at the top, which will be a recurring theme in several upcoming posts and will illustrate a number of different aspects of the develop/deploy cycle in Archimedes.

This will be a quick overview that assumes familiarity with basic Archimedes functionality, including C code generation.
If you're new to Archimedes, check out these guides for background info:

- [Getting Started](../../../getting-started.md)
- [Codegen Tutorial](../../../tutorials/codegen/codegen00.md)
- [Hardware Deployment Tutorial](../../../tutorials/deployment/deployment00.md)

## How the complementary filter works

The idea of the complementary filter is to combine two imperfect sensor data streams into one improved estimate.
A typical IMU has a 3-axis accelerometer and a 3-axis gyroscope, each of which can be used to estimate the pitch and roll angles of a body:

* The gyroscope gives angular velocity measurements, which can be integrated to retrieve attitude
* The accelerometer reading includes the gravity vector, which tells you how far away from vertical you are

However, gyroscopes tend to have low-frequency drift, while accelerometers have high-frequency noise.
A simple way to combine the two estimates is by weighting with some value $\alpha \in (0, 1)$:

```python
att_fused = alpha * att_gyro + (1 - alpha) att_accel
```

I won't show it here, but this acts like a high-pass filter on the gyro measurements (filtering drift) and a low-pass filter on the accelerometer (filtering noise).

This is conceptually simple enough - the hard part is calculating the two attitude estimates.
This just involves some trigonometry for the accelerometer tilt angles and quaternion/Euler kinematics for the gyro integration.
Here we'll use quaternion kinematics for stability and also output the redundant Euler angles (e.g. for feedback attitude control).
Technically, quaternions can't be linearly interpolated and we should be using [spherical linear interpolation (SLERP)](https://en.wikipedia.org/wiki/Slerp), but at reasonably fast sampling rates this is probably not worth the extra complexity.
These calculations are standard and not very illuminating for our purposes, so let's just skip to the implementation.

:::{dropdown}  **Python Implementation**

```python
import numpy as np
import archimedes as arc


@arc.struct
class Attitude:
    q: np.ndarray
    rpy: np.ndarray


def quaternion_derivative(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 0.5 * np.hstack([
        -q[1] * w[0] - q[2] * w[1] - q[3] * w[2],
         q[0] * w[0] + q[2] * w[2] - q[3] * w[1],
         q[0] * w[1] - q[1] * w[2] + q[3] * w[0],
         q[0] * w[2] + q[1] * w[1] - q[2] * w[0]
    ])


def quat_from_accel(accel: np.ndarray, yaw: float) -> np.ndarray:
    # Normalize accelerometer vector
    accel = accel / np.linalg.norm(accel)
    ax, ay, az = accel

    # Calculate pitch and roll from accelerometer
    roll = np.arctan2(-ay, -az)
    pitch = np.arctan2(ax, np.sqrt(ay * ay + az * az))

    # Create quaternion from roll, pitch, yaw
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q_w = cr * cp * cy + sr * sp * sy
    q_x = sr * cp * cy - cr * sp * sy
    q_y = cr * sp * cy + sr * cp * sy
    q_z = cr * cp * sy - sr * sp * cy

    return np.hstack([q_w, q_x, q_y, q_z])


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    # Roll
    sinr_cosp = 2.0 * (q[0] * q[1] + q[2] * q[3])
    cosr_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch
    sinp = 2.0 * (q[0] * q[2] - q[3] * q[1])
    pitch = np.where(
        abs(sinp) >= 1.0,
        np.sign(sinp) * (np.pi / 2),
        np.arcsin(sinp)
    )

    # Yaw
    siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.hstack([roll, pitch, yaw])


def cfilter(att: Attitude, gyro: np.ndarray, accel: np.ndarray, alpha: float, dt: float) -> Attitude:
    # Integrate gyro to update quaternion
    qdot = quaternion_derivative(att.q, gyro)

    q_gyro = att.q + qdot * dt
    q_gyro = q_gyro / np.linalg.norm(q_gyro)
    
    # Estimate quaternion from accelerometer (use current yaw)
    q_accel = quat_from_accel(accel, att.rpy[2])
    q_accel = q_accel / np.linalg.norm(q_accel)

    # Complementary filter
    q_fused = alpha * q_gyro + (1 - alpha) * q_accel

    rpy = quaternion_to_euler(q_fused)
    return Attitude(q_fused, rpy)
```

If you want to see what the same algorithm looks like in handwritten C, there's also an implementation in the [source code](https://github.com/PineTreeLabs/archimedes/tree/main/docs/source/blog/2025/imu_filter/stm32/Core/Inc/hand_cfilter.h) on GitHub.
For something simple like the C code is really no harder to read or write than the Python, though the workflow we're illustrating here scales to more complicated algorithms where development, testing, and code reuse can be easier in Python.
:::

This uses a "scratch" implementation to keep the example self-contained, though we could also have used the [sptial](../spatial.md) module here to do the attitude conversions and quaternion kinematics.

## Python -> C code

We can generate C code from the Archimedes version using some default values as follows:

```python
# Initial values (will be overwritten by the runtime)
q = np.array([1, 0, 0, 0])
rpy = np.zeros(3)
gyro = np.zeros(3)
accel = np.array([0, 0, -1])  # In g's

# Can override these defaults from the runtime
alpha = 0.98
dt = 0.01

# Codegen
args = (att, gyro, accel, alpha, dt)
return_names = ("att_fused",)
arc.codegen(
    cfilter, args, return_names=return_names, output_dir="archimedes"
)
```

As discussed at length in the [codegen](../../../tutorials/codegen/codegen00.md) and [hardware deployment](../../../tutorials/deployment/deployment00.md) tutorial series, this generates a folder with the following structure:

```
archimedes
├── cfilter_kernel.c
├── cfilter_kernel.h
├── cfilter.c
└── cfilter.h
```

The low-level numerics are in the "kernel" files, but the only one you should need to look at is `cfilter.h`, which contains the "API" for the generated code:

```c
typedef struct {
    float q[4];
    float rpy[3];
} attitude_t;

// Input arguments struct
typedef struct {
    attitude_t att;
    float gyro[3];
    float accel[3];
    float alpha;
    float dt;
} cfilter_arg_t;

// Output results struct
typedef struct {
    attitude_t att_fused;
} cfilter_res_t;

// Workspace struct
typedef struct {
    long int iw[cfilter_SZ_IW];
    float w[cfilter_SZ_W];
} cfilter_work_t;

// Runtime API
int cfilter_init(cfilter_arg_t* arg, cfilter_res_t* res, cfilter_work_t* work);
int cfilter_step(cfilter_arg_t* arg, cfilter_res_t* res, cfilter_work_t* work);
```

In our `main.c` runtime, we'll have to declare the top-level structs, initialize them, and then call them from within the main loop.
It will look something like this:

```c
cfilter_arg_t cfilter_arg;
cfilter_res_t cfilter_res;
cfilter_work_t cfilter_w;

int main(void)
{
    // Initialize filter
    cfilter_arg.dt = DT_IMU;
    cfilter_arg.alpha = 0.98f;
    cfilter_init(&cfilter_arg, &cfilter_res, &cfilter_w);

    while(1)
    {
        if (imu_data_ready) {  // Set by timed callback
            imu_read(&imu_dev, &imu_data);
            
            // Move the sensor data to filter inputs
            for (int i=0; i<3; i++) {
                cfilter_arg.gyro[i] = imu_data.gyro[i];
                cfilter_arg.accel[i] = imu_data.accel[i];
            }

            // Call the filter function
            cfilter_step(&cfilter_arg, &cfilter_res, &cfilter_w);

            // Copy the estimated attitude back to the inputs for the next iteration
            cfilter_arg.att = cfilter_res.att_fused;

            imu_data_ready = false;

            // Optional: write to serial for streaming visualization
            printf("Roll: %d  Pitch: %d  Yaw: %d\r\n",
                (int)(1000 * cfilter_res.att_fused.rpy[0]*57.3f),
                (int)(1000 * cfilter_res.att_fused.rpy[1]*57.3f),
                (int)(1000 * cfilter_res.att_fused.rpy[2]*57.3f));
        }
    }
}
```

It's equally important to note what this auto-generated code _doesn't_ do:

- MCU and peripheral configuration (clocks, pins, interrupts)
- HAL function calls
- Communication protocols (SPI, I2C, CAN)
- Device drivers (which registers do we read/write?)

Archimedes generates code for the mathematical algorithm and leaves the embedded implementation details to you.

For me, the low-level embedded configuration is still the hard part, but the good news is that this is mostly a one-time cost.  If your drivers and communication functionality are properly abstracted from the controller logic, once the integrates system is working you can tinker with the algorithms and re-deploy into your runtime with little-to-no changes to the C code.

<!-- 
- TODO: Add a figure for workflow
- Real-time streaming
 -->

---

:::{admonition} About the Author
:class: blog-author-bio

**Jared Callaham** is the creator of Archimedes and principal at Pine Tree Labs.
He is a consulting engineer on modeling, simulation, optimization, and control systems with a particular focus on applications in aerospace engineering.

*Have questions or feedback? [Open a discussion on GitHub](https://github.com/jcallaham/archimedes/discussions)*
:::