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

---

One of the most basic tasks in a flight control system is _attitude estimation_ - how is the vehicle oriented in the world?
The difficulty with this is that it's not directly measurable, and available sensors have noise, bias, and drift.
The solution is _sensor fusion_, combining data from multiple sources to get a better estimate than from any one sensor.

Here we'll walk through a workflow for developing one of the simplest sensor fusion algorithms: a _complementary IMU filter_, with the goal of showing an end-to-end example of developing an algorithm in Python, deploying it, and getting streaming data back into Python.

This will be a quick overview that skims over the underlying principles of the algorithm and assumes familiarity with and basic Archimedes functionality, including C code generation.
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
Technically, quaternions can't be linearly interpolated and we should be using SLERP, but at fast sampling rates this is probably not worth the extra complexity.

These calculations are standard and not very illuminating for our purposes, so let's just skip to the implementation.
Our goal will be to just develop the algorithm in Python and deploy, but for the sake of comparison I'll show a side-by-side in handwritten C and Archimedes-compatible Python.

:::::{dropdown}  **Code Comparison**

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card}  **Handwritten C**

```c
/* Complementary filter for 3-axis accelerometer */

#ifndef CFILT_H
#define CFILT_H


static inline int quaternion_derivative(const float *q, const float *omega, float *q_dot) {
    q_dot[0] = 0.5f * (-q[1] * omega[0] - q[2] * omega[1] - q[3] * omega[2]);
    q_dot[1] =  0.5f * (q[0] * omega[0] + q[2] * omega[2] - q[3] * omega[1]);
    q_dot[2] =  0.5f * (q[0] * omega[1] - q[1] * omega[2] + q[3] * omega[0]);
    q_dot[3] =  0.5f * (q[0] * omega[2] + q[1] * omega[1] - q[2] * omega[0]);
    return 0;
}

static inline int quaternion_update(float *q, const float *omega, float dt) {
    float q_dot[4];
    quaternion_derivative(q, omega, q_dot);
    for (int i = 0; i < 4; i++) {
        q[i] += q_dot[i] * dt;
    }
    return 0;
}

static inline int quaternion_normalize(float *q) {
    float norm = 0.0f;
    for (int i = 0; i < 4; i++) {
        norm += q[i] * q[i];
    }
    norm = sqrtf(norm);
    if (norm > 0.0f) {
        for (int i = 0; i < 4; i++) {
            q[i] /= norm;
        }
    }
    return 0;
}

static inline int quaternion_from_accel(const float *accel, float *q_accel, float yaw) {
    float ax = accel[0];
    float ay = accel[1];
    float az = accel[2];

    // Normalize (set to 1g magnitude)
    float norm = sqrtf(ax * ax + ay * ay + az * az);
    ax /= norm;
    ay /= norm;
    az /= norm;

    float roll = atan2f(-ay, -az);
    float pitch = atan2f(ax, sqrtf(ay * ay + az * az));

    // Convert to quaternion
    float cy = cosf(yaw * 0.5f);
    float sy = sinf(yaw * 0.5f);
    float cr = cosf(roll * 0.5f);
    float sr = sinf(roll * 0.5f);
    float cp = cosf(pitch * 0.5f);
    float sp = sinf(pitch * 0.5f);

    q_accel[0] = cy * cr * cp + sy * sr * sp;
    q_accel[1] = cy * sr * cp - sy * cr * sp;
    q_accel[2] = cy * cr * sp + sy * sr * cp;
    q_accel[3] = sy * cr * cp - cy * sr * sp;

    return 0;
}


static inline int quaternion_to_euler(const float *q, float *euler) {
    // Roll (x-axis rotation)
    float sinr_cosp = 2.0f * (q[0] * q[1] + q[2] * q[3]);
    float cosr_cosp = 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]);
    euler[0] = atan2f(sinr_cosp, cosr_cosp);

    // Pitch (y-axis rotation)
    float sinp = 2.0f * (q[0] * q[2] - q[3] * q[1]);
    if (fabsf(sinp) >= 1)
        euler[1] = copysignf(M_PI / 2.0f, sinp); // use 90 degrees if out of range
    else
        euler[1] = asinf(sinp);

    // Yaw (z-axis rotation)
    float siny_cosp = 2.0f * (q[0] * q[3] + q[1] * q[2]);
    float cosy_cosp = 1.0f - 2.0f * (q[2] * q[2] + q[3] * q[3]);
    euler[2] = atan2f(siny_cosp, cosy_cosp);

    return 0;
}

static inline int cfilter(float *q, const float *gyro, const float *accel, float alpha, float dt) {
    // Calculate current yaw since the accel cannot correct
    float siny_cosp = 2.0f * (q[0] * q[3] + q[1] * q[2]);
    float cosy_cosp = 1.0f - 2.0f * (q[2] * q[2] + q[3] * q[3]);
    float yaw = atan2f(siny_cosp, cosy_cosp);

    quaternion_update(q, gyro, dt);
    quaternion_normalize(q);

    float q_accel[4];
    quaternion_from_accel(accel, q_accel, yaw); // Use current yaw for accel estimate
    quaternion_normalize(q_accel);

    for (int i = 0; i < 4; i++) {
        q[i] = alpha * q[i] + (1.0f - alpha) * q_accel[i];
    }

    return 0;
}

#endif // CFILT_H
```

:::

:::{grid-item-card} **Python**

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
:::
::::

Both of these are around 100 lines, so in a sense we don't really gain much by working in Python here.
Personally, I feel that it's easier to develop and test Python code, and easier to grow it to more complicated projects, but that's just personal preference.

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


<!-- 
- TODO: kHz+ sampling rates
- TODO: Add a figure for workflow
- Real-time streaming
- Hard part: drivers and board configuration (but one-time!)
 -->

---

:::{admonition} About the Author
:class: blog-author-bio

**Jared Callaham** is the creator of Archimedes and principal at Pine Tree Labs.
He is a consulting engineer on modeling, simulation, optimization, and control systems with a particular focus on applications in aerospace engineering.

*Have questions or feedback? [Open a discussion on GitHub](https://github.com/jcallaham/archimedes/discussions)*
:::