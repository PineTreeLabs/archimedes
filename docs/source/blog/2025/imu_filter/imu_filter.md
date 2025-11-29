# Complementary IMU Filter

**_An end-to-end example of simple sensor fusion with Archimedes_**

Jared Callaham • 25 Nov 2025

:::{note}
This is the first post in a series that will document the process of building a high-performance quadrotor drone from scratch, highlighting how Archimedes can be used to "close the loop" with simulation, HIL testing, and deployment. [Subscribe](https://jaredcallaham.substack.com/embed) to the mailing list to see the progress!
:::

---

One of the most basic tasks in a flight control system is _attitude estimation_ - how is the vehicle oriented in the world?
The difficulty with this is that it's not directly measurable, and available sensors have noise, bias, and drift.
To handle this we do _sensor fusion_, combining data from multiple sources to get a better estimate than from any one sensor.

Here we'll walk through a workflow for developing one of the simplest sensor fusion algorithms: a _complementary IMU filter_, with the goal of showing an end-to-end example of developing an algorithm in Python, deploying it, and getting streaming data back into Python.

```{image} _static/streaming_imu.gif
```

The primary goal of this post is to show a relatively simple and self-contained example of this "write in Python, deploy in C" paradigm. In this particular case the logic is simple enough that it would realistically be more efficient to just write the whole thing in C, but the general process can be replicated for much more complicated algorithms. A second goal of the post is to introduce the "drone build" project, which will be a recurring theme in several upcoming posts and will illustrate a number of different aspects of the develop/deploy cycle in Archimedes.

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

:::{note}
Using the accelerometer to estimate tilt angles only makes sense when the IMU is held steady.
Hence, this sensor fusion approach won't work well in general, especially for aggressive flight maneuvers.
Still, it's a simple way to get end-to-end attitude estimation and we can always replace it with more sophisticated sensor fusion when needed.
:::

This filter is conceptually simple enough - the hard part is calculating the two attitude estimates.
This just involves some trigonometry for the accelerometer tilt angles and quaternion/Euler kinematics for the gyro integration.
Here we'll use quaternion kinematics for stability and also output the redundant Euler angles (e.g. for feedback attitude control).
<!-- 
Technically, quaternions can't be linearly interpolated and we should be using [spherical linear interpolation (SLERP)](https://en.wikipedia.org/wiki/Slerp), but at reasonably fast sampling rates this is probably not worth the extra complexity. -->
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

This uses a "scratch" implementation to keep the example self-contained, though we could also have used the [spatial](../spatial.md) module here to do the attitude conversions and quaternion kinematics.

Since we'll be autogenerating and deploying C code from this Python, we can also easily write [test and validation cases](https://github.com/PineTreeLabs/archimedes/tree/main/docs/source/blog/2025/imu_filter/test_cfilter.py) to make sure the algorithm is working as intended.
Combining Python-level unit testing with hardware-in-the-loop (HIL) testing of the integrated embedded controller can be a powerful way to test and validate complex codes.

In practice, you'd also typically validate this filter in simulation before deploying to hardware - which is easy to do, since the filter is written in Python.
Then the same logic that runs in simulation gets deployed to the STM32.
For this simple example I went straight to deployment, but for more complex algorithms (as we'll see in future posts), simulation is essential for debugging and parameter tuning before you flash firmware.

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

### Drivers not included

It's also important to note what this auto-generated code _doesn't_ do:

- MCU and peripheral configuration (clocks, pins, interrupts)
- HAL function calls
- Communication protocols (SPI, I2C, CAN)
- Device drivers (which registers do we read/write?)

Archimedes generates code for the mathematical algorithm and leaves the embedded implementation details to you.
For me, the low-level embedded is the hard part, but the good news is that this is mostly a one-time cost.  If your drivers and communication functionality are properly abstracted from the controller logic, once the integrates system is working you can tinker with the algorithms and re-deploy into your runtime with little-to-no changes to the C code.

The GitHub repo has a custom, portable [driver](https://github.com/PineTreeLabs/archimedes/tree/main/docs/source/blog/2025/imu_filter/stm32/Core/Inc/lsm6dsox.h) for the [LSM6DSOX](https://www.st.com/en/mems-and-sensors/lsm6dsox.html) IMU and a semi-portable (across STM32s) [SPI communication layer](https://github.com/PineTreeLabs/archimedes/tree/main/docs/source/blog/2025/imu_filter/stm32/Core/Inc/stm32_spi_dev.h).

## Board configuration

Since this is a prototype of a system ultimately intended to be a full flight computer, I built it on an STM32H753ZI Nucleo dev board.
The full STM32CubeMX configuration is available on the [GitHub repo](https://github.com/PineTreeLabs/archimedes/tree/main/docs/source/blog/2025/imu_filter/stm32/), but the basic configuration is:

- 480 MHz clock speed for the STM32
- SPI communication to the IMU chip at 10 MHz
- Configure the IMU for data rates up to 6.7 kHz
- Timed interrupt to match the IMU data rate
- USART for streaming data back to the computer
- Cycle counter for profiling the filter performance

For the streaming demo below I lowered the data rate to ~400 Hz so that the streaming output over USB wasn't likely to cause missed samples, but for the real thing I'm planning to push to the maximum sampling rate (and move any slow serial work to DMA).

## Streaming telemetry

With everything in place, the last step in this demo is a real-time visualization of the attitude estimate.
This is convenient (but definitely not necessary) for basic debugging (do your sensor axes match your vehicle coordinate system?), but also it's just generally nice to get the visual feedback.

I used [VisPy](https://vispy.org/) for this demo to get real-time 3D and pyserial to stream the data.
VisPy might be overkill for something like this, but the OpenGL backend makes it snappy for 3D rendering, and the interactivity is also very nice.
The full animation script is [here](https://github.com/PineTreeLabs/archimedes/tree/main/docs/source/blog/2025/imu_filter/stream3d.py)

```{image} _static/streaming_imu.gif
```

## Conclusion

The complementary filter is a rough-and-ready way to do attitude estimation that's not really suitable for aggressive maneuvers.
Still, at this point we can be fairly confident in the board configuration, driver implementation, peripheral communication, and overall runtime logic.
And because we've fully abstracted the control logic from all of these low-level embedded details, we could now move on to more sophisticated sensor fusion algorithms.

For the "build-a-drone-from-scratch" project, the first real milestone will be attitude/rate control for stable hovering, so the complementary filter is a totally reasonable place to start.
The next steps will be setting up the power distribution and motor controllers, building a simple gimbal for bench testing, and testing a stablization controller.
In these blog posts we'll also be exploring several other ways you can use Archimedes in the development process, including for controller design, parameter estimation, and low-cost DIY hardware-in-the-loop (HIL) testing.

For more on this project, including Archimedes release announcements, blog posts, application examples, and case studies, subscribe to the free newsletter:

<iframe src="https://jaredcallaham.substack.com/embed" width="480" height="320" style="border:1px solid #EEE; background:white;" frameborder="0" scrolling="no"></iframe>

### Try it out!

Zooming out, what we've seen here is a simple but complete example of how you can define an algorithm in Python, autogenerate C code, and deploy to a microcontroller.
Beyond the complementary IMU filter, you can apply this workflow to a wide range of control systems and algorithm types.

If you're interested in trying this out for yourself, you can:

- Check out the [source code](https://github.com/PineTreeLabs/archimedes/tree/main/docs/source/blog/2025/imu_filter/) for this post on GitHub
- Work through the [codegen](../../../tutorials/codegen/codegen00.md) and [hardware deployment](../../../tutorials/deployment/deployment00.md) tutorials
- [Post a discussion thread](https://github.com/pinetreelabs/archimedes/discussions) to share what you did

Community feedback on this process is invaluable, so if you do try it please consider sharing what you found: what worked (or didn't), what you liked (or didn't), and above all any bug reports.
All of this will help make Archimedes a more useful and reliable tool to help you build better.

Thanks for checking out Archimedes!

---

:::{admonition} About the Author
:class: blog-author-bio

**Jared Callaham** is the creator of Archimedes and principal at Pine Tree Labs.
He is a consulting engineer on modeling, simulation, optimization, and control systems with a particular focus on applications in aerospace engineering.

*Have questions or feedback? [Open a discussion on GitHub](https://github.com/pinetreelabs/archimedes/discussions)*
:::