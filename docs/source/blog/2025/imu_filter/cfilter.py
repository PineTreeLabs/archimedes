import numpy as np

import archimedes as arc


@arc.struct
class Attitude:
    q: np.ndarray
    rpy: np.ndarray


def quaternion_derivative(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 0.5 * np.hstack(
        [
            -q[1] * w[0] - q[2] * w[1] - q[3] * w[2],
            q[0] * w[0] + q[2] * w[2] - q[3] * w[1],
            q[0] * w[1] - q[1] * w[2] + q[3] * w[0],
            q[0] * w[2] + q[1] * w[1] - q[2] * w[0],
        ]
    )


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
    pitch = np.where(abs(sinp) >= 1.0, np.sign(sinp) * (np.pi / 2), np.arcsin(sinp))

    # Yaw
    siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.hstack([roll, pitch, yaw])


def cfilter(
    att: Attitude, gyro: np.ndarray, accel: np.ndarray, alpha: float, dt: float
) -> Attitude:
    # Integrate gyro to update quaternion
    qdot = quaternion_derivative(att.q, gyro)

    q_gyro = att.q + qdot * dt
    q_gyro = q_gyro

    # Estimate quaternion from accelerometer (use current yaw)
    q_accel = quat_from_accel(accel, att.rpy[2])
    q_accel = q_accel

    # Complementary filter
    q_fused = alpha * q_gyro + (1 - alpha) * q_accel
    q_fused = q_fused / np.linalg.norm(q_fused)

    rpy = quaternion_to_euler(q_fused)
    return Attitude(q_fused, rpy)


if __name__ == "__main__":
    q = np.array([1, 0, 0, 0])
    rpy = np.array([0, 0, 0])
    gyro = np.array([0.01, 0.02, 0.03])
    accel = np.array([0, 0, -1])
    alpha = 0.98
    dt = 0.01

    att = Attitude(q, rpy)
    att_fused = cfilter(att, gyro, accel, alpha, dt)

    print("Fused Quaternion:", att_fused.q)
    print("Fused Euler Angles:", att_fused.rpy)

    # Codegen
    args = (att, gyro, accel, alpha, dt)
    return_names = ("att_fused",)
    output_dir = "stm32/archimedes"
    arc.codegen(cfilter, args, return_names=return_names, output_dir=output_dir)
