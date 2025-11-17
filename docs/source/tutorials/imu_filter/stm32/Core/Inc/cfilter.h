/* Complementary filter for 3-axis accelerometer */

#ifndef CFILT_H
#define CFILT_H

static inline int qq_mult(const float *q1, const float *q2, float *result) {
    result[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
    result[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
    result[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
    result[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
    return 0;
}

static inline int qv_mult(const float *q, const float *v, float *result) {
    result[0] = -q[1] * v[0] - q[2] * v[1] - q[3] * v[2];
    result[1] =  q[0] * v[0] + q[2] * v[2] - q[3] * v[1];
    result[2] =  q[0] * v[1] - q[1] * v[2] + q[3] * v[0];
    result[3] =  q[0] * v[2] + q[1] * v[1] - q[2] * v[0];
    return 0;
}

static inline int quaternion_derivative(const float *q, const float *omega, float *q_dot) {
    qv_mult(q, omega, q_dot);
    for (int i = 0; i < 4; i++) {
        q_dot[i] = 0.5f * q_dot[i];
    }
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

    float roll = atan2f(ay, az);
    float pitch = atan2f(-ax, sqrtf(ay * ay + az * az));

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