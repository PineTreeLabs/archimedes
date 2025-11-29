import numpy as np
from cfilter import Attitude, cfilter, quat_from_accel, quaternion_to_euler


class TestAccelAttitude:
    def test_level_attitude(self):
        """Level accelerometer should give near-zero roll/pitch"""
        accel = np.array([0, 0, -1])
        q = quat_from_accel(accel, 0)
        rpy = quaternion_to_euler(q)
        assert np.abs(rpy[0]) < 1e-6  # roll
        assert np.abs(rpy[1]) < 1e-6  # pitch

    def test_tilt_forward(self):
        """Forward tilt should give positive pitch"""
        accel = np.array([0.5, 0, -0.866])  # ~30° pitch
        q = quat_from_accel(accel, 0)
        rpy = quaternion_to_euler(q)
        assert rpy[1] > 0.4 and rpy[1] < 0.6  # ~0.52 rad = 30°

    def test_tilt_sideways(self):
        """Sideways tilt should give roll"""
        accel = np.array([0, 0.5, -0.866])  # ~30° roll
        q = quat_from_accel(accel, 0)
        rpy = quaternion_to_euler(q)
        assert abs(rpy[0]) > 0.4 and abs(rpy[0]) < 0.6


class TestComplementaryFilter:
    def test_stationary_drift(self):
        """Stationary IMU should stay near initial attitude"""
        att = Attitude(q=np.array([1, 0, 0, 0]), rpy=np.array([0, 0, 0]))
        gyro = np.array([0, 0, 0])
        accel = np.array([0, 0, -1])

        # Run filter for 100 steps
        for _ in range(100):
            att = cfilter(att, gyro, accel, alpha=0.98, dt=0.01)

        # Should stay near identity
        assert np.allclose(att.q, [1, 0, 0, 0], atol=0.01)
        assert np.allclose(att.rpy, [0, 0, 0], atol=0.01)

    def test_pure_gyro_integration(self):
        """Alpha=1.0 should follow gyro only"""
        att = Attitude(q=np.array([1, 0, 0, 0]), rpy=np.array([0, 0, 0]))
        gyro = np.array([0, 0, 1])  # 1 rad/s yaw
        accel = np.array([0, 0, -1])

        # Integrate for 1 second
        for _ in range(100):
            att = cfilter(att, gyro, accel, alpha=1.0, dt=0.01)

        # Should have rotated ~1 radian in yaw
        assert att.rpy[2] > 0.9 and att.rpy[2] < 1.1

    def test_accel_correction(self):
        """Low alpha should trust accelerometer more"""
        # Start with incorrect attitude
        att = Attitude(
            q=np.array([0.966, 0.259, 0, 0]),  # ~30° roll error
            rpy=np.array([0.5, 0, 0]),
        )
        gyro = np.array([0, 0, 0])
        accel = np.array([0, 0, -1])  # Actually level

        # Low alpha = trust accel more
        for _ in range(50):
            att = cfilter(att, gyro, accel, alpha=0.5, dt=0.01)

        # Should correct toward level
        assert abs(att.rpy[0]) < 0.3  # Reduced from 0.5

    def test_quaternion_stays_normalized(self):
        """Quaternion magnitude should stay at 1.0"""
        att = Attitude(q=np.array([1, 0, 0, 0]), rpy=np.array([0, 0, 0]))
        gyro = np.array([0.1, 0.2, 0.3])
        accel = np.array([0.1, 0.2, -0.97])

        for _ in range(1000):
            att = cfilter(att, gyro, accel, alpha=0.98, dt=0.01)
            assert np.abs(np.linalg.norm(att.q) - 1.0) < 1e-6
