"""
CANSAT 3D Trajectory Reconstruction System - IMPROVED VERSION
==============================================================
Author: [Your Name]
Contact: [Your Email/Phone]
Institution: [Your University]
Competition: RESOLVE - SUPARCO Challenge

Improvements Added:
- Online gyro bias estimation during flight
- Numerically stable magnetometer calibration (pseudo-inverse)
- Adaptive time step handling
- Improved ZUPT with exponential decay
- Enhanced motion detection (freefall, periodic motion)
- Comprehensive trajectory validation metrics
- Accelerometer and gyroscope scaling factors
- Better drift correction
- Configurable gravity direction
- Export trajectory comparison to CSV
"""

import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import csv
from datetime import datetime
from dataclasses import dataclass, field
import tkinter as tk
import tkinter.ttk as ttk

plt.style.use('seaborn-v0_8-darkgrid')

@dataclass
class FilterConfig:
    """Configuration parameters - documented and tuned for MPU9250 @ 100Hz"""
    sample_rate: int = 100
    # Beta values tuned for different motion types
    beta_stationary: float = 0.05  # Low - stable orientation
    beta_rotation: float = 0.10    # Moderate - gyro dominant
    beta_linear: float = 0.15      # Higher - track accel changes
    beta_circular: float = 0.12    # Balanced for centripetal
    beta_descent: float = 0.08     # Low for controlled descent
    beta_freefall: float = 0.20    # High - accel unreliable
    beta_periodic: float = 0.11    # For swinging motion
    
    drift_alpha: float = 0.95      # Drift correction (0.92-0.98 recommended)
    zupt_acc_threshold: float = 0.5
    zupt_gyro_threshold: float = 0.1
    zupt_min_frames: int = 10      # Frames before activating ZUPT
    zupt_decay_rate: float = 0.95  # Exponential decay for ZUPT
    
    gravity: float = 9.81
    max_velocity: float = 100.0
    max_position: float = 10000.0
    
    fusion_mode: str = 'madgwick'
    ekf_q_noise: float = 0.01
    ekf_r_acc: float = 0.5
    ekf_r_mag: float = 0.8
    mag_ref: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    
    motion_acc_var_th: float = 0.5
    motion_gyro_var_th: float = 0.1
    auto_motion_detect: bool = True
    
    # New parameters
    online_bias_estimation: bool = True
    bias_estimation_alpha: float = 0.001
    accel_gravity_positive: bool = True  # True if +9.81 when pointing up
    freefall_threshold: float = 2.0      # m/s² for freefall detection
    periodic_fft_threshold: float = 0.3  # FFT threshold for periodic motion

class TrajectoryValidator:
    """Validates estimated trajectory against reference data"""
    def __init__(self):
        self.reference_path = None
        self.metrics_history = []
    
    def set_reference(self, reference_trajectory):
        self.reference_path = reference_trajectory
    
    def compute_metrics(self, estimated_trajectory):
        if self.reference_path is None or len(estimated_trajectory) == 0:
            return {'mae': 0.0, 'rmse': 0.0, 'max_error': 0.0}
        
        min_len = min(len(estimated_trajectory), len(self.reference_path))
        est = estimated_trajectory[:min_len]
        ref = self.reference_path[:min_len]
        
        mae = np.mean(np.abs(est - ref))
        rmse = np.sqrt(np.mean((est - ref)**2))
        max_error = np.max(np.abs(est - ref))
        
        metrics = {'mae': mae, 'rmse': rmse, 'max_error': max_error}
        self.metrics_history.append(metrics)
        return metrics
    
    def export_comparison(self, estimated_trajectory, filename='trajectory_comparison.csv'):
        if self.reference_path is None:
            print("No reference trajectory set")
            return
        
        min_len = min(len(estimated_trajectory), len(self.reference_path))
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['est_x', 'est_y', 'est_z', 'ref_x', 'ref_y', 'ref_z', 
                           'error_x', 'error_y', 'error_z', 'error_mag'])
            for i in range(min_len):
                est = estimated_trajectory[i]
                ref = self.reference_path[i]
                error = est - ref
                writer.writerow([
                    est[0], est[1], est[2],
                    ref[0], ref[1], ref[2],
                    error[0], error[1], error[2],
                    np.linalg.norm(error)
                ])
        print(f"Trajectory comparison exported to: {filename}")

class DataLogger:
    """Logs sensor and estimation data to CSV"""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.file = None
        self.writer = None
        if self.enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filename = f'cansat_log_{timestamp}.csv'
            self.file = open(self.filename, 'w', newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow([
                'timestamp', 'time_s',
                'ax_raw', 'ay_raw', 'az_raw',
                'gx_raw', 'gy_raw', 'gz_raw',
                'mx_raw', 'my_raw', 'mz_raw',
                'ax_cal', 'ay_cal', 'az_cal',
                'gx_cal', 'gy_cal', 'gz_cal',
                'mx_cal', 'my_cal', 'mz_cal',
                'gyro_bias_x', 'gyro_bias_y', 'gyro_bias_z',
                'q0', 'q1', 'q2', 'q3',
                'roll', 'pitch', 'yaw',
                'px', 'py', 'pz',
                'vx', 'vy', 'vz',
                'ax_earth', 'ay_earth', 'az_earth',
                'zupt_active', 'motion_type',
                'rmse', 'mae', 'max_error'
            ])
            print(f"Data logging enabled: {self.filename}")
    
    def log(self, data):
        if self.enabled and self.writer:
            self.writer.writerow(data)
    
    def close(self):
        if self.file:
            self.file.close()
            if self.enabled:
                print(f"Data saved to: {self.filename}")

class SensorFusion:
    """Advanced sensor fusion with improved calibration and bias estimation"""
    def __init__(self, config: FilterConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        self.dt = 1.0 / config.sample_rate
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.beta = 0.1
        self.gyro_bias = np.zeros(3)
        self.gyro_bias_runtime = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.accel_scale = np.ones(3)
        self.gyro_scale = np.ones(3)
        self.mag_offset = np.zeros(3)
        self.mag_scale = np.ones(3)
        self.mag_rotation = np.eye(3)
        self.calibration_samples = []
        self.is_calibrated = False
        self.P = np.eye(4) * 0.1
        self.bias_estimation_window = deque(maxlen=100)
    
    def validate_sensor_data(self, accel, gyro, mag):
        if np.any(np.abs(accel) > 160) or np.any(np.isnan(accel)):
            return False
        if np.any(np.abs(gyro) > 35) or np.any(np.isnan(gyro)):
            return False
        if np.any(np.abs(mag) > 400) or np.any(np.isnan(mag)):
            return False
        return True
    
    def calibrate(self, n_samples=100):
        print(f"Calibrating sensors ({n_samples} samples)...")
        print("   Rotate CANSAT in figure-8 for mag, place flat for accel/gyro")
        self.calibration_samples = []
    
    def add_calibration_sample(self, accel, gyro, mag):
        if not self.validate_sensor_data(accel, gyro, mag):
            return False
        self.calibration_samples.append({
            'accel': accel.copy(),
            'gyro': gyro.copy(),
            'mag': mag.copy()
        })
        if len(self.calibration_samples) >= 100:
            self.compute_biases_and_scales()
            self.compute_mag_ellipsoid()
            return True
        return False
    
    def compute_biases_and_scales(self):
        if len(self.calibration_samples) < 50:
            return
        
        accels = np.array([s['accel'] for s in self.calibration_samples])
        gyros = np.array([s['gyro'] for s in self.calibration_samples])
        
        self.gyro_bias = np.mean(gyros, axis=0)
        
        accel_mean = np.mean(accels, axis=0)
        accel_norm = np.linalg.norm(accel_mean)
        
        if accel_norm > 0:
            if self.config.accel_gravity_positive:
                self.accel_bias = accel_mean - np.array([0, 0, self.config.gravity])
            else:
                self.accel_bias = accel_mean + np.array([0, 0, self.config.gravity])
            
            measured_gravity = accel_norm
            scale_factor = self.config.gravity / measured_gravity
            self.accel_scale = np.ones(3) * scale_factor
        
        print("Bias and scale calibration complete!")
        print(f"   Gyro bias: [{self.gyro_bias[0]:+.5f}, {self.gyro_bias[1]:+.5f}, {self.gyro_bias[2]:+.5f}] rad/s")
        print(f"   Accel bias: [{self.accel_bias[0]:+.3f}, {self.accel_bias[1]:+.3f}, {self.accel_bias[2]:+.3f}] m/s²")
        print(f"   Accel scale: [{self.accel_scale[0]:.4f}, {self.accel_scale[1]:.4f}, {self.accel_scale[2]:.4f}]")
    
    def compute_mag_ellipsoid(self):
        """Numerically stable ellipsoid fitting using least-squares"""
        if len(self.calibration_samples) < 50:
            return
        
        mags = np.array([s['mag'] for s in self.calibration_samples])
        x, y, z = mags[:,0:1], mags[:,1:2], mags[:,2:3]
        
        J = np.hstack((x*x, y*y, z*z, x*y, x*z, y*z, x, y, z))
        K = np.ones_like(x)
        
        try:
            ABC = np.linalg.lstsq(J, K, rcond=None)[0]
        except:
            print("Magnetometer calibration failed")
            return
        
        vec = np.append(ABC, -1)
        Amat = np.array([
            [vec[0], vec[3]/2, vec[4]/2, vec[6]/2],
            [vec[3]/2, vec[1], vec[5]/2, vec[7]/2],
            [vec[4]/2, vec[5]/2, vec[2], vec[8]/2],
            [vec[6]/2, vec[7]/2, vec[8]/2, vec[9]]
        ])
        
        A3 = Amat[0:3,0:3]
        A3inv = np.linalg.pinv(A3)  # Pseudo-inverse for stability
        center = -np.dot(A3inv, vec[6:9]/2)
        
        Tofs = np.eye(4)
        Tofs[3,0:3] = center
        Amat_center = np.dot(np.dot(Tofs.T, Amat), Tofs)
        Amat_center /= -Amat_center[3,3]
        
        evals, evecs = np.linalg.eig(Amat_center[0:3,0:3])
        axes = np.sqrt(1.0 / np.abs(evals))
        axes = np.sort(axes)[::-1]
        
        self.mag_offset = center
        self.mag_rotation = evecs
        self.mag_scale = (1.0 / axes) * np.linalg.norm(self.config.mag_ref)
        self.is_calibrated = True
        
        print("Magnetometer ellipsoid calibration complete!")
        print(f"   Offset: [{center[0]:+.3f}, {center[1]:+.3f}, {center[2]:+.3f}]")
        print(f"   Scales: [{self.mag_scale[0]:+.3f}, {self.mag_scale[1]:+.3f}, {self.mag_scale[2]:+.3f}]")
    
    def update_online_bias(self, gyro, is_stationary):
        """Online gyro bias estimation during stationary periods"""
        if not self.config.online_bias_estimation:
            return
        
        if is_stationary:
            self.bias_estimation_window.append(gyro.copy())
            if len(self.bias_estimation_window) >= 20:
                current_bias = np.mean(list(self.bias_estimation_window), axis=0)
                alpha = self.config.bias_estimation_alpha
                self.gyro_bias_runtime = ((1 - alpha) * self.gyro_bias_runtime + 
                                         alpha * current_bias)
    
    def set_beta(self, beta):
        self.beta = beta
    
    def get_calibrated_data(self, accel, gyro, mag):
        accel_cal = (accel - self.accel_bias) * self.accel_scale
        gyro_cal = (gyro - self.gyro_bias - self.gyro_bias_runtime) * self.gyro_scale
        mag_cal = np.dot(self.mag_rotation, self.mag_scale * (mag - self.mag_offset))
        return accel_cal, gyro_cal, mag_cal
    
    def madgwick_update(self, accel, gyro, mag, dt=None):
        """Madgwick with adaptive dt"""
        if dt is None:
            dt = self.dt
        
        if not self.validate_sensor_data(accel, gyro, mag):
            return False
        
        gyro = (gyro - self.gyro_bias - self.gyro_bias_runtime) * self.gyro_scale
        accel = (accel - self.accel_bias) * self.accel_scale
        mag = np.dot(self.mag_rotation, self.mag_scale * (mag - self.mag_offset))
        
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0:
            accel = accel / accel_norm
        else:
            return False
        
        mag_norm = np.linalg.norm(mag)
        if mag_norm > 0:
            mag = mag / mag_norm
        else:
            return False
        
        q0, q1, q2, q3 = self.q
        f = np.array([
            2*(q1*q3 - q0*q2) - accel[0],
            2*(q0*q1 + q2*q3) - accel[1],
            2*(0.5 - q1**2 - q2**2) - accel[2]
        ])
        J = np.array([
            [-2*q2, 2*q3, -2*q0, 2*q1],
            [2*q1, 2*q0, 2*q3, 2*q2],
            [0, -4*q1, -4*q2, 0]
        ])
        
        step = J.T @ f
        step_norm = np.linalg.norm(step)
        if step_norm > 0:
            step = step / step_norm
        
        q_dot = 0.5 * self.quaternion_multiply(self.q, np.array([0, gyro[0], gyro[1], gyro[2]]))
        q_dot = q_dot - self.beta * step
        self.q = self.q + q_dot * dt
        self.q = self.q / np.linalg.norm(self.q)
        return True
    
    def ekf_update(self, accel, gyro, mag, dt=None):
        if dt is None:
            dt = self.dt
            
        omega = np.array([0, gyro[0], gyro[1], gyro[2]])
        q_dot = 0.5 * self.quaternion_multiply(self.q, omega)
        q_pred = self.q + q_dot * dt
        q_pred = q_pred / np.linalg.norm(q_pred)
        
        Omega = self.skew_symmetric(omega)
        F = np.eye(4) + 0.5 * dt * Omega
        Q = np.eye(4) * self.config.ekf_q_noise
        self.P = F @ self.P @ F.T + Q
        
        accel_norm = accel / (np.linalg.norm(accel) + 1e-6)
        mag_norm = mag / (np.linalg.norm(mag) + 1e-6)
        
        g_ref = np.array([0, 0, 1.0])
        m_ref = self.config.mag_ref
        h_acc = self.rotate_vector(q_pred, g_ref)
        h_mag = self.rotate_vector(q_pred, m_ref)
        h = np.concatenate((h_acc, h_mag))
        z = np.concatenate((accel_norm, mag_norm))
        
        H_acc = self.measurement_jacobian(q_pred, g_ref)
        H_mag = self.measurement_jacobian(q_pred, m_ref)
        H = np.vstack((H_acc, H_mag))
        
        R = np.eye(6)
        R[:3,:3] *= self.config.ekf_r_acc
        R[3:,3:] *= self.config.ekf_r_mag
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        innovation = z - h
        delta_q = K @ innovation
        self.q = q_pred + delta_q
        self.q /= np.linalg.norm(self.q)
        self.P = (np.eye(4) - K @ H) @ self.P
        return True
    
    def update(self, accel, gyro, mag, dt=None):
        if self.config.fusion_mode == 'madgwick':
            return self.madgwick_update(accel, gyro, mag, dt)
        elif self.config.fusion_mode == 'ekf':
            return self.ekf_update(accel, gyro, mag, dt)
        return False
    
    @staticmethod
    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def skew_symmetric(v):
        return np.array([
            [0, -v[1], -v[2], -v[3]],
            [v[1], 0, v[3], -v[2]],
            [v[2], -v[3], 0, v[1]],
            [v[3], v[2], -v[1], 0]
        ])
    
    def rotate_vector(self, q, v):
        qv = np.concatenate(([0], v))
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        return self.quaternion_multiply(self.quaternion_multiply(q, qv), q_conj)[1:]
    
    def measurement_jacobian(self, q, ref):
        w, x, y, z = q
        rx, ry, rz = ref
        H = np.array([
            [2*w*rx + 2*y*rz - 2*z*ry, 2*x*rx + 2*y*ry + 2*z*rz, 2*w*rz + 2*y*rx - 2*x*ry, 2*w*ry + 2*z*rx - 2*x*rz],
            [2*w*ry + 2*z*rx - 2*y*rx, 2*w*rz - 2*x*rx - 2*z*ry, 2*x*ry - 2*y*rx - 2*z*rz, 2*x*rx + 2*y*rz + 2*z*ry],
            [2*w*rz + 2*x*ry - 2*y*rx, -2*w*ry + 2*x*rz - 2*z*ry, 2*w*rx + 2*x*rz + 2*z*rz, -2*w*rx - 2*y*rz - 2*z*rx]
        ])
        return H / 2
    
    def get_rotation_matrix(self):
        q0, q1, q2, q3 = self.q
        return np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])
    
    def get_euler_angles(self):
        q0, q1, q2, q3 = self.q
        roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
        pitch = np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1.0, 1.0))
        yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
        return roll, pitch, yaw

class TrajectoryEstimator:
    """Improved trajectory estimation with better drift correction"""
    def __init__(self, config: FilterConfig):
        self.config = config
        self.gravity = config.gravity
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.accel_prev = np.zeros(3)
        self.velocity_prev = np.zeros(3)
        self.position_history = deque(maxlen=2000)
        self.velocity_history = deque(maxlen=2000)
        self.zupt_active = False
        self.zupt_frame_count = 0
        self.motion_history = deque(maxlen=50)
    
    def reset(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.accel_prev = np.zeros(3)
        self.velocity_prev = np.zeros(3)
        self.position_history.clear()
        self.velocity_history.clear()
        self.zupt_active = False
        self.zupt_frame_count = 0
        self.motion_history.clear()
    
    def detect_zero_velocity(self, accel_body, gyro):
        acc_mag = np.linalg.norm(accel_body - np.array([0, 0, self.gravity]))
        gyro_mag = np.linalg.norm(gyro)
        is_stationary = (acc_mag < self.config.zupt_acc_threshold and 
                        gyro_mag < self.config.zupt_gyro_threshold)
        
        if is_stationary:
            self.zupt_frame_count += 1
        else:
            self.zupt_frame_count = 0
        
        return self.zupt_frame_count >= self.config.zupt_min_frames
    
    def detect_motion_type(self, accel, gyro):
        """Enhanced motion detection including freefall and periodic motion"""
        self.motion_history.append((np.var(accel), np.var(gyro)))
        
        if len(self.motion_history) < 10:
            return 'stationary'
        
        acc_vars, gyro_vars = zip(*self.motion_history)
        acc_var = np.mean(acc_vars)
        gyro_var = np.mean(gyro_vars)
        
        # Check for freefall (low acceleration magnitude)
        acc_mag = np.linalg.norm(accel)
        if acc_mag < self.config.freefall_threshold:
            return 'freefall'
        
        # Check for periodic motion (swinging) using FFT
        if len(self.motion_history) >= 30:
            recent_accels = [a for a, g in list(self.motion_history)[-30:]]
            fft = np.fft.fft(recent_accels)
            fft_mag = np.abs(fft[1:15])  # Check first few frequencies
            if np.max(fft_mag) > self.config.periodic_fft_threshold:
                return 'periodic'
        
        # Original motion type detection
        if gyro_var > self.config.motion_gyro_var_th and acc_var < self.config.motion_acc_var_th:
            return 'rotation'
        elif acc_var > self.config.motion_acc_var_th and gyro_var < self.config.motion_gyro_var_th:
            return 'linear'
        elif acc_var > self.config.motion_acc_var_th and gyro_var > self.config.motion_gyro_var_th:
            return 'circular'
        elif acc_var < self.config.motion_acc_var_th / 2:
            return 'descent'
        else:
            return 'stationary'
    
    def update(self, accel_body, gyro, rotation_matrix, dt):
        """Update with improved drift correction and ZUPT"""
        accel_earth = rotation_matrix @ accel_body
        
        # Remove gravity (sign depends on sensor orientation)
        if self.config.accel_gravity_positive:
            accel_earth[2] -= self.gravity
        else:
            accel_earth[2] += self.gravity
        
        self.zupt_active = self.detect_zero_velocity(accel_body, gyro)
        
        if self.zupt_active:
            # Exponential decay to zero instead of hard reset
            self.velocity *= self.config.zupt_decay_rate
        else:
            alpha = self.config.drift_alpha
            accel_avg = (self.accel_prev + accel_earth) / 2
            
            if len(self.velocity_history) > 0:
                # Drift-corrected integration
                self.velocity = (alpha * self.velocity + 
                               (1 - alpha) * (self.velocity_prev + accel_avg * dt))
            else:
                self.velocity = self.velocity_prev + accel_avg * dt
        
        # Velocity limiting
        vel_mag = np.linalg.norm(self.velocity)
        if vel_mag > self.config.max_velocity:
            self.velocity = self.velocity / vel_mag * self.config.max_velocity
        
        # Trapezoidal integration for position
        vel_avg = (self.velocity_prev + self.velocity) / 2
        self.position = self.position + vel_avg * dt
        
        # Position limiting
        pos_mag = np.linalg.norm(self.position)
        if pos_mag > self.config.max_position:
            self.position = self.position / pos_mag * self.config.max_position
        
        self.position_history.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        self.accel_prev = accel_earth.copy()
        self.velocity_prev = self.velocity.copy()
    
    def get_trajectory(self):
        if len(self.position_history) == 0:
            return np.zeros((1, 3))
        return np.array(list(self.position_history))

class MotionSimulator:
    def __init__(self):
        self.accel = np.zeros(3)
        self.gyro = np.zeros(3)
        self.mag = np.zeros(3)
        self.motion_type = 'rotation'
        self.start_time = time.time()
        self.position_gt = np.zeros(3)
        self.velocity_gt = np.zeros(3)
        self.accel_gt_prev = np.zeros(3)
        self.position_gt_history = deque(maxlen=2000)
    
    def set_motion(self, motion_type):
        self.motion_type = motion_type
        self.start_time = time.time()
        self.position_gt = np.zeros(3)
        self.velocity_gt = np.zeros(3)
        self.accel_gt_prev = np.zeros(3)
        self.position_gt_history.clear()
        print(f"Switched to: {motion_type.upper()}")
    
    def simulate_data(self):
        t = time.time() - self.start_time
        dt = 1.0 / 100
        
        if self.motion_type == 'stationary':
            self.accel = np.array([0, 0, 9.81]) + np.random.normal(0, 0.05, 3)
            self.gyro = np.random.normal(0, 0.005, 3)
            self.mag = np.array([1, 0, 0]) + np.random.normal(0, 0.02, 3)
            accel_gt = np.array([0, 0, 0])
        elif self.motion_type == 'rotation':
            self.accel = np.array([0, 0, 9.81]) + np.random.normal(0, 0.1, 3)
            self.gyro = np.array([
                0.5 * np.sin(t * 0.8),
                0.6 * np.cos(t * 0.6),
                0.4 * np.sin(t * 0.5)
            ]) + np.random.normal(0, 0.01, 3)
            self.mag = np.array([np.cos(t), np.sin(t), 0]) + np.random.normal(0, 0.05, 3)
            accel_gt = np.array([0, 0, 0])
        elif self.motion_type == 'linear':
            accel_gt = np.array([
                2.0 * np.sin(t * 1.5),
                1.5 * np.cos(t * 1.2),
                0
            ])
            self.accel = accel_gt + np.array([0, 0, 9.81]) + np.random.normal(0, 0.15, 3)
            self.gyro = np.random.normal(0, 0.02, 3)
            self.mag = np.array([1, 0, 0]) + np.random.normal(0, 0.05, 3)
        elif self.motion_type == 'circular':
            radius = 1.0
            omega = 0.5
            accel_gt = np.array([
                -radius * omega**2 * np.cos(omega * t),
                -radius * omega**2 * np.sin(omega * t),
                0
            ])
            self.accel = accel_gt + np.array([0, 0, 9.81]) + np.random.normal(0, 0.1, 3)
            self.gyro = np.array([0, 0, omega]) + np.random.normal(0, 0.01, 3)
            self.mag = np.array([np.cos(omega * t), np.sin(omega * t), 0]) + np.random.normal(0, 0.05, 3)
        elif self.motion_type == 'descent':
            accel_gt = np.array([
                0.2 * np.sin(t),
                0.2 * np.cos(t),
                -1.5
            ])
            self.accel = accel_gt + np.array([0, 0, 9.81]) + np.random.normal(0, 0.1, 3)
            self.gyro = np.array([
                0.1 * np.sin(t),
                0.1 * np.cos(t),
                0.05
            ]) + np.random.normal(0, 0.01, 3)
            self.mag = np.array([1, 0, 0]) + np.random.normal(0, 0.05, 3)
        elif self.motion_type == 'freefall':
            accel_gt = np.array([0, 0, -9.81])
            self.accel = np.array([0, 0, 0]) + np.random.normal(0, 0.2, 3)
            self.gyro = np.random.normal(0, 0.5, 3)
            self.mag = np.array([1, 0, 0]) + np.random.normal(0, 0.1, 3)
        elif self.motion_type == 'periodic':
            freq = 0.5
            accel_gt = np.array([
                3.0 * np.sin(2 * np.pi * freq * t),
                0,
                0
            ])
            self.accel = accel_gt + np.array([0, 0, 9.81]) + np.random.normal(0, 0.1, 3)
            self.gyro = np.random.normal(0, 0.02, 3)
            self.mag = np.array([1, 0, 0]) + np.random.normal(0, 0.05, 3)
        else:
            accel_gt = np.array([0, 0, 0])
        
        accel_avg = (self.accel_gt_prev + accel_gt) / 2
        self.velocity_gt += accel_avg * dt
        self.position_gt += self.velocity_gt * dt
        self.position_gt_history.append(self.position_gt.copy())
        self.accel_gt_prev = accel_gt.copy()
    
    def get_gt_trajectory(self):
        if len(self.position_gt_history) == 0:
            return np.zeros((1, 3))
        return np.array(list(self.position_gt_history))

class ControlPanelGUI:
    def __init__(self, parent):
        self.parent = parent
        self.canvas = tk.Canvas(parent, bg="black", highlightthickness=0, width=300)
        self.scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="black")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=False)
        self.scrollbar.pack(side="right", fill="y")
        
        title = tk.Label(self.scrollable_frame, text="CANSAT CONTROL PANEL",
                        font=("Helvetica", 16, "bold"), fg="#00FF00", bg="black")
        title.pack(pady=(10, 20))
        
        self.box_font = ("Courier", 10, "bold")
        self.label_font = ("Helvetica", 10, "bold")
        self.box_bg = "#1a1a2e"
        self.box_fg = "#00FF00"
        
        self._create_section("SYSTEM STATUS")
        self.cal_status = self._create_indicator("Calibration")
        self.zupt_status = self._create_indicator("ZUPT")
        self.motion_label = self._create_value_box("Motion Type")
        self.beta_label = self._create_value_box("Filter Beta")
        self.fusion_label = self._create_value_box("Fusion Mode")
        self.rmse_label = self._create_value_box("RMSE (m)")
        self.mae_label = self._create_value_box("MAE (m)")
        
        ttk.Separator(self.scrollable_frame, orient='horizontal').pack(fill='x', pady=15)
        
        self._create_section("BIAS ESTIMATION")
        self.bias_x_box = self._create_value_box("Gyro Bias X")
        self.bias_y_box = self._create_value_box("Gyro Bias Y")
        self.bias_z_box = self._create_value_box("Gyro Bias Z")
        
        ttk.Separator(self.scrollable_frame, orient='horizontal').pack(fill='x', pady=15)
        
        self._create_section("ORIENTATION (deg)")
        self.roll_box = self._create_value_box("Roll")
        self.pitch_box = self._create_value_box("Pitch")
        self.yaw_box = self._create_value_box("Yaw")
        
        ttk.Separator(self.scrollable_frame, orient='horizontal').pack(fill='x', pady=15)
        
        self._create_section("POSITION (m)")
        self.px_box = self._create_value_box("X")
        self.py_box = self._create_value_box("Y")
        self.pz_box = self._create_value_box("Z")
        self.pos_mag_box = self._create_value_box("|P|")
        
        ttk.Separator(self.scrollable_frame, orient='horizontal').pack(fill='x', pady=15)
        
        self._create_section("VELOCITY (m/s)")
        self.vx_box = self._create_value_box("Vx")
        self.vy_box = self._create_value_box("Vy")
        self.vz_box = self._create_value_box("Vz")
        self.vel_mag_box = self._create_value_box("|V|")
        
        ttk.Separator(self.scrollable_frame, orient='horizontal').pack(fill='x', pady=15)
        
        self._create_section("RUNTIME INFO")
        self.samples_box = self._create_value_box("Samples")
        self.time_box = self._create_value_box("Time (s)")
        
        btn_frame = tk.Frame(self.scrollable_frame, bg="black")
        btn_frame.pack(pady=15, fill='x')
        btn_style = {"font": ("Helvetica", 9, "bold"), "bg": "#2E86AB", 
                    "fg": "white", "width": 12, "height": 1}
        
        tk.Button(btn_frame, text="1: Stationary", command=lambda: self.on_motion_change('1'), 
                 **btn_style).grid(row=0, column=0, padx=5, pady=3)
        tk.Button(btn_frame, text="2: Rotation", command=lambda: self.on_motion_change('2'),
                 **btn_style).grid(row=0, column=1, padx=5, pady=3)
        tk.Button(btn_frame, text="3: Linear", command=lambda: self.on_motion_change('3'),
                 **btn_style).grid(row=1, column=0, padx=5, pady=3)
        tk.Button(btn_frame, text="4: Circular", command=lambda: self.on_motion_change('4'),
                 **btn_style).grid(row=1, column=1, padx=5, pady=3)
        tk.Button(btn_frame, text="5: Descent", command=lambda: self.on_motion_change('5'),
                 **btn_style).grid(row=2, column=0, padx=5, pady=3)
        tk.Button(btn_frame, text="6: Freefall", command=lambda: self.on_motion_change('6'),
                 **btn_style).grid(row=2, column=1, padx=5, pady=3)
        tk.Button(btn_frame, text="7: Periodic", command=lambda: self.on_motion_change('7'),
                 **btn_style).grid(row=3, column=0, padx=5, pady=3)
        tk.Button(btn_frame, text="Q: Quit", command=self.on_quit, 
                 bg="#D62246", fg="white", font=("Helvetica", 9, "bold"),
                 width=12, height=1).grid(row=3, column=1, padx=5, pady=3)
        
        self.motion_callback = None
        self.quit_callback = None
    
    def _create_section(self, text):
        label = tk.Label(self.scrollable_frame, text=text, font=("Helvetica", 12, "bold"),
                        fg="#FFD700", bg="black")
        label.pack(pady=(10, 5), anchor="w")
    
    def _create_indicator(self, label_text):
        frame = tk.Frame(self.scrollable_frame, bg="black")
        frame.pack(fill='x', pady=3)
        tk.Label(frame, text=label_text + ":", font=self.label_font,
                fg=self.box_fg, bg="black").pack(side="left")
        status = tk.Label(frame, text="UNKNOWN", width=15, font=self.box_font,
                         bg=self.box_bg, fg=self.box_fg, relief="ridge", anchor="center")
        status.pack(side="left", padx=(10, 0))
        return status
    
    def _create_value_box(self, label_text):
        frame = tk.Frame(self.scrollable_frame, bg="black")
        frame.pack(fill='x', pady=3)
        tk.Label(frame, text=label_text + ":", font=self.label_font,
                fg=self.box_fg, bg="black").pack(side="left")
        box = tk.Label(frame, text="0.000", width=15, font=self.box_font,
                      bg=self.box_bg, fg=self.box_fg, relief="ridge", anchor="e")
        box.pack(side="left", padx=(10, 0))
        return box
    
    def update(self, orientation, velocity, position, motion_type, beta, fusion_mode, 
              zupt_active, is_calibrated, samples, runtime, metrics, gyro_bias_runtime):
        cal_text = "CALIBRATED" if is_calibrated else "NOT CALIBRATED"
        self.cal_status.config(text=cal_text, fg="#06A77D" if is_calibrated else "#F18F01")
        
        zupt_text = "ACTIVE" if zupt_active else "INACTIVE"
        self.zupt_status.config(text=zupt_text, fg="#06A77D" if zupt_active else "#888888")
        
        self.motion_label.config(text=motion_type.upper())
        self.beta_label.config(text=f"{beta:.3f}" if fusion_mode == 'madgwick' else "N/A")
        self.fusion_label.config(text=fusion_mode.upper())
        self.rmse_label.config(text=f"{metrics['rmse']:.4f}")
        self.mae_label.config(text=f"{metrics['mae']:.4f}")
        
        self.bias_x_box.config(text=f"{gyro_bias_runtime[0]:+.5f}")
        self.bias_y_box.config(text=f"{gyro_bias_runtime[1]:+.5f}")
        self.bias_z_box.config(text=f"{gyro_bias_runtime[2]:+.5f}")
        
        roll, pitch, yaw = orientation
        self.roll_box.config(text=f"{np.degrees(roll):+8.2f}")
        self.pitch_box.config(text=f"{np.degrees(pitch):+8.2f}")
        self.yaw_box.config(text=f"{np.degrees(yaw):+8.2f}")
        
        self.px_box.config(text=f"{position[0]:+8.3f}")
        self.py_box.config(text=f"{position[1]:+8.3f}")
        self.pz_box.config(text=f"{position[2]:+8.3f}")
        self.pos_mag_box.config(text=f"{np.linalg.norm(position):8.3f}")
        
        self.vx_box.config(text=f"{velocity[0]:+8.3f}")
        self.vy_box.config(text=f"{velocity[1]:+8.3f}")
        self.vz_box.config(text=f"{velocity[2]:+8.3f}")
        self.vel_mag_box.config(text=f"{np.linalg.norm(velocity):8.3f}")
        
        self.samples_box.config(text=f"{samples}")
        self.time_box.config(text=f"{runtime:.1f}")
        self.parent.update()
    
    def on_motion_change(self, key):
        if self.motion_callback:
            self.motion_callback(key)
    
    def on_quit(self):
        if self.quit_callback:
            self.quit_callback()
    
    def set_motion_callback(self, callback):
        self.motion_callback = callback
    
    def set_quit_callback(self, callback):
        self.quit_callback = callback

class ProfessionalVisualizer:
    def __init__(self, parent):
        self.fig = plt.Figure(figsize=(12, 8), facecolor='#0d1117')
        self.fig.suptitle('CANSAT 3D Trajectory Reconstruction System v3.1 - IMPROVED', 
                         fontsize=18, fontweight='bold', color='white', y=0.98)
        
        gs = self.fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35, 
                                   left=0.05, right=0.97, top=0.93, bottom=0.05)
        
        self.ax3d = self.fig.add_subplot(gs[0:2, 0:2], projection='3d', facecolor='#161b22')
        self.setup_3d_plot()
        self.ax_orient = self.fig.add_subplot(gs[0, 2], projection='3d', facecolor='#161b22')
        self.setup_orientation_plot()
        self.ax_vel = self.fig.add_subplot(gs[1, 2], projection='3d', facecolor='#161b22')
        self.setup_velocity_plot()
        self.ax_pos = self.fig.add_subplot(gs[2, 0], projection='3d', facecolor='#161b22')
        self.setup_position_plot()
        self.ax_accel = self.fig.add_subplot(gs[2, 1], projection='3d', facecolor='#161b22')
        self.setup_acceleration_plot()
        self.ax_zupt = self.fig.add_subplot(gs[2, 2], facecolor='#161b22')
        self.setup_zupt_plot()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.time_data = deque(maxlen=500)
        self.orientation_data = deque(maxlen=500)
        self.accel_data = deque(maxlen=500)
        self.vel_data = deque(maxlen=500)
        self.pos_data = deque(maxlen=500)
        self.zupt_data = deque(maxlen=500)
        self.start_time = time.time()
    
    def setup_3d_plot(self):
        self.ax3d.set_xlabel('X (m)', color='white', fontsize=10)
        self.ax3d.set_ylabel('Y (m)', color='white', fontsize=10)
        self.ax3d.set_zlabel('Z (m)', color='white', fontsize=10)
        self.ax3d.set_title('3D Trajectory', color='white', fontsize=12)
        self.ax3d.tick_params(colors='white', labelsize=9)
        self.ax3d.xaxis.pane.fill = False
        self.ax3d.yaxis.pane.fill = False
        self.ax3d.zaxis.pane.fill = False
        self.ax3d.grid(True, alpha=0.3, color='gray')
    
    def setup_orientation_plot(self):
        self.ax_orient.set_xlabel('Time', color='white', fontsize=10)
        self.ax_orient.set_ylabel('Component', color='white', fontsize=10)
        self.ax_orient.set_zlabel('°', color='white', fontsize=10)
        self.ax_orient.set_title('Orientation', color='white', fontsize=12)
        self.ax_orient.tick_params(colors='white', labelsize=9)
        self.ax_orient.xaxis.pane.fill = False
        self.ax_orient.yaxis.pane.fill = False
        self.ax_orient.zaxis.pane.fill = False
        self.ax_orient.grid(True, alpha=0.3)
    
    def setup_velocity_plot(self):
        self.ax_vel.set_xlabel('Time', color='white', fontsize=10)
        self.ax_vel.set_ylabel('Component', color='white', fontsize=10)
        self.ax_vel.set_zlabel('m/s', color='white', fontsize=10)
        self.ax_vel.set_title('Velocity', color='white', fontsize=12)
        self.ax_vel.tick_params(colors='white', labelsize=9)
        self.ax_vel.xaxis.pane.fill = False
        self.ax_vel.yaxis.pane.fill = False
        self.ax_vel.zaxis.pane.fill = False
        self.ax_vel.grid(True, alpha=0.3)
    
    def setup_position_plot(self):
        self.ax_pos.set_xlabel('Time', color='white', fontsize=10)
        self.ax_pos.set_ylabel('Component', color='white', fontsize=10)
        self.ax_pos.set_zlabel('m', color='white', fontsize=10)
        self.ax_pos.set_title('Position', color='white', fontsize=12)
        self.ax_pos.tick_params(colors='white', labelsize=9)
        self.ax_pos.xaxis.pane.fill = False
        self.ax_pos.yaxis.pane.fill = False
        self.ax_pos.zaxis.pane.fill = False
        self.ax_pos.grid(True, alpha=0.3)
    
    def setup_acceleration_plot(self):
        self.ax_accel.set_xlabel('Time', color='white', fontsize=10)
        self.ax_accel.set_ylabel('Component', color='white', fontsize=10)
        self.ax_accel.set_zlabel('m/s²', color='white', fontsize=10)
        self.ax_accel.set_title('Acceleration', color='white', fontsize=12)
        self.ax_accel.tick_params(colors='white', labelsize=9)
        self.ax_accel.xaxis.pane.fill = False
        self.ax_accel.yaxis.pane.fill = False
        self.ax_accel.zaxis.pane.fill = False
        self.ax_accel.grid(True, alpha=0.3)
    
    def setup_zupt_plot(self):
        self.ax_zupt.set_xlim(0, 100)
        self.ax_zupt.set_ylim(0, 1)
        self.ax_zupt.set_xlabel('Time (s)', color='white', fontsize=10)
        self.ax_zupt.set_ylabel('ZUPT', color='white', fontsize=10)
        self.ax_zupt.set_title('Zero Velocity Update', color='white', fontsize=12)
        self.ax_zupt.tick_params(colors='white', labelsize=9)
        self.ax_zupt.set_facecolor('#161b22')
        for spine in self.ax_zupt.spines.values():
            spine.set_color('white')
        self.ax_zupt.grid(True, alpha=0.3, color='gray')
    
    def update(self, trajectory, trajectory_gt, orientation, accel_earth, velocity, position, zupt_active):
        current_time = time.time() - self.start_time
        self.time_data.append(current_time)
        self.orientation_data.append(orientation)
        self.accel_data.append(accel_earth)
        self.vel_data.append(velocity)
        self.pos_data.append(position)
        self.zupt_data.append(1.0 if zupt_active else 0.0)
        
        if len(trajectory) > 1:
            self.ax3d.clear()
            self.setup_3d_plot()
            speeds = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            if len(speeds) > 0:
                norm_speeds = (speeds - speeds.min()) / (speeds.max() - speeds.min() + 1e-6)
                colors = plt.cm.plasma(norm_speeds)
                for i in range(len(trajectory) - 1):
                    self.ax3d.plot(trajectory[i:i+2, 0], 
                                  trajectory[i:i+2, 1], 
                                  trajectory[i:i+2, 2],
                                  color=colors[i], linewidth=2.5, alpha=0.8)
            self.ax3d.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], [trajectory[-1, 2]], 
                            c='red', s=100, marker='o', edgecolors='white', linewidths=2)
            if len(trajectory_gt) > 1:
                self.ax3d.plot(trajectory_gt[:, 0], trajectory_gt[:, 1], trajectory_gt[:, 2],
                              color='cyan', linewidth=1.5, alpha=0.5, linestyle='--')
            max_range = np.max([np.ptp(trajectory[:, 0]), np.ptp(trajectory[:, 1]), np.ptp(trajectory[:, 2])]) / 2.0 + 0.5
            mid_x = (np.max(trajectory[:, 0]) + np.min(trajectory[:, 0])) / 2.0
            mid_y = (np.max(trajectory[:, 1]) + np.min(trajectory[:, 1])) / 2.0
            mid_z = (np.max(trajectory[:, 2]) + np.min(trajectory[:, 2])) / 2.0
            self.ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if len(self.time_data) > 2:
            times = np.array(self.time_data)
            orientations = np.array(self.orientation_data)
            X, Y = np.meshgrid(times, np.arange(3))
            Z = np.degrees(orientations[:, [0, 1, 2]].T)
            self.ax_orient.clear()
            self.setup_orientation_plot()
            self.ax_orient.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8)
            self.ax_orient.set_zlim(-180, 180)
        
        if len(self.time_data) > 2:
            times = np.array(self.time_data)
            positions = np.array(self.pos_data)
            X, Y = np.meshgrid(times, np.arange(3))
            Z = positions.T
            self.ax_pos.clear()
            self.setup_position_plot()
            self.ax_pos.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        
        if len(self.time_data) > 2:
            times = np.array(self.time_data)
            accels = np.array(self.accel_data)
            X, Y = np.meshgrid(times, np.arange(3))
            Z = accels.T
            self.ax_accel.clear()
            self.setup_acceleration_plot()
            self.ax_accel.plot_surface(X, Y, Z, cmap='inferno', alpha=0.8)
        
        if len(self.time_data) > 2:
            times = np.array(self.time_data)
            vels = np.array(self.vel_data)
            X, Y = np.meshgrid(times, np.arange(3))
            Z = vels.T
            self.ax_vel.clear()
            self.setup_velocity_plot()
            self.ax_vel.plot_surface(X, Y, Z, cmap='magma', alpha=0.8)
        
        if len(self.time_data) > 1:
            times = np.array(self.time_data)
            zupts = np.array(self.zupt_data)
            self.ax_zupt.clear()
            self.setup_zupt_plot()
            self.ax_zupt.fill_between(times, 0, zupts, color='#06A77D', alpha=0.5)
            self.ax_zupt.plot(times, zupts, color='#06A77D', linewidth=2)
            if len(times) > 0:
                self.ax_zupt.set_xlim(max(0, times[-1] - 50), times[-1] + 5)
        
        self.canvas.draw()
        plt.pause(0.001)

def main():
    print("\n" + "="*70)
    print("  CANSAT 3D TRAJECTORY RECONSTRUCTION - IMPROVED VERSION")
    print("="*70)
    
    config = FilterConfig()
    
    print("\nEnable data logging? (y/n): ", end='')
    try:
        enable_logging = input().lower().strip() == 'y'
    except:
        enable_logging = True
    
    logger = DataLogger(enabled=enable_logging)
    validator = TrajectoryValidator()
    
    print("\nInitializing system components...")
    sensor_fusion = SensorFusion(config)
    trajectory_estimator = TrajectoryEstimator(config)
    sensor = MotionSimulator()
    
    root = tk.Tk()
    root.title("CANSAT 3D Trajectory Reconstruction - IMPROVED")
    root.geometry("1200x800")
    root.configure(bg="black")
    
    control_panel = ControlPanelGUI(root)
    visualizer = ProfessionalVisualizer(root)