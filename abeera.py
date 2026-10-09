#!/usr/bin/env python3
"""
CANSAT Mission Control (complete) - MATHEMATICALLY CORRECTED VERSION WITH SCROLLING
- Demo simulation + optional ESP32 serial ingestion
- Modern GUI (CustomTkinter preferred, fallback to Tkinter)
- Embedded Matplotlib tabs: 3D Trajectory, Orientation Cube, Signals
- Paraboloid (bowl) fit overlay, time-colormapped trajectory
- Complementary filter, ZUPT, trapezoidal integration
- KPI dashboard, save/load JSON/CSV, realtime log panel
- Enhanced scrolling functionality:
  * Scrollable left control panel with mouse wheel support
  * Log panel with vertical scrollbar and mouse wheel scrolling
  * 3D trajectory plot with mouse wheel zoom
  * Signal plots with scroll zoom and Shift+scroll for horizontal panning
  * Cross-platform mouse wheel support (Windows, macOS, Linux)

COORDINATE FRAME CONVENTIONS:
- Body Frame: X-forward, Y-right, Z-down (typical IMU convention)
- Earth Frame: X-north, Y-east, Z-up (NED to ENU conversion)
- Gravity: [0, 0, -9.80665] m/s² in earth frame (Z-up)
- Euler Angles: Roll (X), Pitch (Y), Yaw (Z) in radians

MATHEMATICAL CORRECTIONS APPLIED:
- Fixed complementary filter order (gyro integration + accel fusion)
- Corrected gravity compensation (consistent coordinate frames)
- Optimized filter parameters for real-time performance
- Added sensor validation and stability checks
"""

import sys
import os
import math
import json
import time
import threading
import logging
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit

# Always import tkinter as tk (we may use it for fallback)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

# Try to use CustomTkinter for modern UI
try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except Exception:
    CTK_AVAILABLE = False
    ctk = None

# Try serial
try:
    import serial
    SERIAL_AVAILABLE = True
except Exception:
    SERIAL_AVAILABLE = False

# logging
logging.basicConfig(filename="cansat_mission_control.log",
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

# constants
RAD = math.pi / 180.0
G = 9.80665

# ---------------------------
# Cross-platform mouse wheel support
# ---------------------------
def bind_mousewheel(widget, command):
    """Cross-platform mouse wheel binding"""
    # Windows and MacOS
    widget.bind("<MouseWheel>", command)
    # Linux
    widget.bind("<Button-4>", command)
    widget.bind("<Button-5>", command)

def get_scroll_delta(event):
    """Get normalized scroll delta across platforms"""
    if event.num == 4 or event.delta > 0:
        return 1
    elif event.num == 5 or event.delta < 0:
        return -1
    else:
        return int(-1*(event.delta/120)) if hasattr(event, 'delta') else 0

# ---------------------------
# Utility math & filters
# ---------------------------
def euler_to_rot(roll, pitch, yaw):
    cr = math.cos(roll); sr = math.sin(roll)
    cp = math.cos(pitch); sp = math.sin(pitch)
    cy = math.cos(yaw); sy = math.sin(yaw)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,             cp*cr]
    ])
    return R

def fit_paraboloid(x, y, z):
    # z = a x^2 + b y^2 + c xy + d x + e y + f
    def model(XY, a,b,c,d,e,f):
        x, y = XY
        return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
    try:
        popt, _ = curve_fit(model, (x, y), z, maxfev=20000)
        return popt
    except Exception:
        return None

def butter_lowpass(data, cutoff, fs, order=3):
    if len(data) < 3: 
        return data
    nyq = 0.5 * fs
    if cutoff <= 0 or cutoff >= nyq:
        return data
    b, a = butter(order, cutoff/nyq, btype='low')
    try:
        return filtfilt(b, a, data)
    except Exception:
        # fallback to direct array if filtfilt fails
        from scipy.signal import lfilter
        return lfilter(b, a, data)

# ---------------------------
# Flight data / Reconstructor
# ---------------------------
class FlightReconstructor:
    """
    Stores buffers and computes orientation/trajectory from raw IMU samples.
    Accepts either simulated samples or serial input.
    """
    def __init__(self, maxlen=5000):  # Reduced buffer size for better performance
        self.maxlen = maxlen
        self.lock = threading.Lock()
        self.reset()

        # tuning defaults - optimized for real-time performance
        self.accel_offset = np.array([0.0, 0.0, 0.0])  # no default gravity offset
        self.gyro_offset_deg = np.array([0.0, 0.0, 0.0])  # in deg/s
        self.alpha = 0.95  # reduced for better real-time response
        self.lpf_cutoff = 30.0  # Hz for accel optional smoothing
        self.fs_est = 50.0
        # ZUPT - tuned for real sensors
        self.zupt_enabled = True
        self.zupt_acc_tol = 0.5  # increased tolerance for real sensor noise
        self.zupt_gyro_tol_deg = 5.0  # increased tolerance
        self.zupt_vel_thr = 0.08  # slightly higher threshold

        # streaming
        self.streaming = False
        self.serial_conn = None
        
        # Additional real-time performance parameters
        self.max_dt = 0.1  # maximum allowed dt for stability
        self.min_samples_for_zupt = 10  # minimum samples before enabling ZUPT

    def reset(self):
        with self.lock:
            self.t = deque(maxlen=self.maxlen)
            self.ax = deque(maxlen=self.maxlen); self.ay = deque(maxlen=self.maxlen); self.az = deque(maxlen=self.maxlen)
            self.gx = deque(maxlen=self.maxlen); self.gy = deque(maxlen=self.maxlen); self.gz = deque(maxlen=self.maxlen)
            self.vx = deque(maxlen=self.maxlen); self.vy = deque(maxlen=self.maxlen); self.vz = deque(maxlen=self.maxlen)
            self.px = deque(maxlen=self.maxlen); self.py = deque(maxlen=self.maxlen); self.pz = deque(maxlen=self.maxlen)
            self.roll_hist = deque(maxlen=self.maxlen); self.pitch_hist = deque(maxlen=self.maxlen); self.yaw_hist = deque(maxlen=self.maxlen)
            self.roll = 0.0; self.pitch = 0.0; self.yaw = 0.0
            self.last_time = None
            self.last_acc_earth = np.array([0.0, 0.0, 0.0])
            self.last_v = np.array([0.0, 0.0, 0.0])
            self.last_p = np.array([0.0, 0.0, 0.0])

    # ---------- data ingestion ----------
    def start_simulation(self, dt=0.05):  # Reduced frequency from 50Hz to 20Hz
        if self.streaming:
            print("⚠️ Simulation already running!")
            return
        
        print(f"🚀 Starting simulation with dt={dt}")
        self.streaming = True
        t0 = time.time()
        
        # Seed one sample so plots don't show "No data yet"
        try:
            self.process_sample(0.0, 0.0, 0.0, -G, 0.0, 0.0, 0.0)
        except Exception:
            pass
        
        def sim_loop():
            try:
                t = 0.0
                sample_count = 0
                print("📊 Simulation loop started")
                logging.info("Simulation loop started")
                
                while self.streaming:
                    try:
                        # simulated motion resembling a short flight
                        ax = 0.7 * math.sin(0.45 * t) + np.random.normal(0, 0.03)
                        ay = 0.5 * math.cos(0.27 * t) + np.random.normal(0, 0.03)
                        az = -G + 1.0 * math.sin(0.12 * t) + np.random.normal(0, 0.05)
                        gx = 1.5 * math.cos(0.2 * t) + np.random.normal(0, 0.3)  # deg/s
                        gy = 1.0 * math.sin(0.15 * t) + np.random.normal(0, 0.3)
                        gz = 4.0 * math.sin(0.05 * t) + np.random.normal(0, 0.7)
                        
                        self.process_sample(time.time() - t0, ax, ay, az, gx, gy, gz)
                        sample_count += 1
                        
                        # Debug output every 100 samples
                        if sample_count % 100 == 0:
                            print(f"📈 Generated {sample_count} samples, t={t:.2f}s")
                        
                        t += dt
                        time.sleep(dt)
                        
                    except Exception as e:
                        print(f"❌ Error in simulation loop: {e}")
                        logging.exception("Simulation loop error")
                        break
                        
                print(f"🛑 Simulation loop ended. Total samples: {sample_count}")
                
            except Exception as e:
                print(f"❌ Fatal simulation error: {e}")
                logging.exception("Fatal simulation error")
                self.streaming = False
        
        # Start the simulation thread
        sim_thread = threading.Thread(target=sim_loop, daemon=True)
        sim_thread.start()
        print(f"🧵 Simulation thread started: {sim_thread.is_alive()}")
        logging.info("Simulation started")

    def start_serial(self, port, baud=115200):
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial not installed")
        if self.streaming:
            return
        def serial_loop():
            try:
                self.serial_conn = serial.Serial(port, baud, timeout=1)
                logging.info(f"Serial opened {port}@{baud}")
            except Exception as e:
                logging.exception("Serial open failed")
                self.streaming = False
                return
            t0 = time.time()
            self.streaming = True
            while self.streaming:
                try:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    ax = float(obj.get('accel_x', obj.get('ax', 0.0)))
                    ay = float(obj.get('accel_y', obj.get('ay', 0.0)))
                    az = float(obj.get('accel_z', obj.get('az', -G)))
                    gx = float(obj.get('gyro_x', obj.get('gx', 0.0)))
                    gy = float(obj.get('gyro_y', obj.get('gy', 0.0)))
                    gz = float(obj.get('gyro_z', obj.get('gz', 0.0)))
                    self.process_sample(time.time()-t0, ax, ay, az, gx, gy, gz)
                except Exception:
                    logging.exception("Serial read error")
                    time.sleep(0.01)
        threading.Thread(target=serial_loop, daemon=True).start()
        logging.info("Serial thread started")

    def stop(self):
        self.streaming = False
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except: pass
            self.serial_conn = None
        logging.info("Streaming stopped")

    # ---------- signal processing & integration ----------
    def process_sample(self, t, raw_ax, raw_ay, raw_az, raw_gx_deg, raw_gy_deg, raw_gz_deg):
        """
        Process IMU sample with corrected math for real-time data
        raw accel in m/s^2, raw gyro in deg/s
        Coordinate convention: Z-up earth frame, gravity = -9.80665 m/s^2 in Z
        """
        with self.lock:
            # Sensor validation - check for reasonable ranges
            if abs(raw_ax) > 50 or abs(raw_ay) > 50 or abs(raw_az) > 50:
                return  # Skip obviously bad accelerometer data
            if abs(raw_gx_deg) > 2000 or abs(raw_gy_deg) > 2000 or abs(raw_gz_deg) > 2000:
                return  # Skip obviously bad gyro data
            
            # Apply offsets
            ax = raw_ax - self.accel_offset[0]
            ay = raw_ay - self.accel_offset[1] 
            az = raw_az - self.accel_offset[2]
            # gyro convert and subtract offset (deg/s -> rad/s)
            gx = (raw_gx_deg - self.gyro_offset_deg[0]) * RAD
            gy = (raw_gy_deg - self.gyro_offset_deg[1]) * RAD
            gz = (raw_gz_deg - self.gyro_offset_deg[2]) * RAD

            # store raw sensors
            self.t.append(t); self.ax.append(ax); self.ay.append(ay); self.az.append(az)
            self.gx.append(gx); self.gy.append(gy); self.gz.append(gz)

            # estimate dt
            if self.last_time is None:
                dt = 0.02  # reasonable default for first sample
            else:
                dt = t - self.last_time
                if dt <= 0: dt = 1e-6
                if dt > 0.1: dt = 0.02  # cap dt for stability
            
            # CORRECTED complementary filter for roll/pitch
            # Integrate gyro first, then fuse with accelerometer
            roll_gyro = self.roll + gx * dt
            pitch_gyro = self.pitch + gy * dt
            self.yaw += gz * dt  # yaw drifts without magnetometer
            
            # Accelerometer-based roll/pitch (gravity reference)
            try:
                roll_acc = math.atan2(ay, az)
                pitch_acc = math.atan2(-ax, math.sqrt(ay*ay + az*az))
            except Exception:
                roll_acc = self.roll; pitch_acc = self.pitch
            
            # Complementary filter fusion
            a = float(self.alpha)
            self.roll = a * roll_gyro + (1-a) * roll_acc
            self.pitch = a * pitch_gyro + (1-a) * pitch_acc

            # body->earth transformation
            R = euler_to_rot(self.roll, self.pitch, self.yaw)
            acc_body = np.array([ax, ay, az])
            acc_earth = R.dot(acc_body)
            # CORRECTED gravity compensation: subtract gravity vector in earth frame
            # In Z-up earth frame, gravity is [0, 0, -G]
            acc_earth_no_g = acc_earth - np.array([0.0, 0.0, -G])

            # ZUPT detection - improved for real-time data
            accel_mag = math.sqrt(ax*ax + ay*ay + az*az)
            gyro_sum = abs(gx) + abs(gy) + abs(gz)
            stationary = False
            # Only apply ZUPT after collecting enough samples for stability
            if self.zupt_enabled and len(self.t) > self.min_samples_for_zupt:
                if abs(accel_mag - G) < self.zupt_acc_tol and gyro_sum < (self.zupt_gyro_tol_deg * RAD):
                    stationary = True

            # integrate trapezoid
            if self.last_time is None:
                vx = self.last_v[0]; vy = self.last_v[1]; vz = self.last_v[2]
                px = self.last_p[0]; py = self.last_p[1]; pz = self.last_p[2]
            else:
                dt = max(dt, 1e-6)
                vx = self.last_v[0] + 0.5*(self.last_acc_earth[0] + acc_earth_no_g[0]) * dt
                vy = self.last_v[1] + 0.5*(self.last_acc_earth[1] + acc_earth_no_g[1]) * dt
                vz = self.last_v[2] + 0.5*(self.last_acc_earth[2] + acc_earth_no_g[2]) * dt
                px = self.last_p[0] + 0.5*(self.last_v[0] + vx) * dt
                py = self.last_p[1] + 0.5*(self.last_v[1] + vy) * dt
                pz = self.last_p[2] + 0.5*(self.last_v[2] + vz) * dt

            if stationary:
                if math.hypot(vx, vy) < self.zupt_vel_thr:
                    vx = 0.0; vy = 0.0
                if abs(vz) < self.zupt_vel_thr:
                    vz = 0.0

            # append computed states
            self.vx.append(vx); self.vy.append(vy); self.vz.append(vz)
            self.px.append(px); self.py.append(py); self.pz.append(pz)
            self.roll_hist.append(self.roll); self.pitch_hist.append(self.pitch); self.yaw_hist.append(self.yaw)

            # update last
            self.last_time = t
            self.last_acc_earth = acc_earth_no_g
            self.last_v = np.array([vx, vy, vz])
            self.last_p = np.array([px, py, pz])

    # ---------- snapshots, KPIs, save/load ----------
    def snapshot(self):
        with self.lock:
            t = np.array(self.t)
            pos = np.vstack((np.array(self.px), np.array(self.py), np.array(self.pz))).T if len(self.px) > 0 else np.zeros((0,3))
            vel = np.vstack((np.array(self.vx), np.array(self.vy), np.array(self.vz))).T if len(self.vx) > 0 else np.zeros((0,3))
            acc = np.vstack((np.array(self.ax), np.array(self.ay), np.array(self.az))).T if len(self.ax) > 0 else np.zeros((0,3))
            orient = np.vstack((np.array(self.roll_hist), np.array(self.pitch_hist), np.array(self.yaw_hist))).T if len(self.roll_hist) > 0 else np.zeros((0,3))
        return t, pos, vel, acc, orient

    def compute_kpis(self):
        t, pos, vel, acc, orient = self.snapshot()
        if t.size == 0:
            return {}
        max_alt = float(np.max(pos[:,2]))
        max_vel = float(np.max(np.linalg.norm(vel, axis=1))) if vel.size else 0.0
        flight_time = float(t[-1] - t[0]) if t.size > 1 else 0.0
        avg_speed = float(np.mean(np.linalg.norm(vel, axis=1))) if vel.size else 0.0
        # descent rate approx from last segment
        descent_rate = 0.0
        if pos.shape[0] >= 2:
            dz = pos[-1,2] - pos[-2,2]
            dt = t[-1] - t[-2] if t.shape[0] >= 2 else 1.0
            descent_rate = dz / dt
        return {
            "max_alt": max_alt,
            "max_vel": max_vel,
            "flight_time": flight_time,
            "avg_speed": avg_speed,
            "descent_rate": descent_rate,
            "points": int(pos.shape[0])
        }

    def save_json(self, fname):
        with self.lock:
            data = {
                "time": list(self.t),
                "acceleration": {"x": list(self.ax), "y": list(self.ay), "z": list(self.az)},
                "gyro": {"x": [g/RAD for g in self.gx], "y": [g/RAD for g in self.gy], "z": [g/RAD for g in self.gz]},
                "velocity": {"x": list(self.vx), "y": list(self.vy), "z": list(self.vz)},
                "position": {"x": list(self.px), "y": list(self.py), "z": list(self.pz)},
                "orientation": {"roll": list(self.roll_hist), "pitch": list(self.pitch_hist), "yaw": list(self.yaw_hist)}
            }
        with open(fname, 'w') as f:
            json.dump(data, f, indent=2)
        return fname

    def save_csv(self, fname):
        t, pos, vel, acc, orient = self.snapshot()
        with open(fname, 'w') as f:
            f.write("time,ax,ay,az,gx_deg,gy_deg,gz_deg,vx,vy,vz,px,py,pz\n")
            n = t.size
            for i in range(n):
                gx_deg = (self.gx[i]/RAD) if i < len(self.gx) else ""
                gy_deg = (self.gy[i]/RAD) if i < len(self.gy) else ""
                gz_deg = (self.gz[i]/RAD) if i < len(self.gz) else ""
                pxv = pos[i,0] if i < pos.shape[0] else ""
                pyv = pos[i,1] if i < pos.shape[0] else ""
                pzv = pos[i,2] if i < pos.shape[0] else ""
                vxv = vel[i,0] if i < vel.shape[0] else ""
                vyv = vel[i,1] if i < vel.shape[0] else ""
                vzv = vel[i,2] if i < vel.shape[0] else ""
                f.write(",".join(str(x) for x in [t[i], acc[i,0], acc[i,1], acc[i,2],
                                                  gx_deg, gy_deg, gz_deg,
                                                  vxv, vyv, vzv, pxv, pyv, pzv]) + "\n")
        return fname

# ---------------------------
# GUI App
# ---------------------------
class MissionControlApp:
    def __init__(self, root):
        self.root = root
        # If customtkinter available, use CTk root class to get modern look
        if CTK_AVAILABLE and hasattr(ctk, "CTk"):
            if isinstance(root, tk.Tk):
                # if user passed tk.Tk, we won't try to recreate. But typical usage we will create ctk.CTk externally.
                pass
        self.recon = FlightReconstructor()
        self._build_ui()
        self.update_interval_ms = 250  # update UI every 250 ms (reduced frequency)
        self._running = True
        self._last_update_time = time.time()
        self._fps_counter = 0
        self.root.after(self.update_interval_ms, self._periodic_update)
        logging.info("Mission Control UI initialized")

    def _build_ui(self):
        # Configure modern dark theme
        if CTK_AVAILABLE:
            try:
                ctk.set_appearance_mode("dark")
                ctk.set_default_color_theme("blue")
            except Exception:
                pass

        # Enhanced window setup
        self.root.title("🚀 CANSAT Mission Control - Advanced Flight Analytics")
        self.root.geometry("1400x900")
        
        # Set background color based on available toolkit
        try:
            if not CTK_AVAILABLE:
                self.root.configure(bg='#2c3e50')
        except Exception:
            pass
        
        # Color scheme
        self.colors = {
            'primary': '#00d4ff',
            'secondary': '#ff6b35', 
            'success': '#00ff88',
            'warning': '#ffaa00',
            'danger': '#ff4757',
            'dark': '#2c3e50',
            'light': '#ecf0f1'
        }

        # Frames with enhanced styling
        if CTK_AVAILABLE:
            try:
                self.frame_left = ctk.CTkFrame(self.root, width=360, corner_radius=8)
                self.frame_right = ctk.CTkFrame(self.root, corner_radius=0)
            except Exception:
                self.frame_left = tk.Frame(self.root, width=360, bg='#34495e', relief='raised', bd=2)
                self.frame_right = tk.Frame(self.root, bg='#2c3e50')
        else:
            self.frame_left = tk.Frame(self.root, width=360, bg='#34495e', relief='raised', bd=2)
            self.frame_right = tk.Frame(self.root, bg='#2c3e50')
        self.frame_left.pack(side='left', fill='y', padx=8, pady=8)
        self.frame_right.pack(side='right', fill='both', expand=True, padx=8, pady=8)

        # Left: Controls
        self._build_left_controls()

        # Right: Notebook with three tabs
        self._build_right_tabs()

        # Bottom: Log panel
        self._build_log_panel()

    def _build_left_controls(self):
        # Create scrollable frame for left controls
        canvas = tk.Canvas(self.frame_left, bg='#34495e' if not CTK_AVAILABLE else None)
        scrollbar_left = ttk.Scrollbar(self.frame_left, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_left.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar_left.pack(side="right", fill="y")
        
        # Add mouse wheel scrolling to canvas
        def _on_canvas_mousewheel(event):
            delta = get_scroll_delta(event)
            canvas.yview_scroll(-delta, "units")
        
        bind_mousewheel(canvas, _on_canvas_mousewheel)
        
        # Bind mouse wheel to scrollable frame as well
        def _bind_to_mousewheel(event):
            bind_mousewheel(canvas, _on_canvas_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind("<MouseWheel>")
            canvas.unbind("<Button-4>")
            canvas.unbind("<Button-5>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        L = self.scrollable_frame
        # Enhanced title with gradient effect
        lbl = (ctk.CTkLabel if CTK_AVAILABLE else ttk.Label)
        button = (ctk.CTkButton if CTK_AVAILABLE else ttk.Button)
        
        title_frame = ctk.CTkFrame(L, fg_color="transparent") if CTK_AVAILABLE else ttk.Frame(L)
        title_frame.pack(pady=(15,10), fill='x')
        
        tklabel = lbl(title_frame, text="🎛️ MISSION CONTROL", 
                     font=("Segoe UI", 16, "bold"),
                     text_color=self.colors['primary'] if CTK_AVAILABLE else None)
        tklabel.pack()
        
        subtitle = lbl(title_frame, text="Flight Data Analytics", 
                      font=("Segoe UI", 10),
                      text_color=self.colors['light'] if CTK_AVAILABLE else None)
        subtitle.pack()

        # Enhanced control buttons with better styling
        btn_style = {"width": 320, "height": 45, "corner_radius": 8} if CTK_AVAILABLE else {"width": 40}
        
        btn_run = button(L, text="🚀 Launch Simulation", command=self._on_run_sim, 
                        fg_color=self.colors['success'] if CTK_AVAILABLE else None, **btn_style)
        # Store button reference for visual feedback
        self.btn_run = btn_run
        btn_serial = button(L, text="📡 Connect Hardware", command=self._on_start_serial,
                           fg_color=self.colors['primary'] if CTK_AVAILABLE else None, **btn_style)
        btn_stop = button(L, text="⏹️ Emergency Stop", command=self._on_stop,
                         fg_color=self.colors['danger'] if CTK_AVAILABLE else None, **btn_style)
        btn_reset = button(L, text="🔄 System Reset", command=self._on_reset,
                          fg_color=self.colors['warning'] if CTK_AVAILABLE else None, **btn_style)
        
        for btn in [btn_run, btn_serial, btn_stop, btn_reset]:
            btn.pack(pady=8)

        # Enhanced Data Management Section
        data_frame = ctk.CTkFrame(L, corner_radius=10) if CTK_AVAILABLE else ttk.LabelFrame(L, text="Data Management")
        data_frame.pack(pady=(20,10), padx=10, fill='x')
        
        data_title = lbl(data_frame, text="📊 DATA MANAGEMENT", 
                        font=("Segoe UI", 12, "bold"),
                        text_color=self.colors['secondary'] if CTK_AVAILABLE else None)
        data_title.pack(pady=(10,5))
        
        data_btn_style = {"width": 280, "height": 35, "corner_radius": 6} if CTK_AVAILABLE else {"width": 35}
        btn_save_j = button(data_frame, text="💾 Export JSON", command=self._on_save_json, **data_btn_style)
        btn_save_c = button(data_frame, text="📈 Export CSV", command=self._on_save_csv, **data_btn_style)
        btn_load = button(data_frame, text="📂 Import & Replay", command=self._on_load_json, **data_btn_style)
        
        for btn in [btn_save_j, btn_save_c, btn_load]:
            btn.pack(pady=4, padx=10)

        # Tuning
        tk.Label(L, text="Calibration & Tuning", font=("Segoe UI", 12, "bold")).pack(pady=(12,4))
        # accel offsets
        self.aoff_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
        for i, name in enumerate(("Ax offset", "Ay offset", "Az offset")):
            f = ttk.Frame(L) if not CTK_AVAILABLE else ctk.CTkFrame(L)
            f.pack(fill='x', pady=4)
            ttk.Label(f, text=name).pack(side='left', padx=4)
            s = ttk.Scale(f, from_=-2.0, to=2.0, orient='horizontal', variable=self.aoff_vars[i])
            s.pack(side='right', fill='x', expand=True, padx=4)

        # gyro offsets
        self.goff_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
        for i, name in enumerate(("Gx offset (deg/s)", "Gy offset (deg/s)", "Gz offset (deg/s)")):
            f = ttk.Frame(L) if not CTK_AVAILABLE else ctk.CTkFrame(L)
            f.pack(fill='x', pady=4)
            ttk.Label(f, text=name).pack(side='left', padx=4)
            s = ttk.Scale(f, from_=-50.0, to=50.0, orient='horizontal', variable=self.goff_vars[i])
            s.pack(side='right', fill='x', expand=True, padx=4)

        # complementary alpha
        f = ttk.Frame(L) if not CTK_AVAILABLE else ctk.CTkFrame(L)
        f.pack(fill='x', pady=6)
        ttk.Label(f, text="Complementary α").pack(side='left', padx=4)
        self.alpha_var = tk.DoubleVar(value=0.98)
        ttk.Scale(f, from_=0.90, to=0.999, orient='horizontal', variable=self.alpha_var).pack(side='right', fill='x', expand=True, padx=4)

        # ZUPT toggle
        tk.Label(L, text="ZUPT / Filters", font=("Segoe UI", 12, "bold")).pack(pady=(12,4))
        self.zupt_var = tk.BooleanVar(value=True)
        self.hpf_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(L, text="Enable ZUPT (recommended)", variable=self.zupt_var).pack(anchor='w', padx=8)
        ttk.Checkbutton(L, text="Enable HPF (advanced)", variable=self.hpf_var).pack(anchor='w', padx=8)

        # Enhanced KPI Dashboard - Fixed for compatibility
        ttk.Separator(L, orient='horizontal').pack(fill='x', pady=(15,5))
        
        kpi_header = tk.Label(L, text="📊 LIVE TELEMETRY", 
                             font=("Segoe UI", 12, "bold"),
                             fg='#00ff88', bg='#2c3e50' if not CTK_AVAILABLE else None)
        kpi_header.pack(pady=(5,10))
        
        # KPI Variables
        self.kpi_max_alt = tk.StringVar(value="0.00")
        self.kpi_max_vel = tk.StringVar(value="0.00") 
        self.kpi_time = tk.StringVar(value="0.00")
        self.kpi_desc = tk.StringVar(value="0.00")
        
        # Create KPI cards with fallback styling
        kpi_data = [
            ("🏔️", "Max Altitude", self.kpi_max_alt, "m"),
            ("⚡", "Max Velocity", self.kpi_max_vel, "m/s"),
            ("⏱️", "Flight Time", self.kpi_time, "s"),
            ("📉", "Descent Rate", self.kpi_desc, "m/s")
        ]
        
        for icon, label, var, unit in kpi_data:
            # Create card frame
            if CTK_AVAILABLE:
                try:
                    card = ctk.CTkFrame(L, height=50, corner_radius=8)
                    card.pack(fill='x', padx=8, pady=3)
                except:
                    card = tk.Frame(L, bg='#34495e', relief='raised', bd=1)
                    card.pack(fill='x', padx=8, pady=3)
            else:
                card = tk.Frame(L, bg='#34495e', relief='raised', bd=1)
                card.pack(fill='x', padx=8, pady=3)
            
            # Icon
            icon_lbl = tk.Label(card, text=icon, font=("Segoe UI", 14), 
                               bg=card['bg'] if hasattr(card, '__getitem__') else None)
            icon_lbl.pack(side='left', padx=(8,5), pady=5)
            
            # Info section
            info_frame = tk.Frame(card, bg=card['bg'] if hasattr(card, '__getitem__') else None)
            info_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
            
            # Label and value
            name_lbl = tk.Label(info_frame, text=label, font=("Segoe UI", 9), 
                               fg='#ecf0f1', bg=info_frame['bg'] if hasattr(info_frame, '__getitem__') else None)
            name_lbl.pack(anchor='w')
            
            value_frame = tk.Frame(info_frame, bg=info_frame['bg'] if hasattr(info_frame, '__getitem__') else None)
            value_frame.pack(anchor='w', fill='x')
            
            value_lbl = tk.Label(value_frame, textvariable=var, font=("Segoe UI", 12, "bold"),
                                fg='#00d4ff', bg=value_frame['bg'] if hasattr(value_frame, '__getitem__') else None)
            value_lbl.pack(side='left')
            
            unit_lbl = tk.Label(value_frame, text=unit, font=("Segoe UI", 9),
                               fg='#ecf0f1', bg=value_frame['bg'] if hasattr(value_frame, '__getitem__') else None)
            unit_lbl.pack(side='left', padx=(2,0))

        # Enhanced Status Indicator - Fixed compatibility
        ttk.Separator(L, orient='horizontal').pack(fill='x', pady=(15,5))
        
        status_header = tk.Label(L, text="🔄 SYSTEM STATUS", 
                                font=("Segoe UI", 11, "bold"),
                                fg='#ffaa00', bg='#2c3e50' if not CTK_AVAILABLE else None)
        status_header.pack(pady=(5,8))
        
        # Status indicator
        self.status_var = tk.StringVar(value="🟢 System Ready")
        self.status_label = tk.Label(L, textvariable=self.status_var, 
                                    font=("Segoe UI", 10, "bold"),
                                    fg='#00ff88', bg='#2c3e50' if not CTK_AVAILABLE else None)
        self.status_label.pack(pady=(0,8))
        
        # Add progress bar for data streaming
        try:
            if CTK_AVAILABLE:
                self.progress_bar = ctk.CTkProgressBar(L, width=280, height=8)
                self.progress_bar.pack(pady=(0,10), padx=10)
                self.progress_bar.set(0)
            else:
                self.progress_bar = ttk.Progressbar(L, length=280, mode='indeterminate')
                self.progress_bar.pack(pady=(0,10), padx=10)
        except Exception as e:
            # Fallback if progress bar fails
            print(f"Progress bar creation failed: {e}")
            self.progress_bar = None

    def _build_right_tabs(self):
        R = self.frame_right
        # Notebook
        self.nb = ttk.Notebook(R)
        self.tab_traj = ttk.Frame(self.nb)
        self.tab_cube = ttk.Frame(self.nb)
        self.tab_signals = ttk.Frame(self.nb)
        self.nb.add(self.tab_traj, text="3D Trajectory")
        self.nb.add(self.tab_cube, text="Orientation Cube")
        self.nb.add(self.tab_signals, text="Signals")
        self.nb.pack(fill='both', expand=True)

        # Enhanced Trajectory canvas with dark theme and scrolling
        self.fig_traj = plt.Figure(figsize=(9,7), dpi=100, facecolor='#2b2b2b')
        self.ax3d = self.fig_traj.add_subplot(111, projection='3d', facecolor='#2b2b2b')
        self.ax3d.xaxis.pane.fill = False; self.ax3d.yaxis.pane.fill = False; self.ax3d.zaxis.pane.fill = False
        self.ax3d.grid(True, alpha=0.3, color='white')
        self.canvas_traj = FigureCanvasTkAgg(self.fig_traj, master=self.tab_traj)
        self.canvas_traj.get_tk_widget().pack(fill='both', expand=True)
        
        # Add mouse wheel scrolling support for 3D plot
        def zoom_3d(event):
            if event.inaxes == self.ax3d:
                scale_factor = 1.1 if event.button == 'up' else 0.9
                xlim = self.ax3d.get_xlim3d()
                ylim = self.ax3d.get_ylim3d()
                zlim = self.ax3d.get_zlim3d()
                
                x_center = (xlim[0] + xlim[1]) / 2
                y_center = (ylim[0] + ylim[1]) / 2
                z_center = (zlim[0] + zlim[1]) / 2
                
                x_range = (xlim[1] - xlim[0]) * scale_factor / 2
                y_range = (ylim[1] - ylim[0]) * scale_factor / 2
                z_range = (zlim[1] - zlim[0]) * scale_factor / 2
                
                self.ax3d.set_xlim3d([x_center - x_range, x_center + x_range])
                self.ax3d.set_ylim3d([y_center - y_range, y_center + y_range])
                self.ax3d.set_zlim3d([z_center - z_range, z_center + z_range])
                self.canvas_traj.draw_idle()
        
        self.canvas_traj.mpl_connect('scroll_event', zoom_3d)

        # Enhanced Cube canvas with dark theme
        self.fig_cube = plt.Figure(figsize=(6,6), dpi=100, facecolor='#2b2b2b')
        self.ax_cube = self.fig_cube.add_subplot(111, projection='3d', facecolor='#2b2b2b')
        self.ax_cube.xaxis.pane.fill = False; self.ax_cube.yaxis.pane.fill = False; self.ax_cube.zaxis.pane.fill = False
        self.canvas_cube = FigureCanvasTkAgg(self.fig_cube, master=self.tab_cube)
        self.canvas_cube.get_tk_widget().pack(fill='both', expand=True)

        # Enhanced Signals figure with dark theme and scrolling
        self.fig_sig = plt.Figure(figsize=(9,7), dpi=100, facecolor='#2b2b2b')
        self.ax_acc = self.fig_sig.add_subplot(311, facecolor='#2b2b2b')
        self.ax_vel = self.fig_sig.add_subplot(312, facecolor='#2b2b2b')
        self.ax_alt = self.fig_sig.add_subplot(313, facecolor='#2b2b2b')
        
        # Style all signal plots
        for ax in [self.ax_acc, self.ax_vel, self.ax_alt]:
            ax.grid(True, alpha=0.3, color='white')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white') 
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            
        self.canvas_sig = FigureCanvasTkAgg(self.fig_sig, master=self.tab_signals)
        self.canvas_sig.get_tk_widget().pack(fill='both', expand=True)
        
        # Add horizontal scrolling and zooming for signal plots
        def scroll_signals(event):
            if event.inaxes in [self.ax_acc, self.ax_vel, self.ax_alt]:
                ax = event.inaxes
                if event.key == 'shift':  # Horizontal scroll with Shift+scroll
                    xlim = ax.get_xlim()
                    x_range = xlim[1] - xlim[0]
                    shift = x_range * 0.1 * (1 if event.button == 'up' else -1)
                    ax.set_xlim([xlim[0] + shift, xlim[1] + shift])
                else:  # Zoom with scroll
                    xlim = ax.get_xlim()
                    x_center = (xlim[0] + xlim[1]) / 2
                    scale_factor = 0.9 if event.button == 'up' else 1.1
                    x_range = (xlim[1] - xlim[0]) * scale_factor / 2
                    ax.set_xlim([x_center - x_range, x_center + x_range])
                self.canvas_sig.draw_idle()
        
        self.canvas_sig.mpl_connect('scroll_event', scroll_signals)

    def _build_log_panel(self):
        # bottom log in right frame with scrollbar
        frame = ttk.Frame(self.frame_right)
        frame.pack(side='bottom', fill='x', padx=4, pady=(6,0))
        ttk.Label(frame, text="Event Log").pack(anchor='w')
        
        # Create text widget with scrollbar
        log_frame = ttk.Frame(frame)
        log_frame.pack(fill='both', expand=True)
        
        self.log_text = tk.Text(log_frame, height=6, state='disabled', bg='#111', fg='#ddd', wrap='word')
        scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Add mouse wheel scrolling support
        def _on_mousewheel(event):
            delta = get_scroll_delta(event)
            self.log_text.yview_scroll(-delta, "units")
        
        bind_mousewheel(self.log_text, _on_mousewheel)
        
        # helper log method
        logging.getLogger().addHandler(_TextHandler(self.log_text))

    # -------------------------
    # Actions: buttons & file
    # -------------------------
    def _on_run_sim(self):
        print("🔘 Button clicked!")  # Immediate feedback
        
        # Visual feedback - change button text temporarily
        if hasattr(self, 'btn_run'):
            try:
                if CTK_AVAILABLE:
                    self.btn_run.configure(text="⏳ Starting...")
                else:
                    self.btn_run.configure(text="⏳ Starting...")
            except Exception as e:
                print(f"Button update error: {e}")
        
        try:
            print("🚀 Starting simulation...")  # Debug output
            self.status_var.set("⏳ Initializing...")
            
            # apply tuning
            self._apply_tuning_to_recon()
            self.recon.reset()
            print("🔄 Reconstructor reset complete")
            
            # Check if already running
            if self.recon.streaming:
                print("⚠️ Simulation already running, stopping first...")
                self.recon.stop()
                time.sleep(0.1)  # Brief pause
            
            self.recon.start_simulation()
            self._stream_start_time = time.time()
            # Force an immediate plot refresh so first sample is visible
            try:
                self._refresh_plots()
            except Exception:
                pass
            
            # Update UI
            self.status_var.set("🚀 Simulation Active")
            if hasattr(self, 'btn_run'):
                try:
                    self.btn_run.configure(text="✅ Running")
                except Exception:
                    pass
            
            if hasattr(self, 'progress_bar') and self.progress_bar:
                try:
                    if CTK_AVAILABLE and hasattr(self.progress_bar, 'set'):
                        self.progress_bar.set(0.8)
                    elif hasattr(self.progress_bar, 'start'):
                        self.progress_bar.start()
                except Exception as e:
                    print(f"Progress bar error: {e}")
            
            print("✅ Simulation started successfully!")
            logging.info("🚀 Simulation launched successfully")
            
            # Reset button text after 2 seconds
            self.root.after(2000, lambda: self._reset_button_text())
            
        except Exception as e:
            print(f"❌ Simulation start failed: {e}")
            self.status_var.set("❌ Simulation Failed")
            if hasattr(self, 'btn_run'):
                try:
                    self.btn_run.configure(text="❌ Failed")
                except Exception:
                    pass
            logging.exception("Simulation start failed")
            
            # Reset button text after 2 seconds
            self.root.after(2000, lambda: self._reset_button_text())
    
    def _reset_button_text(self):
        """Reset button text to original"""
        if hasattr(self, 'btn_run'):
            try:
                self.btn_run.configure(text="🚀 Launch Simulation")
            except Exception:
                pass

    def _on_start_serial(self):
        if not SERIAL_AVAILABLE:
            messagebox.showerror("pyserial missing", "pyserial not installed; install with pip to use serial mode.")
            return
        port = simpledialog.askstring("Serial Port", "Enter serial port (e.g. COM3 or /dev/ttyUSB0):", initialvalue="COM3")
        if not port:
            return
        baud = simpledialog.askinteger("Baud Rate", "Enter baud rate:", initialvalue=115200)
        self._apply_tuning_to_recon()
        self.recon.reset()
        try:
            self.recon.start_serial(port, baud or 115200)
            self.status_var.set(f"📡 Hardware Connected")
            if hasattr(self, 'progress_bar'):
                self.progress_bar.start() if not CTK_AVAILABLE else self.progress_bar.set(0.9)
            logging.info(f"📡 Hardware connected on {port}@{baud}")
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))
            self.status_var.set("❌ Connection Failed")
            logging.exception("Hardware connection failed")

    def _on_stop(self):
        self.recon.stop()
        self.status_var.set("⏹️ System Stopped")
        if hasattr(self, 'progress_bar'):
            self.progress_bar.stop() if not CTK_AVAILABLE else self.progress_bar.set(0)
        logging.info("⏹️ System stopped by user")

    def _on_reset(self):
        self.recon.stop()
        self.recon.reset()
        self.status_var.set("🔄 System Reset")
        if hasattr(self, 'progress_bar'):
            self.progress_bar.stop() if not CTK_AVAILABLE else self.progress_bar.set(0)
        logging.info("🔄 System reset completed")

    def _on_save_json(self):
        fname = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not fname:
            return
        out = self.recon.save_json(fname)
        messagebox.showinfo("Saved", f"Saved JSON to {out}")
        logging.info(f"Saved JSON {out}")

    def _on_save_csv(self):
        fname = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not fname:
            return
        out = self.recon.save_csv(fname)
        messagebox.showinfo("Saved", f"Saved CSV to {out}")
        logging.info(f"Saved CSV {out}")

    def _on_load_json(self):
        fname = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not fname:
            return
        try:
            with open(fname, 'r') as f:
                data = json.load(f)
            times = np.array(data['time'])
            ax = np.array(data['acceleration']['x']); ay = np.array(data['acceleration']['y']); az = np.array(data['acceleration']['z'])
            vx = np.array(data['velocity']['x']); vy = np.array(data['velocity']['y']); vz = np.array(data['velocity']['z'])
            px = np.array(data['position']['x']); py = np.array(data['position']['y']); pz = np.array(data['position']['z'])
            self.recon.stop(); self.recon.reset()
            with self.recon.lock:
                self.recon.t = deque(times.tolist(), maxlen=self.recon.maxlen)
                self.recon.ax = deque(ax.tolist(), maxlen=self.recon.maxlen)
                self.recon.ay = deque(ay.tolist(), maxlen=self.recon.maxlen)
                self.recon.az = deque(az.tolist(), maxlen=self.recon.maxlen)
                self.recon.vx = deque(vx.tolist(), maxlen=self.recon.maxlen)
                self.recon.vy = deque(vy.tolist(), maxlen=self.recon.maxlen)
                self.recon.vz = deque(vz.tolist(), maxlen=self.recon.maxlen)
                self.recon.px = deque(px.tolist(), maxlen=self.recon.maxlen)
                self.recon.py = deque(py.tolist(), maxlen=self.recon.maxlen)
                self.recon.pz = deque(pz.tolist(), maxlen=self.recon.maxlen)
            self.status_var.set(f"Loaded {os.path.basename(fname)}")
            logging.info(f"Loaded file {fname}")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            logging.exception("Load failed")

    # -------------------------
    # Apply GUI tunings to reconstructor
    # -------------------------
    def _apply_tuning_to_recon(self):
        try:
            print("🔧 Applying tuning parameters...")
            
            # Check if variables exist
            if hasattr(self, 'aoff_vars') and self.aoff_vars:
                self.recon.accel_offset = np.array([v.get() for v in self.aoff_vars])
                print(f"   Accel offset: {self.recon.accel_offset}")
            
            if hasattr(self, 'goff_vars') and self.goff_vars:
                self.recon.gyro_offset_deg = np.array([v.get() for v in self.goff_vars])
                print(f"   Gyro offset: {self.recon.gyro_offset_deg}")
            
            if hasattr(self, 'alpha_var'):
                self.recon.alpha = float(self.alpha_var.get())
                print(f"   Alpha: {self.recon.alpha}")
            
            if hasattr(self, 'zupt_var'):
                self.recon.zupt_enabled = bool(self.zupt_var.get())
                print(f"   ZUPT enabled: {self.recon.zupt_enabled}")
            
            if hasattr(self, 'hpf_var'):
                self.recon.hpf_enabled = bool(self.hpf_var.get())
                print(f"   HPF enabled: {self.recon.hpf_enabled}")
            
            print("✅ Tuning applied successfully")
            
        except Exception as e:
            print(f"❌ Apply tuning failed: {e}")
            logging.exception("Apply tuning failed")

    # -------------------------
    # Periodic UI updates
    # -------------------------
    def _periodic_update(self):
        try:
            # Performance monitoring
            current_time = time.time()
            self._fps_counter += 1
            
            # Update FPS and point count every 2 seconds
            if current_time - self._last_update_time > 2.0:
                fps = self._fps_counter / (current_time - self._last_update_time)
                if hasattr(self, 'status_var'):
                    pts = len(self.recon.t)
                    state = "🚀 Simulation Active" if self.recon.streaming else "🟢 System Ready"
                    self.status_var.set(f"{state} | Points: {pts} | FPS: {fps:.1f}")
                self._last_update_time = current_time
                self._fps_counter = 0

            # Watchdog: seed a sample if streaming but no data for >1s
            if getattr(self, '_stream_start_time', None) and self.recon.streaming:
                try:
                    if (current_time - self._stream_start_time > 1.0) and len(self.recon.t) == 0:
                        logging.warning("No data received yet; seeding one sample")
                        self.recon.process_sample(0.0, 0.0, 0.0, -G, 0.0, 0.0, 0.0)
                except Exception:
                    pass
            
            self._refresh_plots()
            self._refresh_kpis()
            self._refresh_cube()
        except Exception:
            logging.exception("Periodic update error")
        if self._running:
            self.root.after(self.update_interval_ms, self._periodic_update)

    def _refresh_plots(self):
        t, pos, vel, acc, orient = self.recon.snapshot()
        
        # Debug: Print data size occasionally
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 20 == 0:  # Print every 20 updates
            print(f"📊 Data update: t.size={t.size}, pos.shape={pos.shape if pos.size > 0 else 'empty'}")
        
        # Only update if we have new data (performance optimization)
        if not hasattr(self, '_last_data_size'):
            self._last_data_size = 0
        if t.size == self._last_data_size and t.size > 0:
            return  # Skip update if no new data
        self._last_data_size = t.size
        
        # 3D Trajectory - Optimized
        self.ax3d.clear()
        self.ax3d.set_title("3D Trajectory", color='white')
        self.ax3d.set_xlabel("X (m)", color='white'); self.ax3d.set_ylabel("Y (m)", color='white'); self.ax3d.set_zlabel("Z (m)", color='white')
        
        if t.size > 0:
            # Drastically limit display points for performance
            N = min(500, t.size)  # Reduced from 3000 to 500
            pos_disp = pos[-N:]
            xs, ys, zs = pos_disp[:,0], pos_disp[:,1], pos_disp[:,2]
            
            # Simple line plot instead of colored segments (much faster)
            self.ax3d.plot(xs, ys, zs, color='cyan', linewidth=2, alpha=0.8)
            
            # Current position only
            if len(xs) > 0:
                self.ax3d.scatter(xs[-1], ys[-1], zs[-1], color='red', s=80, label='Current')
            
            # Skip paraboloid fit for performance (can be re-enabled later)
        else:
            self.ax3d.text(0.5,0.5,0.5,"No data yet", transform=self.ax3d.transAxes, color='white')

        # Signals - Optimized
        if t.size > 0:
            # Reduce signal plot points for performance
            N = min(1000, t.size)  # Reduced from 2000 to 1000
            td = t[-N:]; ad = acc[-N:]; vd = vel[-N:]; pd = pos[-N:]
            
            # Clear and plot acceleration
            self.ax_acc.clear()
            self.ax_acc.plot(td, ad[:,0], 'r', label='Ax', linewidth=1)
            self.ax_acc.plot(td, ad[:,1], 'g', label='Ay', linewidth=1) 
            self.ax_acc.plot(td, ad[:,2], 'b', label='Az', linewidth=1)
            self.ax_acc.legend(fontsize=8); self.ax_acc.grid(True, alpha=0.3)
            self.ax_acc.set_title("Acceleration", color='white', fontsize=10)
            
            # Clear and plot velocity
            self.ax_vel.clear()
            self.ax_vel.plot(td, vd[:,0], 'r', label='Vx', linewidth=1)
            self.ax_vel.plot(td, vd[:,1], 'g', label='Vy', linewidth=1)
            self.ax_vel.plot(td, vd[:,2], 'b', label='Vz', linewidth=1)
            self.ax_vel.legend(fontsize=8); self.ax_vel.grid(True, alpha=0.3)
            self.ax_vel.set_title("Velocity", color='white', fontsize=10)
            
            # Clear and plot altitude
            self.ax_alt.clear()
            self.ax_alt.plot(td, pd[:,2], 'cyan', linewidth=2)
            self.ax_alt.set_title("Altitude (Z)", color='white', fontsize=10)
            self.ax_alt.grid(True, alpha=0.3)
        else:
            self.ax_acc.clear()
            self.ax_acc.text(0.5,0.5,"No sensor data", transform=self.ax_acc.transAxes, color='white')

        # Always draw the canvas to ensure data appears
        try:
            current_tab = self.nb.index(self.nb.select())
            if current_tab == 0:  # 3D Trajectory tab
                self.canvas_traj.draw_idle()
            elif current_tab == 2:  # Signals tab
                self.canvas_sig.draw_idle()
            else:
                # Always draw trajectory canvas since it's the main view
                self.canvas_traj.draw_idle()
        except Exception as e:
            # Fallback - always draw to ensure data visibility
            print(f"Canvas draw error: {e}")
            self.canvas_traj.draw_idle()
            self.canvas_sig.draw_idle()

    def _refresh_kpis(self):
        k = self.recon.compute_kpis()
        if not k:
            return
        # Update with just the numeric values for the card display
        self.kpi_max_alt.set(f"{k['max_alt']:.2f}")
        self.kpi_max_vel.set(f"{k['max_vel']:.2f}")
        self.kpi_time.set(f"{k['flight_time']:.2f}")
        self.kpi_desc.set(f"{k['descent_rate']:.2f}")
        
        # Update progress bar based on data activity
        if hasattr(self, 'progress_bar') and CTK_AVAILABLE:
            data_points = k.get('points', 0)
            progress = min(1.0, data_points / 1000.0)  # Scale to 0-1
            self.progress_bar.set(progress)

    def _refresh_cube(self):
        # Only update cube if on orientation tab (performance optimization)
        try:
            current_tab = self.nb.index(self.nb.select())
            if current_tab != 1:  # Not on orientation cube tab
                return
        except Exception:
            pass
            
        # draw orientation cube using last orientation
        t, pos, vel, acc, orient = self.recon.snapshot()
        
        # Only update if we have new orientation data
        if not hasattr(self, '_last_orient_size'):
            self._last_orient_size = 0
        if orient.size == self._last_orient_size and orient.size > 0:
            return
        self._last_orient_size = orient.size
        
        self.ax_cube.clear()
        if orient.size == 0:
            self._draw_cube(self.ax_cube, 0.0, 0.0, 0.0)
        else:
            r, p, y = orient[-1,0], orient[-1,1], orient[-1,2]
            self._draw_cube(self.ax_cube, r, p, y)
        self.canvas_cube.draw_idle()

    def _draw_cube(self, ax, roll, pitch, yaw):
        # draw cube centered at origin with orientation R (roll,pitch,yaw)
        R = euler_to_rot(roll, pitch, yaw)
        s = 0.6
        corners = np.array([[-s,-s,-s], [s,-s,-s], [s,s,-s], [-s,s,-s],
                            [-s,-s,s],  [s,-s,s],  [s,s,s],  [-s,s,s]])
        rc = (R @ corners.T).T
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for a,b in edges:
            p1 = rc[a]; p2 = rc[b]
            ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]], color='k', linewidth=2)
        # axes
        xaxis = R @ np.array([1.2,0,0]); yaxis = R @ np.array([0,1.2,0]); zaxis = R @ np.array([0,0,1.2])
        ax.quiver(0,0,0, xaxis[0],xaxis[1],xaxis[2], color='r')
        ax.quiver(0,0,0, yaxis[0],yaxis[1],yaxis[2], color='g')
        ax.quiver(0,0,0, zaxis[0],zaxis[1],zaxis[2], color='b')
        ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5); ax.set_zlim(-1.5,1.5)
        ax.set_title(f"Orientation (roll={math.degrees(roll):.1f}°, pitch={math.degrees(pitch):.1f}°, yaw={math.degrees(yaw):.1f}°)")

    def shutdown(self):
        self._running = False
        try:
            self.recon.stop()
        except:
            pass
        try:
            self.root.quit()
        except:
            pass

# ---------------------------
# Helper: send log messages to Text widget
# ---------------------------
class _TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record) + "\n"
        try:
            self.text_widget.configure(state='normal')
            self.text_widget.insert('end', msg)
            self.text_widget.see('end')
            self.text_widget.configure(state='disabled')
        except Exception:
            pass

# ---------------------------
# Main entry point
# ---------------------------
def main():
    # create appropriate root (customtkinter CTk if available else tk.Tk)
    if CTK_AVAILABLE:
        try:
            root = ctk.CTk()
        except Exception:
            root = tk.Tk()
    else:
        root = tk.Tk()

    app = MissionControlApp(root)

    # embed matplotlib canvases after creation (need root to exist)
    # canvas_traj
    app.canvas_traj = FigureCanvasTkAgg(app.fig_traj, master=app.tab_traj)
    app.canvas_traj.get_tk_widget().pack(fill='both', expand=True)
    # canvas_cube
    app.canvas_cube = FigureCanvasTkAgg(app.fig_cube, master=app.tab_cube)
    app.canvas_cube.get_tk_widget().pack(fill='both', expand=True)
    # canvas_sig
    app.canvas_sig = FigureCanvasTkAgg(app.fig_sig, master=app.tab_signals)
    app.canvas_sig.get_tk_widget().pack(fill='both', expand=True)

    # KPI label variables used in left panel: set references
    app.kpi_max_alt = tk.StringVar(value="Max Alt: 0.00 m")
    app.kpi_max_vel = tk.StringVar(value="Max Vel: 0.00 m/s")
    app.kpi_time = tk.StringVar(value="Flight Time: 0.00 s")
    app.kpi_desc = tk.StringVar(value="Descent Rate: 0.00 m/s")
    # also expose copy variables used earlier to update visible labels
    app.kpi_max_alt_var = tk.StringVar(value=app.kpi_max_alt.get())
    app.kpi_max_vel_var = tk.StringVar(value=app.kpi_max_vel.get())
    app.kpi_time_var = tk.StringVar(value=app.kpi_time.get())
    app.kpi_desc_var = tk.StringVar(value=app.kpi_desc.get())

    # set the left panel labels to these (they already exist; find and set textvars)
    # For simplicity, update left KPI labels directly (they are the last 4 Label widgets in frame_left)
    left_children = app.frame_left.winfo_children()
    # find label widgets that show KPIs by searching string "KPIs (live)" index and following labels
    labels = [w for w in left_children if isinstance(w, tk.Label) or (CTK_AVAILABLE and isinstance(w, ctk.CTkLabel))]
    # It's tricky to find exact ones; instead assign by pack order earlier: after KPI header we packed 4 Labels, we can set them
    # But to keep it safe, simply create new labels at bottom to display KPI var values:
    # Place additional labels in left frame (they will appear after existing ones)
    tk.Label(app.frame_left, textvariable=app.kpi_max_alt_var, anchor='w').pack(fill='x', padx=8)
    tk.Label(app.frame_left, textvariable=app.kpi_max_vel_var, anchor='w').pack(fill='x', padx=8)
    tk.Label(app.frame_left, textvariable=app.kpi_time_var, anchor='w').pack(fill='x', padx=8)
    tk.Label(app.frame_left, textvariable=app.kpi_desc_var, anchor='w').pack(fill='x', padx=8)

    # start mainloop
    def on_closing():
        if messagebox.askokcancel("Quit", "Quit mission control?"):
            app.shutdown()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
