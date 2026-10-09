#!/usr/bin/env python3
"""
CANSAT Flight Telemetry Dashboard

A comprehensive desktop application for analyzing and visualizing CANSAT flight data.
Provides real-time telemetry analysis, flight phase detection, and interactive visualizations.

Features:
- Automatic flight phase detection (launch, apogee, deployment, landing)
- Multi-plot visualization dashboard with 4 synchronized views
- Robust CSV parsing with multiple format support
- Data quality assessment and reporting
- Export capabilities (plots and text reports)
- Interactive click-to-expand plot functionality

Author: Team 13
Requires: Python 3.8+, PyQt6, matplotlib, pandas, numpy
"""
import sys
import os
from datetime import datetime

import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.figure import Figure # type: ignore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas # type: ignore
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar # type: ignore

from PyQt6.QtWidgets import ( # type: ignore
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QTextEdit,
    QSplitter, QGroupBox, QGridLayout, QScrollArea, QSizePolicy, QSpacerItem
)
from PyQt6.QtCore import Qt # type: ignore
from PyQt6.QtGui import QFont # type: ignore

# --- Flight Detection Constants ---
ALTITUDE_LAUNCH_THRESHOLD = 10  # meters - minimum altitude gain to detect launch
ALTITUDE_LANDING_THRESHOLD = 50  # meters - altitude below which landing is considered
LANDING_VELOCITY_THRESHOLD = 1  # m/s - max descent rate to confirm landing
MAX_EXPECTED_ALTITUDE = 20000  # meters - sanity check for altitude readings
MIN_BATTERY_VOLTAGE = 6  # volts - threshold for low battery warning
POST_APOGEE_WINDOW = 100  # data points to analyze for parachute deployment
DEPLOYMENT_SEARCH_WINDOW = 50  # data points to search for deployment spike

# --- Export Settings ---
EXPORT_DPI = 200  # dots per inch for exported plots
DEFAULT_CSV_FILE = "Flight_TEAM_001.csv"  # default data file to load

# --- Theme ---
THEME = {
    "window_bg": "#0B1B2B",    # dark navy
    "panel_bg": "#0E2A3A",     # deep teal panel
    "accent": "#4DD0E1",       # bright teal accent
    "muted": "#89A7B3",        # muted secondary text
    "text": "#E6F0F2",         # light text
    "card": "#0F3A4A",         # card/panel background
    "grid": "#123844",         # grid lines
    "plot_bg": "#071823",      # plot background
}
FIG_FC = THEME["plot_bg"]
AXES_FC = THEME["panel_bg"]
TEXT_COLOR = THEME["text"]
ACCENT = THEME["accent"]
GRID_COLOR = THEME["grid"]

# Professional button style used on key buttons
BUTTON_STYLE = f"""
QPushButton {{
  background-color: {ACCENT};
  color: {THEME['panel_bg']};
  border-radius: 6px;
  padding: 6px 10px;
  font-weight: 600;
}}
QPushButton:hover {{
  background-color: #3ac5cc;
}}
QPushButton:pressed {{
  background-color: #2aaeb3;
}}
"""

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize CSV column names to internal format and coerce to numeric types.
    
    Handles various naming conventions from different CANSAT systems:
    - PythonTimestamp, HardwareTimestamp, timestamp -> time_seconds
    - gpsAlt, GPS_Alt, altitude, alt -> altitude
    - temp, temperature -> temperature
    - volt, battery, battery_voltage -> battery_voltage
    - latitude, lat -> latitude
    - longitude, lon, long -> longitude
    - acceleration, accel, a_mag -> acceleration
    
    Args:
        df: Raw DataFrame from CSV file
        
    Returns:
        DataFrame with standardized column names and numeric types
    """
    df = df.copy()
    orig_cols = list(df.columns)
    lowered = {orig: orig.lower().strip() for orig in orig_cols}

    def find_variant(*tokens):
        for orig, low in lowered.items():
            for t in tokens:
                if t in low:
                    return orig
        return None

    mapping = {}
    # prefer PythonTimestamp, then HardwareTimestamp, then any time-like header
    time_col = find_variant("pythontimestamp", "python_timestamp", "hardwaretimestamp", "hardware_timestamp", "time", "timestamp", "t_s")
    if time_col:
        mapping[time_col] = "time_seconds"

    alt_col = find_variant("gpsalt", "gps_alt", "altitude", "alt", "height")
    if alt_col:
        mapping[alt_col] = "altitude"

    temp_col = find_variant("temp", "temperature")
    if temp_col:
        mapping[temp_col] = "temperature"

    volt_col = find_variant("volt", "battery")
    if volt_col:
        mapping[volt_col] = "battery_voltage"

    lat_col = find_variant("latitude", "lat")
    lon_col = find_variant("longitude", "lon", "long")
    if lat_col:
        mapping[lat_col] = "latitude"
    if lon_col:
        mapping[lon_col] = "longitude"

    accel_col = find_variant("acceleration", "accel", "a_mag")
    if accel_col:
        mapping[accel_col] = "acceleration"

    # rename columns
    if mapping:
        df = df.rename(columns=mapping)

    # coerce numeric for known columns
    for c in ["time_seconds", "altitude", "temperature", "battery_voltage", "latitude", "longitude", "acceleration"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ensure time_seconds exists
    if "time_seconds" not in df.columns:
        # try HardwareTimestamp / PythonTimestamp by substring
        for possible in orig_cols:
            low = possible.lower()
            if "time" in low or "timestamp" in low or "hardware" in low:
                try:
                    df = df.reset_index(drop=True)
                    df["time_seconds"] = pd.to_numeric(df[possible], errors="coerce")
                    break
                except Exception:
                    pass
        if "time_seconds" not in df.columns:
            df = df.reset_index(drop=True)
            df["time_seconds"] = np.arange(len(df))

    return df

def enable_click_to_expand(fig):
    def on_click(event):
        if event.inaxes is None:
            return
        src_ax = event.inaxes
        
        # Create new figure with professional styling
        fig_new = plt.figure(figsize=(14, 9), facecolor=FIG_FC)
        # Remove duplicate title - individual plots have their own titles
        # fig_new.suptitle("CANSAT Flight Data - Detailed View", 
                        # fontsize=16, fontweight='bold', color=TEXT_COLOR, y=0.95)
        ax_new = fig_new.add_subplot(1, 1, 1, facecolor=AXES_FC)
        
        # Copy ALL elements with enhanced professional styling
        
        # 1. Copy all line plots with exact properties
        for line in src_ax.get_lines():
            try:
                x = line.get_xdata()
                y = line.get_ydata()
                ax_new.plot(x, y, 
                           color=line.get_color(),
                           linestyle=line.get_linestyle(),
                           linewidth=line.get_linewidth() * 1.2,  # Slightly thicker for popup
                           alpha=line.get_alpha(),
                           marker=line.get_marker(),
                           markersize=line.get_markersize(),
                           markeredgecolor=line.get_markeredgecolor(),
                           markerfacecolor=line.get_markerfacecolor(),
                           label=line.get_label() if line.get_label() and not line.get_label().startswith("_") else None)
            except Exception:
                pass
        
        # 2. Copy all scatter plots with exact properties including colormaps
        for coll in src_ax.collections:
            try:
                offsets = coll.get_offsets()
                if len(offsets) == 0:
                    continue
                    
                xs = offsets[:, 0]
                ys = offsets[:, 1]
                
                # Get exact sizes (slightly larger for popup)
                sizes = coll.get_sizes()
                s = sizes * 1.3 if len(sizes) > 1 else (sizes[0] * 1.3 if len(sizes) == 1 else 48)
                
                # Get color information
                try:
                    arr = coll.get_array()
                    if arr is not None and len(arr) > 0:
                        # Has color mapping - preserve it exactly
                        cmap = getattr(coll, 'cmap', 'viridis')
                        norm = getattr(coll, 'norm', None)
                        sc = ax_new.scatter(xs, ys, c=arr, cmap=cmap, norm=norm, s=s,
                                          edgecolors=coll.get_edgecolors() if hasattr(coll, 'get_edgecolors') else None, # type: ignore
                                          alpha=coll.get_alpha(), linewidths=1.5)
                        
                        # Add colorbar with professional styling
                        try:
                            cbar = fig_new.colorbar(sc, ax=ax_new, shrink=0.8, pad=0.02)
                            cbar.set_label("Time (seconds)", color=TEXT_COLOR, fontsize=12, fontweight='bold')
                            cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=10)
                            cbar.outline.set_edgecolor(ACCENT) # type: ignore
                            cbar.outline.set_linewidth(1.2) # type: ignore
                        except Exception:
                            pass
                    else:
                        # Single color scatter
                        facecolors = coll.get_facecolors()
                        if len(facecolors) > 0:
                            color = facecolors[0] if len(facecolors) == 1 else facecolors
                        else:
                            color = ACCENT
                        
                        ax_new.scatter(xs, ys, c=color, s=s,
                                     edgecolors=coll.get_edgecolors() if hasattr(coll, 'get_edgecolors') else None, # type: ignore
                                     alpha=coll.get_alpha(), linewidths=1.5)
                except Exception:
                    # Fallback
                    ax_new.scatter(xs, ys, s=s, color=ACCENT, alpha=coll.get_alpha(), linewidths=1.5)
                    
            except Exception:
                pass
        # 3. Copy all text annotations with enhanced styling and correct positioning
        for txt in src_ax.texts:
            try:
                x, y = txt.get_position()
                # Keep exact same font size as grouped view
                fontsize = txt.get_fontsize()
                
                # Get the text transform and handle coordinate systems properly
                try:
                    transform = txt.get_transform()
                    
                    # Handle different coordinate systems
                    if transform == src_ax.transAxes:
                        # Text uses axes coordinates (0-1), use same in new axes
                        transform = ax_new.transAxes
                    elif transform == src_ax.transData:
                        # Text uses data coordinates, keep it that way
                        transform = ax_new.transData
                    else:
                        # For any other transform, convert to data coordinates
                        try:
                            pts = transform.transform([(x, y)])
                            inv = src_ax.transData.inverted()
                            x, y = inv.transform(pts)[0]
                            transform = ax_new.transData
                        except:
                            transform = ax_new.transData
                except:
                    # Fallback to data coordinates if transform handling fails
                    transform = ax_new.transData
                
                # Detect if this is a phase label or stats box (excluding Launch)
                text = txt.get_text()
                is_phase_label = any(phase in text for phase in ['Apogee', 'Deploy', 'Landing'])
                is_stats_box = 'Max Altitude' in text or 'Flight Duration' in text

                # Custom positioning for expanded view to avoid overlap
                ha, va = txt.get_ha(), txt.get_va()
                if is_phase_label:
                    # Get the main data line for trend analysis
                    main_line = next((line for line in src_ax.get_lines() if len(line.get_ydata()) > 100), None)
                    if main_line is not None:
                        ha, va = _get_best_text_position(x, y, main_line.get_xdata(), main_line.get_ydata(), src_ax.get_xlim(), src_ax.get_ylim()) # type: ignore
                    
                    # Add a small offset to move the label away from the point
                    y_range = src_ax.get_ylim()[1] - src_ax.get_ylim()[0]
                    offset = y_range * 0.05  # 5% of the y-axis range
                    if va == 'bottom':
                        y += offset
                    else:
                        y -= offset
                
                # Apply appropriate styling based on label type
                ax_new.text(x, y, text,
                           fontsize=fontsize * 1.1,  # Slightly smaller font for better fit
                           color=txt.get_color(),
                           ha=ha,
                           va=va,
                           transform=transform,
                           fontweight='bold' if (is_phase_label or is_stats_box) else 'normal',
                           bbox=dict(
                               boxstyle='round,pad=0.4',
                               facecolor=THEME['card'] if is_stats_box else AXES_FC,
                               alpha=0.9,
                               edgecolor=ACCENT,
                               linewidth=1
                           ) if (is_phase_label or is_stats_box) else None)
            except Exception:
                pass
        
        # 3.5. Copy annotations (arrows and labels) with recalculated positioning for expanded view
        try:
            # Get plot dimensions for both source and new axes
            src_xlim = src_ax.get_xlim()
            src_ylim = src_ax.get_ylim()
            new_xlim = ax_new.get_xlim()
            new_ylim = ax_new.get_ylim()
            
            src_width = src_xlim[1] - src_xlim[0]
            src_height = src_ylim[1] - src_ylim[0]
            new_width = new_xlim[1] - new_xlim[0]
            new_height = new_ylim[1] - new_ylim[0]
            
            # Get the source axes children to find annotations
            for child in src_ax.get_children():
                if hasattr(child, 'xytext') and hasattr(child, 'xy'):
                    # This is an annotation
                    try:
                        # Get annotation properties
                        text = child.get_text()
                        xy = child.xy
                        xytext = child.xytext
                        
                        # Check if this is a phase label (excluding Launch)
                        is_phase_annotation = any(phase in text for phase in ['Apogee', 'Deploy', 'Landing'])
                        
                        if is_phase_annotation:
                            # Recalculate label position for expanded view
                            # Keep the same relative position but scale to new plot dimensions
                            
                            # Calculate relative offset from the point in the source plot
                            rel_x_offset = (xytext[0] - xy[0]) / src_width
                            rel_y_offset = (xytext[1] - xy[1]) / src_height
                            
                            # Apply the same relative offset in the new plot
                            new_xytext = (
                                xy[0] + rel_x_offset * new_width,
                                xy[1] + rel_y_offset * new_height
                            )
                            
                            # Determine phase type for enhanced positioning (excluding Launch)
                            phase_type = None
                            for phase in ['Apogee', 'Deploy', 'Landing']:
                                if phase in text:
                                    phase_type = phase.lower()
                                    break
                            
                            # Apply enhanced positioning for expanded view
                            if phase_type == 'deploy' or phase_type == 'landing':
                                # For deployment and landing, ensure they're positioned properly
                                # relative to the expanded plot dimensions
                                base_offset = new_height * 0.08  # Reduced to 8% of new plot height
                                
                                if phase_type == 'deploy':
                                    # Deployment: close positioning for clear connection
                                    new_xytext = (
                                        xy[0] + new_width * 0.06,  # 6% to the right (minimal)
                                        xy[1] - base_offset * 1.2   # Small downward offset
                                    )
                                elif phase_type == 'landing':
                                    # Landing: close positioning for clear connection
                                    new_xytext = (
                                        xy[0] + new_width * 0.05,   # 5% to the right (minimal)
                                        xy[1] - base_offset * 1.0   # Small downward offset
                                    )
                            
                            # Ensure label stays within plot bounds
                            margin_x = new_width * 0.02
                            margin_y = new_height * 0.02
                            new_xytext = (
                                max(new_xlim[0] + margin_x, min(new_xlim[1] - margin_x, new_xytext[0])),
                                max(new_ylim[0] + margin_y, min(new_ylim[1] - margin_y, new_xytext[1]))
                            )
                            
                            # Copy the annotation with enhanced styling for expanded view
                            ax_new.annotate(
                                text,
                                xy=xy,
                                xytext=new_xytext,
                                fontsize=child.get_fontsize() * 1.3,  # Larger font for expanded view
                                color=child.get_color(),
                                fontweight='bold',
                                ha=child.get_ha(),
                                va=child.get_va(),
                                bbox=dict(
                                    boxstyle='round,pad=0.6',
                                    facecolor=THEME['card'],
                                    alpha=0.45,  # Further reduced opacity for maximum trajectory visibility in expanded view
                                    edgecolor=ACCENT,
                                    linewidth=2.5
                                ),
                                arrowprops=dict(
                                    arrowstyle='-|>',
                                    color=ACCENT,
                                    lw=2.5,
                                    connectionstyle='arc3,rad=0.1',
                                    shrinkA=4,
                                    shrinkB=6,
                                    alpha=0.50  # Further reduced arrow opacity for maximum trajectory visibility in expanded view
                                ),
                                zorder=15  # Higher z-order for expanded view
                            )
                    except Exception:
                        pass
        except Exception:
            pass
        
        # 4. Copy titles and labels with enhanced professional styling
        try:
            # Enhanced title with larger font
            title = src_ax.get_title()
            ax_new.set_title(title, color=TEXT_COLOR, fontweight="bold", fontsize=16, pad=20)
            
            # Enhanced axis labels with larger fonts
            xlabel = src_ax.get_xlabel()
            ylabel = src_ax.get_ylabel()
            ax_new.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=14, fontweight='bold')
            ax_new.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=14, fontweight='bold')
        except Exception:
            pass
        
        # 5. Copy axis limits and scales exactly
        try:
            ax_new.set_xlim(src_ax.get_xlim())
            ax_new.set_ylim(src_ax.get_ylim())
            ax_new.set_xscale(src_ax.get_xscale())
            ax_new.set_yscale(src_ax.get_yscale())
        except Exception:
            pass
        
        # 6. Enhanced professional grid and styling
        try:
            ax_new.grid(True, alpha=0.4, color=GRID_COLOR, linestyle='-', linewidth=0.6)
            ax_new.set_axisbelow(True)  # Put grid behind data
        except Exception:
            ax_new.grid(True, alpha=0.4, color=GRID_COLOR)
        
        # Professional tick styling
        ax_new.tick_params(colors=TEXT_COLOR, labelsize=12, which='both', direction='in', 
                          top=True, right=True, length=5, width=1.2)
        ax_new.minorticks_on()  # Add minor ticks for professional look
        
        # 7. Enhanced spine styling
        for spine in ax_new.spines.values():
            spine.set_color(ACCENT)
            spine.set_linewidth(1.5)
        
        # 8. Copy legend with enhanced styling
        try:
            src_legend = src_ax.get_legend()
            if src_legend:
                handles, labels = src_ax.get_legend_handles_labels()
                if handles and labels:
                    legend = ax_new.legend(handles, labels, facecolor=AXES_FC, edgecolor=ACCENT,
                                         fontsize=11, framealpha=0.95, fancybox=True, shadow=True)
                    legend.get_frame().set_linewidth(1.2)
        except Exception:
            pass
        
        # 9. Add professional footer with metadata
        footer_text = f"CANSAT Flight Analysis Dashboard | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        fig_new.text(0.5, 0.02, footer_text, ha='center', va='bottom', 
                    fontsize=9, color=THEME['muted'], style='italic')
        
        # 10. Professional layout with proper margins
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        # Set window title
        manager = fig_new.canvas.manager # type: ignore
        if hasattr(manager, 'set_window_title'):
            manager.set_window_title('CANSAT Flight Data - Detailed Analysis')
        
        plt.show()
    
    # Connect the click event
    if hasattr(fig, "canvas") and fig.canvas is not None:
        try:
            fig.canvas.mpl_connect("button_press_event", on_click)
        except Exception:
            pass

def create_sample_data():
    time_points = np.arange(0, 300, 1)
    ascent_time = 60
    max_altitude = 1000
    altitude = []
    for t in time_points:
        if t < ascent_time:
            alt = max_altitude * (1 - np.exp(-3 * t / ascent_time))
        elif t < 180:
            alt = max_altitude * np.exp(-0.01 * (t - ascent_time))
        else:
            alt = max_altitude * np.exp(-0.01 * 120) * np.exp(-0.003 * (t - 180))
        altitude.append(max(alt, 0))
    altitude = np.array(altitude) + np.random.normal(0, 5, len(time_points))
    
    # Original sensor data
    temp = 25 - 0.0065 * altitude + np.random.normal(0, 1, len(time_points))
    voltage = 9 - 0.001 * time_points + np.random.normal(0, 0.05, len(time_points))
    lat = 37.0 + np.cumsum(np.random.normal(0, 0.0001, len(time_points)))
    lon = -122.0 + np.cumsum(np.random.normal(0, 0.0001, len(time_points)))
    
    # Magnetometer data simulation
    # Simulate Earth's magnetic field with rotation during flight
    phase = 2 * np.pi * time_points / 300  # Full rotation during flight
    mx = 30 * np.cos(phase) + np.random.normal(0, 2, len(time_points))
    my = 30 * np.sin(phase) + np.random.normal(0, 2, len(time_points))
    mz = 40 + np.random.normal(0, 1, len(time_points))  # Vertical component
    
    # Gyroscope data simulation
    # Simulate rotation rates during different flight phases
    gx = np.zeros(len(time_points))
    gy = np.zeros(len(time_points))
    gz = np.zeros(len(time_points))
    
    # Launch phase rotation
    gx[:ascent_time] = np.random.normal(0.5, 0.2, ascent_time)
    gy[:ascent_time] = np.random.normal(0.3, 0.2, ascent_time)
    gz[:ascent_time] = np.random.normal(2.0, 0.5, ascent_time)
    
    # Descent phase rotation
    gx[ascent_time:] = np.random.normal(-0.2, 0.3, len(time_points) - ascent_time)
    gy[ascent_time:] = np.random.normal(0.1, 0.3, len(time_points) - ascent_time)
    gz[ascent_time:] = np.random.normal(-1.0, 0.4, len(time_points) - ascent_time)
    
    # Add some spinning events
    spin_start = 80
    spin_duration = 20
    gz[spin_start:spin_start+spin_duration] += 5 * np.sin(np.linspace(0, 4*np.pi, spin_duration))
    
    # Accelerometer data simulation
    # Base acceleration components
    g = 9.81  # Earth's gravity
    ax = np.zeros(len(time_points))
    ay = np.zeros(len(time_points))
    az = -g * np.ones(len(time_points))  # Initially pointing down
    
    # Launch phase acceleration
    launch_accel = 30  # m/s^2 during launch
    az[:ascent_time] += -launch_accel * np.exp(-time_points[:ascent_time]/20)
    
    # Add vibration and noise
    ax += np.random.normal(0, 0.5, len(time_points))
    ay += np.random.normal(0, 0.5, len(time_points))
    az += np.random.normal(0, 0.8, len(time_points))
    
    # Add launch vibration
    vib_freq = 10  # Hz
    vib_t = np.linspace(0, ascent_time/10, ascent_time)
    ax[:ascent_time] += 2 * np.sin(2*np.pi*vib_freq*vib_t) * np.exp(-vib_t/2)
    ay[:ascent_time] += 2 * np.sin(2*np.pi*vib_freq*vib_t) * np.exp(-vib_t/2)
    
    return pd.DataFrame({
        "time_seconds": time_points,
        "altitude": altitude,
        "temperature": temp,
        "battery_voltage": voltage,
        "latitude": lat,
        "longitude": lon,
        # Magnetometer data
        "mx": mx,
        "my": my,
        "mz": mz,
        # Gyroscope data
        "gx": gx,
        "gy": gy,
        "gz": gz,
        # Accelerometer data
        "ax": ax,
        "ay": ay,
        "az": az
    })

class DataLoader:
    @staticmethod
    def load_csv(filepath):
        """Robust CSV loader: normalize headers, coerce numeric, fix GPS/alt issues."""
        try:
            raw = pd.read_csv(filepath)
            # normalize header whitespace and remove BOM if present
            raw.columns = [str(c).strip().replace("\ufeff", "") for c in raw.columns]

            # quick common renames (case-insensitive)
            col_lc = {c.lower(): c for c in raw.columns}
            if "pythontimestamp" in col_lc and "time_seconds" not in raw.columns:
                raw = raw.rename(columns={col_lc["pythontimestamp"]: "PythonTimestamp"})
            if "hardwaretimestamp" in col_lc and "time_seconds" not in raw.columns:
                raw = raw.rename(columns={col_lc["hardwaretimestamp"]: "HardwareTimestamp"})
            if "gpsalt" in col_lc and "gpsAlt" not in raw.columns:
                raw = raw.rename(columns={col_lc["gpsalt"]: "gpsAlt"})

            # use the standardizer which handles many name variants
            df = _standardize_columns(raw)

            # coerce all columns that look numeric to numeric (leave others)
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce") if df[c].dtype == object else df[c]

            # If altitude is present but all zeros / NaN, prefer gpsAlt from original file
            if "altitude" in df.columns:
                alt_vals = df["altitude"].dropna()
                alt_all_zero = alt_vals.size == 0 or (alt_vals.abs().sum() == 0)
                if alt_all_zero and "gpsAlt" in raw.columns:
                    df["altitude"] = pd.to_numeric(raw["gpsAlt"], errors="coerce")

            # Preserve GPS 0.0 values: just coerce to numeric
            for c in ("latitude", "longitude"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # Ensure time_seconds exists (standardizer should have created it)
            if "time_seconds" not in df.columns:
                # try common timestamp columns from raw
                for cand in ("PythonTimestamp", "HardwareTimestamp", "timestamp", "time"):
                    if cand in raw.columns:
                        df["time_seconds"] = pd.to_numeric(raw[cand], errors="coerce")
                        break
            if "time_seconds" not in df.columns:
                df = df.reset_index(drop=True)
                df["time_seconds"] = np.arange(len(df))

            # interpolate altitude if sparse
            if "altitude" in df.columns and df["altitude"].isnull().any():
                df["altitude"] = df["altitude"].interpolate(method="linear", limit_direction="both")

            # final check: require time_seconds
            if "time_seconds" not in df.columns:
                return None, "Could not determine time column."

            return df, None
        except Exception as e:
            return None, str(e)

    @staticmethod
    def auto_load_default():
        default_file = "Flight_TEAM_001.csv"
        if os.path.exists(default_file):
            return DataLoader.load_csv(default_file)
        return None, "Default file not found"

class CANSATAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy() if data is not None else pd.DataFrame()
        self.flight_phases = {}
        self.stats = {}
        self.data_quality = {}

    def detect_flight_phases(self):
        """
        Detect key flight phases from altitude and sensor data.
        
        Phases detected:
        - Launch: First significant altitude gain (>ALTITUDE_LAUNCH_THRESHOLD)
        - Apogee: Maximum altitude point
        - Deployment: Parachute deployment (detected from acceleration spike)
        - Landing: Altitude below threshold with minimal descent rate
        
        Results stored in self.flight_phases with keys: launch, apogee, deployment, landing
        Each phase contains: time (seconds), altitude (meters), index (row number)
        """
        if "altitude" in self.data.columns and not self.data["altitude"].isnull().all():
            alt = self.data["altitude"].to_numpy()
            time = self.data["time_seconds"].to_numpy()
            try:
                alt_rate = np.gradient(alt, time)
            except Exception:
                alt_rate = np.gradient(alt)
            apogee_idx = int(np.argmax(alt))
            launch_idx = 0
            threshold = 10
            for i in range(len(alt)):
                if alt[i] > alt[0] + ALTITUDE_LAUNCH_THRESHOLD:
                    launch_idx = i
                    break
            deploy_idx = apogee_idx
            if apogee_idx < len(alt) - 20:
                post = alt_rate[apogee_idx:apogee_idx + POST_APOGEE_WINDOW]
                if len(post) > 10:
                    rd = np.abs(np.diff(post))
                    if len(rd) > 0:
                        off = np.argmax(rd[:DEPLOYMENT_SEARCH_WINDOW]) if len(rd) >= DEPLOYMENT_SEARCH_WINDOW else np.argmax(rd)
                        deploy_idx = min(len(alt) - 1, apogee_idx + off + 10)
            landing_idx = len(time) - 1
            for i in range(apogee_idx, len(alt)):
                if alt[i] < ALTITUDE_LANDING_THRESHOLD and abs(alt_rate[i]) < LANDING_VELOCITY_THRESHOLD:
                    landing_idx = i
                    break
        elif all(c in self.data.columns for c in ("ax", "ay", "az")):
            time = self.data["time_seconds"].to_numpy()
            a = np.sqrt(self.data["ax"].to_numpy() ** 2 + self.data["ay"].to_numpy() ** 2 + self.data["az"].to_numpy() ** 2)
            apogee_idx = int(np.argmax(a))
            launch_idx = 0
            deploy_idx = apogee_idx
            landing_idx = len(time) - 1
        else:
            n = len(self.data)
            apogee_idx = n // 2
            launch_idx = 0
            deploy_idx = apogee_idx
            landing_idx = n - 1
            time = self.data["time_seconds"].to_numpy() if "time_seconds" in self.data.columns else np.arange(n)

        self.flight_phases = {
            "launch": {"time": float(time[launch_idx]), "altitude": float(self.data["altitude"].iloc[launch_idx]) if "altitude" in self.data.columns else 0.0, "index": int(launch_idx)},
            "apogee": {"time": float(time[apogee_idx]), "altitude": float(self.data["altitude"].iloc[apogee_idx]) if "altitude" in self.data.columns else 0.0, "index": int(apogee_idx)},
            "deployment": {"time": float(time[deploy_idx]), "altitude": float(self.data["altitude"].iloc[deploy_idx]) if "altitude" in self.data.columns else 0.0, "index": int(deploy_idx)},
            "landing": {"time": float(time[landing_idx]), "altitude": float(self.data["altitude"].iloc[landing_idx]) if "altitude" in self.data.columns else 0.0, "index": int(landing_idx)},
        }

    def calculate_key_parameters(self):
        """
        Calculate flight statistics and key parameters.
        
        Calculates:
        - Max altitude, total flight time
        - Ascent/descent times and rates
        - Freefall vs parachute descent rates
        - Temperature statistics (min/max/avg)
        - Battery voltage (initial/final/min)
        - GPS ground distance traveled
        
        Results stored in self.stats dictionary.
        """
        fp = self.flight_phases
        data = self.data
        total_time = fp["landing"]["time"] - fp["launch"]["time"]
        ascent_time = fp["apogee"]["time"] - fp["launch"]["time"]
        descent_time = fp["landing"]["time"] - fp["apogee"]["time"]
        freefall_time = fp["deployment"]["time"] - fp["apogee"]["time"]
        freefall_dist = fp["apogee"]["altitude"] - fp["deployment"]["altitude"]
        freefall_rate = freefall_dist / freefall_time if freefall_time > 0 else 0
        parachute_time = fp["landing"]["time"] - fp["deployment"]["time"]
        parachute_dist = fp["deployment"]["altitude"] - fp["landing"]["altitude"]
        parachute_rate = parachute_dist / parachute_time if parachute_time > 0 else 0
        avg_ascent_rate = fp["apogee"]["altitude"] / ascent_time if ascent_time > 0 else 0
        temp_data = data.get("temperature", pd.Series()).dropna() if "temperature" in data else pd.Series()
        temp_stats = {"min": float(temp_data.min()) if len(temp_data) > 0 else 0,
                      "max": float(temp_data.max()) if len(temp_data) > 0 else 0,
                      "avg": float(temp_data.mean()) if len(temp_data) > 0 else 0}
        volt = data.get("battery_voltage", pd.Series()).dropna() if "battery_voltage" in data else pd.Series()
        volt_stats = {"initial": float(volt.iloc[0]) if len(volt) > 0 else 0,
                      "final": float(volt.iloc[-1]) if len(volt) > 0 else 0,
                      "min": float(volt.min()) if len(volt) > 0 else 0}
        total_distance = 0
        if "latitude" in data.columns and "longitude" in data.columns:
            lat = data["latitude"].dropna()
            lon = data["longitude"].dropna()
            if len(lat) > 1 and len(lon) > 1:
                lat_diff = (lat.iloc[-1] - lat.iloc[0]) * 111000
                lon_diff = (lon.iloc[-1] - lon.iloc[0]) * 111000 * np.cos(np.radians(lat.mean()))
                total_distance = float(np.sqrt(lat_diff ** 2 + lon_diff ** 2))
        self.stats = {
            "max_altitude": float(fp["apogee"]["altitude"]),
            "total_time": float(total_time),
            "ascent_time": float(ascent_time),
            "descent_time": float(descent_time),
            "freefall_rate": float(freefall_rate),
            "parachute_rate": float(parachute_rate),
            "avg_ascent_rate": float(avg_ascent_rate),
            "temperature": temp_stats,
            "battery": volt_stats,
            "gps_distance": total_distance,
        }

    def assess_data_quality(self):
        """
        Assess data quality and identify potential issues.
        
        Checks for:
        - Missing data (null percentages by column)
        - Negative altitude values
        - Unusually high altitudes
        - Low battery voltage
        - GPS data issues (all zeros)
        
        Results stored in self.data_quality with total_records and issues list.
        """
        issues = []
        n = len(self.data) if self.data is not None else 0
        if n == 0:
            self.data_quality = {"total_records": 0, "issues": ["Empty dataset"]}
            return
        for col in self.data.columns:
            try:
                null_pct = (self.data[col].isnull().sum() / n) * 100
            except Exception:
                null_pct = 0.0
            if null_pct > 0:
                issues.append(f"{col}: {null_pct:.1f}% missing")
        if "altitude" in self.data.columns:
            alt = self.data["altitude"].dropna()
            if len(alt) > 0:
                if (alt < 0).any():
                    issues.append("Negative altitude values")
                if alt.max() > MAX_EXPECTED_ALTITUDE:
                    issues.append(f"Unusually high altitude (>{MAX_EXPECTED_ALTITUDE}m)")
        if "battery_voltage" in self.data.columns:
            v = self.data["battery_voltage"].dropna()
            if len(v) > 0 and v.min() < MIN_BATTERY_VOLTAGE:
                issues.append(f"Low battery voltage (<{MIN_BATTERY_VOLTAGE}V)")
        if "latitude" in self.data.columns and "longitude" in self.data.columns:
            lat = self.data["latitude"].dropna()
            lon = self.data["longitude"].dropna()
            if len(lat) and len(lon) and (lat.abs().sum() == 0 or lon.abs().sum() == 0):
                issues.append("GPS present but values appear all-zero")
        self.data_quality = {"total_records": int(n), "issues": issues if issues else ["No major issues detected"]}

class FlightDashboard(QMainWindow):
    def __init__(self, analyzer: CANSATAnalyzer = None, filepath: str = None): # type: ignore
        super().__init__()
        self.analyzer = analyzer
        self.current_filepath = filepath
        self.setWindowTitle("CANSAT Flight Telemetry Dashboard")
        self.setMinimumSize(1400, 900)
        self._setup_ui()
        if self.analyzer:
            self._update_all_displays()

    def _setup_ui(self):
        central = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        central.setLayout(layout)
        # main background + global text color
        central.setStyleSheet(f"background-color: {THEME['window_bg']}; color: {TEXT_COLOR};")
        self.setCentralWidget(central)

        header = QWidget()
        hl = QHBoxLayout()
        hl.setContentsMargins(0, 0, 0, 0)
        header.setLayout(hl)
        title = QLabel("CANSAT FLIGHT TELEMETRY")
        f = QFont()
        f.setPointSize(16)
        f.setBold(True)
        title.setFont(f)
        title.setStyleSheet(f"color: {TEXT_COLOR}; font-family: 'Segoe UI', Arial, sans-serif;")
        hl.addWidget(title)
        
        # Add subtitle
        subtitle = QLabel("Real-time Flight Data Analysis Dashboard")
        subtitle_font = QFont()
        subtitle_font.setPointSize(10)
        subtitle_font.setItalic(True)
        subtitle.setFont(subtitle_font)
        subtitle.setStyleSheet(f"color: {THEME['muted']}; margin-left: 5px;")
        hl.addWidget(subtitle)
        hl.addItem(QSpacerItem(20, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        self.file_label = QLabel(self.current_filepath or "No file loaded")
        self.file_label.setStyleSheet(f"color:{THEME['muted']};")
        hl.addWidget(self.file_label)
        layout.addWidget(header)

        controls = QWidget()
        cl = QHBoxLayout()
        cl.setContentsMargins(0, 0, 0, 0)
        controls.setLayout(cl)
        load_btn = QPushButton("Load CSV")
        load_btn.clicked.connect(self._load_file)
        reload_btn = QPushButton("Reload Default")
        reload_btn.clicked.connect(self._reload_default)
        # apply professional button style
        load_btn.setStyleSheet(BUTTON_STYLE)
        reload_btn.setStyleSheet(BUTTON_STYLE)
        cl.addWidget(load_btn)
        cl.addWidget(reload_btn)
        cl.addItem(QSpacerItem(20, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        export_plots = QPushButton("Export Plots")
        export_plots.clicked.connect(self._export_plots)
        export_report = QPushButton("Export Report")
        export_report.clicked.connect(self._export_report)
        export_plots.setStyleSheet(BUTTON_STYLE)
        export_report.setStyleSheet(BUTTON_STYLE)
        cl.addWidget(export_plots)
        cl.addWidget(export_report)
        layout.addWidget(controls)

        main_split = QSplitter(Qt.Orientation.Vertical)
        self.stats_panel = self._create_stats_panel()
        self.stats_panel.setMinimumHeight(100)
        self.stats_panel.setMaximumHeight(150)
        main_split.addWidget(self.stats_panel)

        # Create a scroll area for graphs with enhanced scrolling
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        
        # Style the scroll bars
        scroll_style = f"""
            QScrollBar:vertical {{
                background: {THEME['panel_bg']};
                width: 12px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {THEME['muted']};
                min-height: 20px;
                border-radius: 6px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {ACCENT};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: {THEME['panel_bg']};
                border-radius: 6px;
            }}
        """
        scroll_area.setStyleSheet(scroll_style)
        
        # Container for all graphs with increased spacing
        graphs_container = QWidget()
        gl = QVBoxLayout()
        gl.setContentsMargins(10, 10, 10, 10)
        gl.setSpacing(15)  # Increase spacing between elements
        graphs_container.setLayout(gl)
        
        # Add toolbar
        dummy_fig = Figure(facecolor=FIG_FC)
        dummy_canvas = FigureCanvas(dummy_fig)
        toolbar = NavigationToolbar(dummy_canvas, self)
        toolbar.setMaximumHeight(36)
        gl.addWidget(toolbar)

        # Original plots section with increased size
        grid_split = QSplitter(Qt.Orientation.Horizontal)
        grid_split.setMinimumHeight(800)  # Increase minimum height for better visibility
        left_split = QSplitter(Qt.Orientation.Vertical)
        right_split = QSplitter(Qt.Orientation.Vertical)
        
        # Set size policy for better resizing behavior
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_split.setSizePolicy(size_policy)
        right_split.setSizePolicy(size_policy)

        # Create figures with increased size and better spacing
        self.fig1 = Figure(facecolor=FIG_FC, figsize=(8, 6))
        self.canvas1 = FigureCanvas(self.fig1)
        self.ax1 = self.fig1.add_subplot(111, facecolor=AXES_FC)
        self.canvas1.setMinimumHeight(400)  # Increased minimum height
        left_split.addWidget(self.canvas1)

        self.fig3 = Figure(facecolor=FIG_FC, figsize=(8, 6))
        self.canvas3 = FigureCanvas(self.fig3)
        self.ax3 = self.fig3.add_subplot(111, facecolor=AXES_FC)
        self.canvas3.setMinimumHeight(400)  # Increased minimum height
        left_split.addWidget(self.canvas3)

        self.fig2 = Figure(facecolor=FIG_FC, figsize=(8, 6))
        self.canvas2 = FigureCanvas(self.fig2)
        self.ax2 = self.fig2.add_subplot(111, facecolor=AXES_FC)
        self.canvas2.setMinimumHeight(400)  # Increased minimum height
        right_split.addWidget(self.canvas2)

        self.fig4 = Figure(facecolor=FIG_FC, figsize=(8, 6))
        self.canvas4 = FigureCanvas(self.fig4)
        self.ax4 = self.fig4.add_subplot(111, facecolor=AXES_FC)
        self.canvas4.setMinimumHeight(400)  # Increased minimum height
        right_split.addWidget(self.canvas4)

        left_split.setSizes([500, 400])
        right_split.setSizes([450, 450])
        grid_split.addWidget(left_split)
        grid_split.addWidget(right_split)
        gl.addWidget(grid_split)
        
        # Additional sensor plots section
        sensor_split = QSplitter(Qt.Orientation.Horizontal)
        
        # Magnetometer plot
        self.fig_mag = Figure(facecolor=FIG_FC)
        self.canvas_mag = FigureCanvas(self.fig_mag)
        self.ax_mag = self.fig_mag.add_subplot(111, facecolor=AXES_FC)
        
        # Gyroscope plot
        self.fig_gyro = Figure(facecolor=FIG_FC)
        self.canvas_gyro = FigureCanvas(self.fig_gyro)
        self.ax_gyro = self.fig_gyro.add_subplot(111, facecolor=AXES_FC)
        
        # Accelerometer plot
        self.fig_accel = Figure(facecolor=FIG_FC)
        self.canvas_accel = FigureCanvas(self.fig_accel)
        self.ax_accel = self.fig_accel.add_subplot(111, facecolor=AXES_FC)
        
        # Add the new plots to a horizontal splitter with increased size
        sensor_left = QSplitter(Qt.Orientation.Vertical)
        sensor_right = QSplitter(Qt.Orientation.Vertical)
        
        # Set minimum sizes for better visibility
        self.canvas_mag.setMinimumHeight(300)
        self.canvas_gyro.setMinimumHeight(300)
        self.canvas_accel.setMinimumHeight(300)
        
        # Add plots to splitters
        sensor_left.addWidget(self.canvas_mag)
        sensor_left.addWidget(self.canvas_gyro)
        sensor_right.addWidget(self.canvas_accel)
        
        # Set size policies for better resizing
        sensor_left.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        sensor_right.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        
        sensor_split.addWidget(sensor_left)
        sensor_split.addWidget(sensor_right)
        
        # Set the split sizes
        sensor_split.setSizes([500, 500])
        
        gl.addWidget(sensor_split)
        
        # Enable click-to-expand for new plots
        for fig in (self.fig_mag, self.fig_gyro, self.fig_accel):
            enable_click_to_expand(fig)
        
        # Set up scroll area
        scroll_area.setWidget(graphs_container)
        main_split.addWidget(scroll_area)
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        main_split.setSizes([150, 700])
        layout.addWidget(main_split)

        self.status = self.statusBar()
        self.status.showMessage("Ready")

        for fig in (self.fig1, self.fig2, self.fig3, self.fig4):
            enable_click_to_expand(fig)

    def _create_stats_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        v = QVBoxLayout()
        v.setContentsMargins(8, 8, 8, 8)
        container.setLayout(v)
        title = QLabel("FLIGHT STATISTICS")
        tf = QFont()
        tf.setPointSize(11)
        tf.setBold(True)
        title.setFont(tf)
        title.setStyleSheet(f"color:{TEXT_COLOR};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(title)

        grid = QGridLayout()
        self.stat_labels = {}
        stats_config = [
            ("Max Altitude", "max_alt", 0, 0),
            ("Total Time", "total_time", 0, 1),
            ("Ascent Time", "ascent_time", 0, 2),
            ("Descent Time", "descent_time", 0, 3),
            ("Avg Ascent Rate", "ascent_rate", 1, 0),
            ("Freefall Rate", "freefall_rate", 1, 1),
            ("Parachute Rate", "parachute_rate", 1, 2),
            ("GPS Distance", "gps_dist", 1, 3),
        ]
        for label_text, key, r, c in stats_config:
            g = QGroupBox(label_text)
            gl = QVBoxLayout()
            val = QLabel("--")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val.setStyleSheet(f"color:{TEXT_COLOR}; font-weight:bold;")
            gl.addWidget(val)
            g.setLayout(gl)
            # card style with subtle border and background
            g.setStyleSheet(
                "QGroupBox {"
                f" color: {TEXT_COLOR};"
                f" background: {THEME['card']};"
                f" border: 1px solid rgba(77,208,225,0.18);"
                " border-radius:8px;"
                " padding:8px;"
                "}"
            )
            grid.addWidget(g, r, c)
            self.stat_labels[key] = val

        v.addLayout(grid)
        qg = QGroupBox("Data Quality")
        ql = QVBoxLayout()
        self.quality_text = QTextEdit()
        self.quality_text.setReadOnly(True)
        self.quality_text.setMaximumHeight(60)
        self.quality_text.setStyleSheet(f"color:{TEXT_COLOR}; background:{AXES_FC};")
        ql.addWidget(self.quality_text)
        qg.setLayout(ql)
        qg.setStyleSheet(
            "QGroupBox {"
            f" color: {TEXT_COLOR};"
            f" background: {THEME['card']};"
            " border: 1px solid rgba(77,208,225,0.14);"
            " border-radius:8px;"
            " padding:8px;"
            "}"
        )
        v.addWidget(qg)
        scroll.setWidget(container)
        return scroll

    def _load_file(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv);;All Files (*)")
        if fp:
            self._process_file(fp)

    def _reload_default(self):
        df, err = DataLoader.auto_load_default()
        if err:
            QMessageBox.warning(self, "Load Error", f"Could not load default: {err}")
            return
        self._process_dataframe(df, "Flight_TEAM_001.csv")

    def _process_file(self, filepath):
        df, err = DataLoader.load_csv(filepath)
        if err:
            # show warning but continue if df present
            if df is None:
                QMessageBox.critical(self, "Load Error", f"Failed to load: {err}")
                return
            else:
                QMessageBox.warning(self, "Load Warning", err)
        self._process_dataframe(df, filepath)

    def _process_dataframe(self, df, filepath):
        self.analyzer = CANSATAnalyzer(df)
        self.analyzer.detect_flight_phases()
        self.analyzer.calculate_key_parameters()
        self.analyzer.assess_data_quality()
        self.current_filepath = filepath
        self.file_label.setText(f"Loaded: {os.path.basename(filepath)}")
        self._update_all_displays()
        QMessageBox.information(self, "Success", "Data loaded and analyzed")

    def _export_plots(self):
        if not self.analyzer:
            QMessageBox.warning(self, "Export Error", "No data loaded")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if not folder:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            self.fig1.savefig(os.path.join(folder, f"altitude_{ts}.png"), dpi=200, facecolor=self.fig1.get_facecolor())
            self.fig2.savefig(os.path.join(folder, f"sensors_{ts}.png"), dpi=200, facecolor=self.fig2.get_facecolor())
            self.fig3.savefig(os.path.join(folder, f"gps_{ts}.png"), dpi=200, facecolor=self.fig3.get_facecolor())
            self.fig4.savefig(os.path.join(folder, f"descent_{ts}.png"), dpi=200, facecolor=self.fig4.get_facecolor())
            QMessageBox.information(self, "Export", "Plots exported")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_report(self):
        if not self.analyzer:
            QMessageBox.warning(self, "Export Error", "No data loaded")
            return
        fp, _ = QFileDialog.getSaveFileName(self, "Save report", f"flight_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "Text Files (*.txt);;All Files (*)")
        if not fp:
            return
        try:
            with open(fp, "w") as f:
                f.write("CANSAT FLIGHT TELEMETRY ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Flight Phases Section
                f.write("FLIGHT PHASES\n")
                f.write("-" * 20 + "\n")
                for k, v in self.analyzer.flight_phases.items():
                    f.write(f"{k.title()}: Time={v['time']:.1f}s, Altitude={v['altitude']:.1f}m\n")
                
                # Flight Statistics Section
                f.write("\nFLIGHT STATISTICS\n")
                f.write("-" * 20 + "\n")
                for k, v in self.analyzer.stats.items():
                    if isinstance(v, dict):
                        f.write(f"{k.replace('_', ' ').title()}:\n")
                        for sub_k, sub_v in v.items():
                            f.write(f"  {sub_k}: {sub_v:.2f}\n")
                    else:
                        f.write(f"{k.replace('_', ' ').title()}: {v:.2f}\n")
                
                # Sensor Data Analysis Section
                data = self.analyzer.data
                
                # Magnetometer Analysis
                if all(col in data.columns for col in ['mx', 'my', 'mz']):
                    f.write("\nMAGNETOMETER ANALYSIS\n")
                    f.write("-" * 25 + "\n")
                    mx_data = data['mx'].dropna()
                    my_data = data['my'].dropna()
                    mz_data = data['mz'].dropna()
                    
                    if len(mx_data) > 0:
                        f.write(f"X-axis: Min={mx_data.min():.2f}, Max={mx_data.max():.2f}, Avg={mx_data.mean():.2f}, Std={mx_data.std():.2f}\n")
                        f.write(f"Y-axis: Min={my_data.min():.2f}, Max={my_data.max():.2f}, Avg={my_data.mean():.2f}, Std={my_data.std():.2f}\n")
                        f.write(f"Z-axis: Min={mz_data.min():.2f}, Max={mz_data.max():.2f}, Avg={mz_data.mean():.2f}, Std={mz_data.std():.2f}\n")
                        
                        # Calculate magnetic field magnitude
                        mag_magnitude = (mx_data**2 + my_data**2 + mz_data**2)**0.5
                        f.write(f"Magnetic Field Magnitude: Min={mag_magnitude.min():.2f}, Max={mag_magnitude.max():.2f}, Avg={mag_magnitude.mean():.2f}\n")
                
                # Gyroscope Analysis
                if all(col in data.columns for col in ['gx', 'gy', 'gz']):
                    f.write("\nGYROSCOPE ANALYSIS\n")
                    f.write("-" * 20 + "\n")
                    gx_data = data['gx'].dropna()
                    gy_data = data['gy'].dropna()
                    gz_data = data['gz'].dropna()
                    
                    if len(gx_data) > 0:
                        f.write(f"X-axis Rotation: Min={gx_data.min():.2f}, Max={gx_data.max():.2f}, Avg={gx_data.mean():.2f}, Std={gx_data.std():.2f} rad/s\n")
                        f.write(f"Y-axis Rotation: Min={gy_data.min():.2f}, Max={gy_data.max():.2f}, Avg={gy_data.mean():.2f}, Std={gy_data.std():.2f} rad/s\n")
                        f.write(f"Z-axis Rotation: Min={gz_data.min():.2f}, Max={gz_data.max():.2f}, Avg={gz_data.mean():.2f}, Std={gz_data.std():.2f} rad/s\n")
                        
                        # Calculate angular velocity magnitude
                        angular_velocity = (gx_data**2 + gy_data**2 + gz_data**2)**0.5
                        f.write(f"Angular Velocity Magnitude: Min={angular_velocity.min():.2f}, Max={angular_velocity.max():.2f}, Avg={angular_velocity.mean():.2f} rad/s\n")
                        
                        # Detect high rotation events
                        high_rotation_threshold = angular_velocity.mean() + 2 * angular_velocity.std()
                        high_rotation_events = len(angular_velocity[angular_velocity > high_rotation_threshold])
                        f.write(f"High Rotation Events (>2σ): {high_rotation_events} data points\n")
                
                # Accelerometer Analysis
                if all(col in data.columns for col in ['ax', 'ay', 'az']):
                    f.write("\nACCELEROMETER ANALYSIS\n")
                    f.write("-" * 23 + "\n")
                    ax_data = data['ax'].dropna()
                    ay_data = data['ay'].dropna()
                    az_data = data['az'].dropna()
                    
                    if len(ax_data) > 0:
                        f.write(f"X-axis Acceleration: Min={ax_data.min():.2f}, Max={ax_data.max():.2f}, Avg={ax_data.mean():.2f}, Std={ax_data.std():.2f} m/s²\n")
                        f.write(f"Y-axis Acceleration: Min={ay_data.min():.2f}, Max={ay_data.max():.2f}, Avg={ay_data.mean():.2f}, Std={ay_data.std():.2f} m/s²\n")
                        f.write(f"Z-axis Acceleration: Min={az_data.min():.2f}, Max={az_data.max():.2f}, Avg={az_data.mean():.2f}, Std={az_data.std():.2f} m/s²\n")
                        
                        # Calculate total acceleration magnitude
                        total_accel = (ax_data**2 + ay_data**2 + az_data**2)**0.5
                        f.write(f"Total Acceleration: Min={total_accel.min():.2f}, Max={total_accel.max():.2f}, Avg={total_accel.mean():.2f} m/s²\n")
                        
                        # Detect launch acceleration
                        launch_threshold = 15.0  # m/s² above gravity
                        launch_events = len(total_accel[total_accel > launch_threshold])
                        f.write(f"High Acceleration Events (>{launch_threshold} m/s²): {launch_events} data points\n")
                        
                        # Detect free fall or low-g events
                        freefall_threshold = 5.0  # m/s² below normal gravity
                        freefall_events = len(total_accel[total_accel < freefall_threshold])
                        f.write(f"Low Acceleration Events (<{freefall_threshold} m/s²): {freefall_events} data points\n")
                
                # Data Quality Section
                f.write("\nDATA QUALITY ASSESSMENT\n")
                f.write("-" * 25 + "\n")
                f.write(f"Total Records: {self.analyzer.data_quality.get('total_records', 0)}\n")
                f.write("Issues Detected:\n")
                for issue in self.analyzer.data_quality.get("issues", []):
                    f.write(f"- {issue}\n")
                
                # Data Coverage Summary
                f.write("\nDATA COVERAGE SUMMARY\n")
                f.write("-" * 23 + "\n")
                sensor_columns = {
                    'Basic Sensors': ['time_seconds', 'altitude', 'temperature', 'battery_voltage'],
                    'GPS': ['latitude', 'longitude'],
                    'Magnetometer': ['mx', 'my', 'mz'],
                    'Gyroscope': ['gx', 'gy', 'gz'],
                    'Accelerometer': ['ax', 'ay', 'az']
                }
                
                for category, columns in sensor_columns.items():
                    available = sum(1 for col in columns if col in data.columns)
                    total = len(columns)
                    f.write(f"{category}: {available}/{total} sensors available\n")
                
            QMessageBox.information(self, "Export", "Comprehensive report saved")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _plot_altitude_on_ax(self, ax):
        data = self.analyzer.data
        phases = self.analyzer.flight_phases
        if "time_seconds" in data.columns and "altitude" in data.columns:
            # Main trajectory line - thicker and more prominent
            ax.plot(data["time_seconds"], data["altitude"], color=ACCENT, linewidth=2.5)
            
            # Phase markers with enhanced visibility
            for phase, pdata in phases.items():
                ax.axvline(pdata["time"], color="#FFD166", linestyle="--", linewidth=1.6)
                ax.scatter(pdata["time"], pdata["altitude"], s=100, edgecolors=ACCENT, 
                         facecolor="#FFFFFF", linewidth=2)
            
            # Larger, more prominent titles and labels
            ax.set_title("Altitude Profile - Flight Trajectory", color=TEXT_COLOR, 
                        fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel("Time (seconds)", color=TEXT_COLOR, fontsize=14, fontweight='bold', labelpad=10)
            ax.set_ylabel("Altitude Above Ground Level (m)", color=TEXT_COLOR, 
                        fontsize=14, fontweight='bold', labelpad=10)
            
            # Enhanced phase annotations with improved positioning and collision avoidance
            # Get plot dimensions for adaptive positioning
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            plot_width = x_max - x_min
            plot_height = y_max - y_min
            
            # Store label positions to avoid overlaps
            used_positions = []
            
            # Define phase order for consistent positioning (excluding launch)
            phase_order = ["apogee", "deployment", "landing"]
            
            for phase in phase_order:
                if phase not in phases:
                    continue
                    
                pdata = phases[phase]
                phase_names = {
                    "launch": "Launch",
                    "apogee": "Apogee", 
                    "deployment": "Deploy",
                    "landing": "Landing"
                }
                
                # Get main trajectory data for intelligent positioning
                time_data = data["time_seconds"].to_numpy()
                alt_data = data["altitude"].to_numpy()
                
                # Analyze trajectory curve around this phase point for optimal positioning
                def find_clear_label_position(phase_time, phase_alt, time_data, alt_data, plot_width, plot_height):
                    """Find the best position for a label that doesn't obscure the trajectory line."""
                    
                    # Find the index of the phase point in the data
                    phase_idx = np.searchsorted(time_data, phase_time)
                    
                    # Define analysis window around the phase point
                    window_size = min(20, len(time_data) // 10)  # Adaptive window size
                    start_idx = max(0, phase_idx - window_size)
                    end_idx = min(len(time_data), phase_idx + window_size)
                    
                    # Get local trajectory segment
                    local_time = time_data[start_idx:end_idx]
                    local_alt = alt_data[start_idx:end_idx]
                    
                    if len(local_time) < 3:
                        # Not enough data for analysis, use default positioning
                        return plot_height * 0.15, 0, 'center', 'bottom', 'arc3,rad=0'
                    
                    # Calculate local slope and curvature
                    try:
                        # First derivative (slope)
                        dt = np.gradient(local_time)
                        dalt = np.gradient(local_alt)
                        slope = dalt / dt
                        
                        # Second derivative (curvature)
                        d2alt = np.gradient(slope)
                        curvature = d2alt / dt[:-1] if len(dt) > 1 else np.array([0])
                        
                        # Get slope and curvature at phase point
                        phase_local_idx = phase_idx - start_idx
                        if phase_local_idx >= len(slope):
                            phase_local_idx = len(slope) - 1
                        
                        local_slope = slope[phase_local_idx] if len(slope) > phase_local_idx else 0
                        local_curv = curvature[phase_local_idx] if len(curvature) > phase_local_idx else 0
                        
                    except (IndexError, ValueError):
                        local_slope = 0
                        local_curv = 0
                    
                    # Determine optimal positioning based on trajectory characteristics
                    base_offset = plot_height * 0.08  # Reduced base offset for closer positioning
                    
                    if phase == "apogee":
                        # Apogee: always place above since it's the peak
                        y_offset = base_offset
                        x_offset = 0
                        align_h = 'center'
                        align_v = 'bottom'
                        connection_style = 'arc3,rad=0'
                        
                    elif phase == "launch":
                        # Launch: analyze initial trajectory
                        if abs(local_slope) > 2:  # Steep ascent
                            # Place to the left to avoid steep line
                            y_offset = base_offset * 0.5
                            x_offset = -plot_width * 0.12
                            align_h = 'right'
                            align_v = 'center'
                            connection_style = 'arc3,rad=0.3'
                        else:  # Gradual ascent
                            # Place above
                            y_offset = base_offset
                            x_offset = -plot_width * 0.05
                            align_h = 'center'
                            align_v = 'bottom'
                            connection_style = 'arc3,rad=0.1'
                            
                    elif phase == "deployment":
                        # Deployment: position close to the actual deployment point for clarity
                        # Use minimal offsets to keep the label close and clearly connected
                        
                        # Always use minimal, consistent positioning regardless of proximity to apogee
                        # Position slightly to the right and below to avoid trajectory overlap
                        y_offset = -base_offset * 1.2  # Small downward offset
                        x_offset = plot_width * 0.06   # Small rightward offset (6% of plot width)
                        align_h = 'left'
                        align_v = 'top'
                        connection_style = 'arc3,rad=-0.1'
                            
                    else:  # landing
                        # Landing: position close to the actual landing point for clarity
                        # Use minimal offsets to keep the label close and clearly connected
                        
                        # Always use minimal, consistent positioning
                        # Position slightly below and to the right to avoid trajectory overlap
                        y_offset = -base_offset * 1.0  # Small downward offset
                        x_offset = plot_width * 0.05   # Small rightward offset (5% of plot width)
                        align_h = 'left'
                        align_v = 'top'
                        connection_style = 'arc3,rad=-0.1'
                    
                    return y_offset, x_offset, align_h, align_v, connection_style
                
                # Get optimal positioning for this phase
                y_offset, x_offset, align_h, align_v, connection_style = find_clear_label_position(
                    pdata["time"], pdata["altitude"], time_data, alt_data, plot_width, plot_height
                )
                
                # Calculate initial label position
                label_x = pdata["time"] + x_offset
                label_y = pdata["altitude"] + y_offset
                
                # Enhanced collision avoidance with trajectory line awareness
                min_distance = plot_height * 0.10  # Increased minimum distance between labels
                collision_detected = False
                
                for used_pos in used_positions:
                    distance = np.sqrt((label_x - used_pos[0])**2 + (label_y - used_pos[1])**2)
                    if distance < min_distance:
                        collision_detected = True
                        break
                
                if collision_detected:
                    # Try alternative positioning to avoid both collisions and trajectory line
                    if phase == "apogee":
                        # Try positioning further above or to the side
                        y_offset = plot_height * 0.20
                        x_offset = plot_width * 0.05  # Slight side offset
                    elif phase == "deployment":
                        # Try positioning slightly further but still close to deployment point
                        x_offset = plot_width * 0.08   # Minimal horizontal offset
                        y_offset = -plot_height * 0.08  # Minimal downward offset
                    elif phase == "landing":
                        # Try positioning slightly further but still close to landing point
                        y_offset = -plot_height * 0.08  # Minimal downward offset
                        x_offset = plot_width * 0.07    # Minimal horizontal offset
                    else:  # launch
                        # Try positioning further to the left or above
                        x_offset = -plot_width * 0.15
                        y_offset = plot_height * 0.12
                    
                    label_x = pdata["time"] + x_offset
                    label_y = pdata["altitude"] + y_offset
                
                # Boundary checks to keep labels within plot area
                label_x = max(x_min + plot_width * 0.02, min(x_max - plot_width * 0.02, label_x))
                label_y = max(y_min + plot_height * 0.02, min(y_max - plot_height * 0.02, label_y))
                
                # Store this position
                used_positions.append((label_x, label_y))
                
                # Format values with appropriate precision
                alt_str = f"{pdata['altitude']:.0f}m" if pdata['altitude'] >= 100 else f"{pdata['altitude']:.1f}m"
                time_str = f"{pdata['time']:.0f}s"
                
                # Create informative label text
                phase_text = phase_names.get(phase, phase.title())
                label_text = f"{phase_text}\n{alt_str} @ {time_str}"
                
                # Enhanced arrow properties with better visibility
                arrow_props = dict(
                    arrowstyle='-|>',
                    color=ACCENT,
                    lw=2.2,  # Thicker line for better visibility
                    connectionstyle=connection_style,
                    shrinkA=1,  # Minimal shrink from text box
                    shrinkB=1,  # Minimal shrink from point for clearer connection
                    alpha=0.9   # Higher opacity for better visibility in grouped view
                )
                
                # Add the label with enhanced visibility and style
                ax.annotate(
                    label_text,
                    xy=(pdata["time"], pdata["altitude"]),  # Point to annotate
                    xytext=(label_x, label_y),  # Label position
                    fontsize=9,  # Slightly larger for better readability
                    color=TEXT_COLOR,
                    fontweight='bold',
                    ha=align_h,
                    va=align_v,
                    bbox=dict(
                        boxstyle='round,pad=0.4',
                        facecolor=THEME['card'],
                        alpha=0.85,  # Increased opacity for better readability in grouped view
                        edgecolor=ACCENT,
                        linewidth=1.5
                    ),
                    arrowprops=arrow_props,
                    zorder=10  # Ensure labels appear on top
                )
 
            
            # Enhanced statistics box with improved visibility
            stats_text = f"Flight Statistics:\nMax Altitude: {max(data['altitude']):.1f}m\nFlight Duration: {max(data['time_seconds']):.0f}s"
            ax.text(0.98, 0.02, stats_text,
                   transform=ax.transAxes,  # Use axes coordinates (0-1) instead of data coordinates
                   fontsize=12, color=TEXT_COLOR,
                   ha='right', va='bottom',  # Align to bottom-right corner
                   bbox=dict(boxstyle='round,pad=0.6',
                           facecolor=THEME['card'],
                           alpha=0.80,  # Reduced opacity for consistency
                           edgecolor=ACCENT,
                           linewidth=1.8))
            # Enhanced tick parameters for better readability
            ax.tick_params(colors=TEXT_COLOR, labelsize=12, width=1.5, length=6, 
                         direction='out', top=True, right=True)
            
            # More visible but not overwhelming grid
            ax.grid(True, alpha=0.3, color=GRID_COLOR, linestyle='-', linewidth=0.8)
            ax.set_axisbelow(True)  # Keep grid behind data
            
            # Enhanced plot borders
            for s in ax.spines.values():
                s.set_color(ACCENT)
                s.set_linewidth(1.8)  # Thicker border
                
            # Add minor grid for more precise reading
            ax.grid(True, which='minor', alpha=0.15, color=GRID_COLOR, linestyle='-', linewidth=0.4)
            ax.minorticks_on()
        else:
            ax.text(0.5, 0.5, "No altitude data", ha="center", va="center", color=TEXT_COLOR)

    def _plot_sensors_on_ax(self, ax):
        data = self.analyzer.data
        ax.clear()
        if "time_seconds" in data.columns:
            t = data["time_seconds"]
            if "temperature" in data.columns:
                ax.plot(t, data["temperature"], color="#FF7A7A", label="Temp")
            # Remove twinx() for click-to-expand compatibility
            if "battery_voltage" in data.columns:
                # Scale battery voltage to be visible alongside temperature
                batt_scaled = data["battery_voltage"] * 5  # Scale up for visibility
                ax.plot(t, batt_scaled, color=ACCENT, label="Battery (V × 5)")
            ax.set_title("Environmental Sensors - Temperature & Battery", color=TEXT_COLOR, fontsize=14, fontweight='bold')
            ax.set_xlabel("Time (seconds)", color=TEXT_COLOR, fontsize=12)
            ax.set_ylabel("Temperature (°C) / Battery (V × 5)", color=TEXT_COLOR, fontsize=12)
            # Updated ylabel to reflect both sensors
            ax.tick_params(colors=TEXT_COLOR)
            for s in ax.spines.values():
                s.set_color(ACCENT)
                s.set_linewidth(1.2)
            # Simplified legend for single axis
            ax.legend(facecolor=AXES_FC, edgecolor=ACCENT)
            # Add data quality indicators
            temp_points = len(data["temperature"].dropna()) if "temperature" in data.columns else 0
            batt_points = len(data["battery_voltage"].dropna()) if "battery_voltage" in data.columns else 0
            quality_text = f"Data Points:\nTemp: {temp_points}\nBattery: {batt_points}"
            ax.text(0.98, 0.02, quality_text, transform=ax.transAxes, 
                   fontsize=9, color=TEXT_COLOR, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=THEME['card'], alpha=0.9, edgecolor=ACCENT))
            ax.grid(True, alpha=0.4, color=GRID_COLOR, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)  # Put grid behind data
        else:
            ax.text(0.5, 0.5, "No sensor timestamps", ha="center", va="center", color=TEXT_COLOR)

    def _plot_gps_on_ax(self, ax):
        data = self.analyzer.data
        ax.clear()
        if "latitude" in data.columns and "longitude" in data.columns:
            # Get GPS data (preserve zeros, only drop actual NaN)
            lat = data["latitude"]
            lon = data["longitude"]
            
            # Remove only NaN values, keep 0.0 values
            valid_mask = ~(lat.isna() | lon.isna())
            lat_clean = lat[valid_mask]
            lon_clean = lon[valid_mask]
            
            if len(lat_clean) > 0 and len(lon_clean) > 0:
                # Get corresponding time values
                times = data["time_seconds"][valid_mask].to_numpy()
                
                # Check if all GPS values are zeros (no GPS fix)
                all_lat_zero = (lat_clean == 0.0).all()
                all_lon_zero = (lon_clean == 0.0).all()
                
                if all_lat_zero and all_lon_zero:
                    # Special handling for all-zero GPS data
                    ax.scatter([0], [0], c='red', s=100, marker='x', 
                             label=f'No GPS Fix ({len(lat_clean)} points at origin)')
                    ax.set_xlim(-0.1, 0.1)
                    ax.set_ylim(-0.1, 0.1)
                    ax.text(0.02, 0.02, f'{len(lat_clean)} GPS readings\nat (0,0)', 
                           fontsize=10, color=TEXT_COLOR, 
                           bbox=dict(boxstyle='round', facecolor=AXES_FC, alpha=0.8))
                else:
                    # Normal GPS plotting with valid coordinates
                    sc = ax.scatter(lon_clean, lat_clean, c=times, cmap="viridis", 
                                  s=24, edgecolors=ACCENT, alpha=0.8)
                    ax.plot(lon_clean, lat_clean, color=ACCENT, linewidth=1, alpha=0.6)
                    
                    # Add colorbar
                    try:
                        cbar = self.fig3.colorbar(sc, ax=ax)
                        cbar.set_label("Time (s)", color=TEXT_COLOR)
                        if hasattr(cbar, "ax"):
                            cbar.ax.tick_params(colors=TEXT_COLOR)
                    except Exception:
                        pass
            else:
                # No valid GPS data at all
                ax.text(0.5, 0.5, "No GPS data available", 
                       ha="center", va="center", color=TEXT_COLOR, 
                       transform=ax.transAxes)
        else:
            # No GPS columns found
            ax.text(0.5, 0.5, "No GPS columns found", 
                   ha="center", va="center", color=TEXT_COLOR, 
                   transform=ax.transAxes)
        
        ax.set_title("GPS Tracking - Flight Path", color=TEXT_COLOR, fontsize=14, fontweight='bold')
        ax.set_xlabel("Longitude (degrees)", color=TEXT_COLOR, fontsize=12)
        ax.set_ylabel("Latitude (degrees)", color=TEXT_COLOR, fontsize=12)
        ax.tick_params(colors=TEXT_COLOR)
        ax.grid(True, alpha=0.4, color=GRID_COLOR, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)  # Put grid behind data
        for s in ax.spines.values():
            s.set_color(ACCENT)
        
        # Add legend if there are labeled elements
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, facecolor=AXES_FC, edgecolor=ACCENT)

    def _plot_descent_on_ax(self, ax):
        data = self.analyzer.data
        fp = self.analyzer.flight_phases
        ax.clear()
        if "altitude" in data.columns:
            aidx = fp["apogee"]["index"]
            didx = fp["deployment"]["index"]
            lidx = fp["landing"]["index"]
            ax.plot(data["time_seconds"].iloc[aidx:didx+1], data["altitude"].iloc[aidx:didx+1], color="#FF6B00", linewidth=2, label="Freefall Phase")
            ax.plot(data["time_seconds"].iloc[didx:lidx+1], data["altitude"].iloc[didx:lidx+1], color="#4CAF50", linewidth=2, label="Parachute Phase")
            ax.set_title("Descent Analysis - Freefall vs Parachute", color=TEXT_COLOR, fontsize=14, fontweight='bold')
            ax.set_xlabel("Time (seconds)", color=TEXT_COLOR, fontsize=12)
            ax.set_ylabel("Altitude Above Ground Level (m)", color=TEXT_COLOR, fontsize=12)
            
            # Add descent phase labels
            ax.text(0.02, 0.98, "Freefall Phase", transform=ax.transAxes, 
                   color="#FF6B00", fontsize=10, fontweight='bold', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=AXES_FC, alpha=0.8))
            ax.text(0.02, 0.88, "Parachute Phase", transform=ax.transAxes, 
                   color="#4CAF50", fontsize=10, fontweight='bold', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=AXES_FC, alpha=0.8))
            ax.legend(facecolor=AXES_FC, edgecolor=ACCENT, fontsize=10, loc='upper right')
            # Add descent statistics
            freefall_rate = self.analyzer.stats.get('freefall_rate', 0)
            parachute_rate = self.analyzer.stats.get('parachute_rate', 0)
            stats_text = f"Freefall Rate: {freefall_rate:.1f} m/s\nParachute Rate: {parachute_rate:.1f} m/s"
            ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                   fontsize=9, color=TEXT_COLOR, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=THEME['card'], alpha=0.9, edgecolor=ACCENT))
            ax.tick_params(colors=TEXT_COLOR)
            for s in ax.spines.values():
                s.set_color(ACCENT)
                s.set_linewidth(1.2)
            ax.grid(True, alpha=0.4, color=GRID_COLOR, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)  # Put grid behind data
        else:
            ax.text(0.5, 0.5, "No altitude data", ha="center", va="center", color=TEXT_COLOR)

    def _plot_magnetometer_on_ax(self, ax):
        """Plot magnetometer data with professional styling."""
        data = self.analyzer.data
        ax.clear()
        if all(col in data.columns for col in ["time_seconds", "mx", "my", "mz"]):
            t = data["time_seconds"]
            
            # Plot each axis
            ax.plot(t, data["mx"], color="#FF7A7A", label="X-axis", linewidth=1.5)
            ax.plot(t, data["my"], color="#4CAF50", label="Y-axis", linewidth=1.5)
            ax.plot(t, data["mz"], color=ACCENT, label="Z-axis", linewidth=1.5)
            
            ax.set_title("Magnetometer Readings", color=TEXT_COLOR, fontsize=14, fontweight='bold')
            ax.set_xlabel("Time (seconds)", color=TEXT_COLOR, fontsize=12)
            ax.set_ylabel("Magnetic Field Strength", color=TEXT_COLOR, fontsize=12)
            
            # Add legend with professional styling
            ax.legend(facecolor=AXES_FC, edgecolor=ACCENT, fontsize=10)
            
            # Add grid and styling
            ax.grid(True, alpha=0.4, color=GRID_COLOR, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(colors=TEXT_COLOR)
            
            for spine in ax.spines.values():
                spine.set_color(ACCENT)
                spine.set_linewidth(1.2)
                
            # Add data statistics
            stats_text = f"Peak Values:\nX: {data['mx'].max():.1f}\nY: {data['my'].max():.1f}\nZ: {data['mz'].max():.1f}"
            ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                   fontsize=9, color=TEXT_COLOR, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=THEME['card'], alpha=0.9, edgecolor=ACCENT))
        else:
            ax.text(0.5, 0.5, "No magnetometer data available", 
                   ha="center", va="center", color=TEXT_COLOR,
                   transform=ax.transAxes)

    def _plot_gyroscope_on_ax(self, ax):
        """Plot gyroscope data with professional styling."""
        data = self.analyzer.data
        ax.clear()
        if all(col in data.columns for col in ["time_seconds", "gx", "gy", "gz"]):
            t = data["time_seconds"]
            
            # Plot each axis
            ax.plot(t, data["gx"], color="#FF7A7A", label="Roll (X)", linewidth=1.5)
            ax.plot(t, data["gy"], color="#4CAF50", label="Pitch (Y)", linewidth=1.5)
            ax.plot(t, data["gz"], color=ACCENT, label="Yaw (Z)", linewidth=1.5)
            
            ax.set_title("Gyroscope Angular Rates", color=TEXT_COLOR, fontsize=14, fontweight='bold')
            ax.set_xlabel("Time (seconds)", color=TEXT_COLOR, fontsize=12)
            ax.set_ylabel("Angular Rate (rad/s)", color=TEXT_COLOR, fontsize=12)
            
            # Add legend with professional styling
            ax.legend(facecolor=AXES_FC, edgecolor=ACCENT, fontsize=10)
            
            # Add grid and styling
            ax.grid(True, alpha=0.4, color=GRID_COLOR, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(colors=TEXT_COLOR)
            
            for spine in ax.spines.values():
                spine.set_color(ACCENT)
                spine.set_linewidth(1.2)
                
            # Add data statistics
            max_rates = [data[ax].abs().max() for ax in ['gx', 'gy', 'gz']]
            stats_text = f"Peak Rates (rad/s):\nRoll: {max_rates[0]:.2f}\nPitch: {max_rates[1]:.2f}\nYaw: {max_rates[2]:.2f}"
            ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                   fontsize=9, color=TEXT_COLOR, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=THEME['card'], alpha=0.9, edgecolor=ACCENT))
        else:
            ax.text(0.5, 0.5, "No gyroscope data available", 
                   ha="center", va="center", color=TEXT_COLOR,
                   transform=ax.transAxes)

    def _plot_accelerometer_on_ax(self, ax):
        """Plot accelerometer data with professional styling."""
        data = self.analyzer.data
        ax.clear()
        if all(col in data.columns for col in ["time_seconds", "ax", "ay", "az"]):
            t = data["time_seconds"]
            
            # Plot each axis
            ax.plot(t, data["ax"], color="#FF7A7A", label="X-axis", linewidth=1.5)
            ax.plot(t, data["ay"], color="#4CAF50", label="Y-axis", linewidth=1.5)
            ax.plot(t, data["az"], color=ACCENT, label="Z-axis", linewidth=1.5)
            
            # Calculate total acceleration magnitude
            acc_mag = np.sqrt(data["ax"]**2 + data["ay"]**2 + data["az"]**2)
            ax.plot(t, acc_mag, color="#FFD700", label="Magnitude", linewidth=1.2, alpha=0.7)
            
            ax.set_title("Accelerometer Readings", color=TEXT_COLOR, fontsize=14, fontweight='bold')
            ax.set_xlabel("Time (seconds)", color=TEXT_COLOR, fontsize=12)
            ax.set_ylabel("Acceleration (m/s²)", color=TEXT_COLOR, fontsize=12)
            
            # Add legend with professional styling
            ax.legend(facecolor=AXES_FC, edgecolor=ACCENT, fontsize=10)
            
            # Add grid and styling
            ax.grid(True, alpha=0.4, color=GRID_COLOR, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(colors=TEXT_COLOR)
            
            for spine in ax.spines.values():
                spine.set_color(ACCENT)
                spine.set_linewidth(1.2)
                
            # Add data statistics
            max_acc = acc_mag.max()
            min_acc = acc_mag.min()
            stats_text = f"Acceleration Stats:\nPeak: {max_acc:.2f} m/s²\nMin: {min_acc:.2f} m/s²\nMean: {acc_mag.mean():.2f} m/s²"
            ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                   fontsize=9, color=TEXT_COLOR, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=THEME['card'], alpha=0.9, edgecolor=ACCENT))
        else:
            ax.text(0.5, 0.5, "No accelerometer data available", 
                   ha="center", va="center", color=TEXT_COLOR,
                   transform=ax.transAxes)

    def _update_all_displays(self):
        if not self.analyzer:
            return
        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.clear()
        self._plot_altitude_on_ax(self.ax1)
        self._plot_sensors_on_ax(self.ax2)
        self._plot_gps_on_ax(self.ax3)
        self._plot_descent_on_ax(self.ax4)
        self._plot_magnetometer_on_ax(self.ax_mag)
        self._plot_gyroscope_on_ax(self.ax_gyro)
        self._plot_accelerometer_on_ax(self.ax_accel)
        
        # Update all canvases
        self.canvas1.draw_idle()
        self.canvas2.draw_idle()
        self.canvas3.draw_idle()
        self.canvas4.draw_idle()
        self.canvas_mag.draw_idle()
        self.canvas_gyro.draw_idle()
        self.canvas_accel.draw_idle()
        self._update_statistics()
        src = os.path.basename(self.current_filepath) if self.current_filepath else "Sample Data"
        self.status.showMessage(f"Data loaded: {src}")

    def _update_statistics(self):
        if not self.analyzer or not self.analyzer.stats:
            return
        s = self.analyzer.stats
        self.stat_labels["max_alt"].setText(f"{s.get('max_altitude',0):.1f} m")
        self.stat_labels["total_time"].setText(f"{s.get('total_time',0):.1f} s")
        self.stat_labels["ascent_time"].setText(f"{s.get('ascent_time',0):.1f} s")
        self.stat_labels["descent_time"].setText(f"{s.get('descent_time',0):.1f} s")
        self.stat_labels["ascent_rate"].setText(f"{s.get('avg_ascent_rate',0):.2f} m/s")
        self.stat_labels["freefall_rate"].setText(f"{s.get('freefall_rate',0):.2f} m/s")
        self.stat_labels["parachute_rate"].setText(f"{s.get('parachute_rate',0):.2f} m/s")
        self.stat_labels["gps_dist"].setText(f"{s.get('gps_distance',0):.1f} m")
        q = self.analyzer.data_quality or {"total_records":0, "issues":["No assessment"]}
        self.quality_text.setText(f"Total Records: {q.get('total_records',0)}\nIssues: " + ", ".join(q.get("issues", [])))

def _get_best_text_position(x, y, x_data, y_data, x_lim, y_lim, preferred_ha='center', preferred_va='bottom'):
    """
    Determines the best position for a text label to avoid overlapping with data lines.
    Analyzes the local slope of the data to place the label in an open area.
    """
    try:
        # Find the index in the data closest to the label's x-coordinate
        idx = np.searchsorted(x_data, x)
        
        # Define a small window around the point to analyze the local trend
        start = max(0, idx - 5)
        end = min(len(x_data) - 1, idx + 5)
        
        if start >= end:
            # Not enough data to determine trend, use preferred alignment
            return preferred_ha, preferred_va

        # Get local data segment
        local_x = x_data[start:end]
        local_y = y_data[start:end]
        
        # Calculate the slope of the line of best fit for the local data
        if len(local_x) > 1:
            slope = np.polyfit(local_x, local_y, 1)[0]
        else:
            slope = 0

        # Determine vertical alignment based on slope
        if abs(slope) < 0.5:  # Relatively flat line
            # Check if point is in upper or lower half of the plot
            y_range = y_lim[1] - y_lim[0]
            if (y - y_lim[0]) / y_range > 0.5:
                va = 'top'  # Point is high, place label below
            else:
                va = 'bottom'  # Point is low, place label above
        elif slope > 0:  # Upward trend
            va = 'top'  # Place label below the point
        else:  # Downward trend
            va = 'bottom'  # Place label above the point

        # Determine horizontal alignment
        x_range = x_lim[1] - x_lim[0]
        if (x - x_lim[0]) / x_range < 0.1:
            ha = 'left'  # Point is on the far left
        elif (x - x_lim[0]) / x_range > 0.9:
            ha = 'right'  # Point is on the far right
        else:
            ha = 'center'
            
        return ha, va
    except Exception:
        # Fallback to preferred alignment in case of any error
        return preferred_ha, preferred_va

def main():
    try:
        QApplication.setAttribute(Qt.ApplicationAttribute.EnableHighDpiScaling, True)
    except Exception:
        pass
    try:
        QApplication.setAttribute(Qt.ApplicationAttribute.UseHighDpiPixmaps, True)
    except Exception:
        pass

    app = QApplication(sys.argv)

    cli_filepath = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.isfile(arg):
            cli_filepath = arg

    if cli_filepath:
        df, error = DataLoader.load_csv(cli_filepath)
        if error and df is None:
            QMessageBox.critical(None, "Load Error", f"Failed to load file {cli_filepath}:\n{error}")
            return
        analyzer = CANSATAnalyzer(df) # type: ignore
        analyzer.detect_flight_phases()
        analyzer.calculate_key_parameters()
        analyzer.assess_data_quality()
        window = FlightDashboard(analyzer, cli_filepath)
    else:
        df, err = DataLoader.auto_load_default()
        if df is not None:
            analyzer = CANSATAnalyzer(df)
            analyzer.detect_flight_phases()
            analyzer.calculate_key_parameters()
            analyzer.assess_data_quality()
            window = FlightDashboard(analyzer, "Flight_TEAM_001.csv")
        else:
            df = create_sample_data()
            analyzer = CANSATAnalyzer(df)
            analyzer.detect_flight_phases()
            analyzer.calculate_key_parameters()
            analyzer.assess_data_quality()
            window = FlightDashboard(analyzer, "Sample Data")

    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
