import sys
import random
import numpy as np
import time
from PyQt6 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import csv
import threading
import queue
from datetime import datetime

class CanSatGUI(QtWidgets.QMainWindow):
    """
    Championship-Grade CanSat Ground Station GUI
    Optimized for hackathon judging criteria with NASA-grade polish
    """

    # Enhanced color scheme for maximum readability
    BG_COLOR = "#0a0e27"  # Deep space blue
    PANEL_BG = "#1a1f3a"
    CARD_BG = "#252b48"
    ACCENT_COLOR = "#00d9ff"  # Cyan accent
    SUCCESS_COLOR = "#00ff88"
    WARNING_COLOR = "#ffaa00"
    CRITICAL_COLOR = "#ff3366"
    TEXT_COLOR = "#e8e8e8"
    LABEL_COLOR = "#a0a8c0"
    
    MAX_HISTORY = 150
    UPDATE_INTERVAL_MS = 100
    ALERT_BLINK_INTERVAL = 500

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🛰️ CanSat Mission Control - Championship Edition")
        self.setStyleSheet(f"background-color: {self.BG_COLOR}; color: {self.TEXT_COLOR};")
        
        # Get screen geometry and maximize
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(50, 50, min(screen.width() - 100, 1600), min(screen.height() - 100, 1000))
        
        # State variables
        self.time = 0.0
        self.last_time = time.time()
        self.graph_data = {param: [] for param in self.params}
        self.angles = {"X": 0.0, "Y": 0.0, "Z": 0.0}
        self.bias = {"gx": 0.0, "gy": 0.0, "gz": 0.0}
        self.calibrating = False
        self.received_packets = 0
        self.lost_packets = 0
        self.data_queue = queue.Queue(maxsize=100)
        self.simulating = False
        self.sim_thread = None
        self.alert_state = False
        self.last_packet_time = time.time()
        self.connection_status = "CONNECTED"
        
        # Alert thresholds
        self.thresholds = {
            "Battery (V)": {"critical": 6.5, "warning": 7.0},
            "Temperature (°C)": {"critical": 40, "warning": 35},
            "Altitude (m)": {"max_safe": 200}
        }
        
        self.initUI()
        
        # Timers
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(self.UPDATE_INTERVAL_MS)
        
        self.alert_timer = QtCore.QTimer()
        self.alert_timer.timeout.connect(self.toggle_alert)
        self.alert_timer.start(self.ALERT_BLINK_INTERVAL)
        
        self.start_simulation()

    @property
    def params(self):
        return [
            "Altitude (m)", "Battery (V)", "Temperature (°C)", "Pressure (hPa)",
            "Accel X (m/s²)", "Accel Y (m/s²)", "Accel Z (m/s²)",
            "Gyro X (°/s)", "Gyro Y (°/s)", "Gyro Z (°/s)",
            "Mag X (µT)", "Mag Y (µT)", "Mag Z (µT)"
        ]

    def initUI(self):
        """Initialize championship-grade UI"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # ========== HEADER ==========
        header = self.create_header()
        main_layout.addWidget(header)
        
        # ========== MAIN CONTENT ==========
        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setSpacing(15)
        
        # Left panel - Status & Controls
        left_panel = self.create_left_panel()
        content_layout.addWidget(left_panel, 2)
        
        # Right panel - Graphs
        right_panel = self.create_graphs_panel()
        content_layout.addWidget(right_panel, 5)
        
        main_layout.addLayout(content_layout, 1)
        
        # ========== FOOTER ==========
        footer = self.create_footer()
        main_layout.addWidget(footer)

    def create_header(self):
        """Create professional header with live indicators"""
        header_frame = QtWidgets.QFrame()
        header_frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {self.PANEL_BG}, stop:1 {self.CARD_BG});
                border: 2px solid {self.ACCENT_COLOR};
                border-radius: 10px;
                padding: 15px;
            }}
        """)
        header_layout = QtWidgets.QHBoxLayout(header_frame)
        
        # Title section
        title_layout = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("🛰️ CANSAT MISSION CONTROL")
        title.setFont(QtGui.QFont("Consolas", 32, QtGui.QFont.Weight.Bold))
        title.setStyleSheet(f"color: {self.ACCENT_COLOR}; border: none;")
        title_layout.addWidget(title)
        
        subtitle = QtWidgets.QLabel("Real-Time Telemetry Monitoring System • Championship Edition")
        subtitle.setFont(QtGui.QFont("Consolas", 14))
        subtitle.setStyleSheet(f"color: {self.LABEL_COLOR}; border: none;")
        title_layout.addWidget(subtitle)
        header_layout.addLayout(title_layout, 3)
        
        # Live status indicator
        status_layout = QtWidgets.QVBoxLayout()
        self.live_indicator = QtWidgets.QLabel("● LIVE")
        self.live_indicator.setFont(QtGui.QFont("Consolas", 20, QtGui.QFont.Weight.Bold))
        self.live_indicator.setStyleSheet(f"color: {self.SUCCESS_COLOR}; border: none;")
        self.live_indicator.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        status_layout.addWidget(self.live_indicator)
        
        self.connection_label = QtWidgets.QLabel("TELEMETRY ACTIVE")
        self.connection_label.setFont(QtGui.QFont("Consolas", 12))
        self.connection_label.setStyleSheet(f"color: {self.SUCCESS_COLOR}; border: none;")
        self.connection_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        status_layout.addWidget(self.connection_label)
        header_layout.addLayout(status_layout, 1)
        
        return header_frame

    def create_left_panel(self):
        """Create left panel with status and controls"""
        panel = QtWidgets.QFrame()
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {self.PANEL_BG};
                border: 2px solid {self.ACCENT_COLOR};
                border-radius: 10px;
            }}
        """)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Mission Status Card
        mission_card = self.create_mission_status_card()
        layout.addWidget(mission_card)
        
        # Critical Parameters Card
        critical_card = self.create_critical_params_card()
        layout.addWidget(critical_card)
        
        # Telemetry Health Card
        health_card = self.create_telemetry_health_card()
        layout.addWidget(health_card)
        
        # GPS & Navigation Card
        gps_card = self.create_gps_card()
        layout.addWidget(gps_card)
        
        # Control Panel
        controls = self.create_control_panel()
        layout.addWidget(controls)
        
        layout.addStretch()
        return panel

    def create_mission_status_card(self):
        """Create mission status display card"""
        card = QtWidgets.QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {self.CARD_BG};
                border: 1px solid {self.ACCENT_COLOR};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        layout = QtWidgets.QVBoxLayout(card)
        
        title = QtWidgets.QLabel("⚡ MISSION STATUS")
        title.setFont(QtGui.QFont("Consolas", 16, QtGui.QFont.Weight.Bold))
        title.setStyleSheet(f"color: {self.ACCENT_COLOR}; border: none;")
        layout.addWidget(title)
        
        self.mission_status_label = QtWidgets.QLabel("NOMINAL")
        self.mission_status_label.setFont(QtGui.QFont("Consolas", 24, QtGui.QFont.Weight.Bold))
        self.mission_status_label.setStyleSheet(f"""
            color: {self.SUCCESS_COLOR};
            background-color: rgba(0, 255, 136, 0.1);
            border: 2px solid {self.SUCCESS_COLOR};
            border-radius: 5px;
            padding: 10px;
        """)
        self.mission_status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.mission_status_label)
        
        # Mission time
        time_layout = QtWidgets.QHBoxLayout()
        time_label = QtWidgets.QLabel("Mission Time:")
        time_label.setFont(QtGui.QFont("Consolas", 12))
        time_label.setStyleSheet(f"color: {self.LABEL_COLOR}; border: none;")
        self.time_value = QtWidgets.QLabel("00:00:00")
        self.time_value.setFont(QtGui.QFont("Consolas", 18, QtGui.QFont.Weight.Bold))
        self.time_value.setStyleSheet(f"color: {self.ACCENT_COLOR}; border: none;")
        time_layout.addWidget(time_label)
        time_layout.addStretch()
        time_layout.addWidget(self.time_value)
        layout.addLayout(time_layout)
        
        return card

    def create_critical_params_card(self):
        """Create critical parameters display"""
        card = QtWidgets.QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {self.CARD_BG};
                border: 1px solid {self.ACCENT_COLOR};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        layout = QtWidgets.QVBoxLayout(card)
        
        title = QtWidgets.QLabel("📊 CRITICAL PARAMETERS")
        title.setFont(QtGui.QFont("Consolas", 16, QtGui.QFont.Weight.Bold))
        title.setStyleSheet(f"color: {self.ACCENT_COLOR}; border: none;")
        layout.addWidget(title)
        
        self.critical_params = {}
        params = [
            ("Altitude", "m", "🚀"),
            ("Battery", "V", "🔋"),
            ("Temperature", "°C", "🌡️"),
            ("Speed", "m/s", "⚡")
        ]
        
        for param, unit, icon in params:
            param_layout = QtWidgets.QVBoxLayout()
            
            label = QtWidgets.QLabel(f"{icon} {param}")
            label.setFont(QtGui.QFont("Consolas", 11, QtGui.QFont.Weight.Bold))
            label.setStyleSheet(f"color: {self.LABEL_COLOR}; border: none;")
            param_layout.addWidget(label)
            
            value = QtWidgets.QLabel(f"-- {unit}")
            value.setFont(QtGui.QFont("Consolas", 20, QtGui.QFont.Weight.Bold))
            value.setStyleSheet(f"color: {self.TEXT_COLOR}; border: none;")
            param_layout.addWidget(value)
            
            self.critical_params[param] = value
            layout.addLayout(param_layout)
            
            # Add separator except for last item
            if param != "Speed":
                separator = QtWidgets.QFrame()
                separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
                separator.setStyleSheet(f"background-color: {self.ACCENT_COLOR};")
                separator.setFixedHeight(1)
                layout.addWidget(separator)
        
        return card

    def create_telemetry_health_card(self):
        """Create telemetry health monitoring card"""
        card = QtWidgets.QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {self.CARD_BG};
                border: 1px solid {self.ACCENT_COLOR};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        layout = QtWidgets.QVBoxLayout(card)
        
        title = QtWidgets.QLabel("📡 TELEMETRY HEALTH")
        title.setFont(QtGui.QFont("Consolas", 16, QtGui.QFont.Weight.Bold))
        title.setStyleSheet(f"color: {self.ACCENT_COLOR}; border: none;")
        layout.addWidget(title)
        
        # Packets
        packets_layout = QtWidgets.QHBoxLayout()
        packets_label = QtWidgets.QLabel("Packets:")
        packets_label.setFont(QtGui.QFont("Consolas", 11))
        packets_label.setStyleSheet(f"color: {self.LABEL_COLOR}; border: none;")
        self.packets_value = QtWidgets.QLabel("0 / 0")
        self.packets_value.setFont(QtGui.QFont("Consolas", 14, QtGui.QFont.Weight.Bold))
        self.packets_value.setStyleSheet(f"color: {self.SUCCESS_COLOR}; border: none;")
        packets_layout.addWidget(packets_label)
        packets_layout.addStretch()
        packets_layout.addWidget(self.packets_value)
        layout.addLayout(packets_layout)
        
        # Packet loss rate
        loss_layout = QtWidgets.QHBoxLayout()
        loss_label = QtWidgets.QLabel("Loss Rate:")
        loss_label.setFont(QtGui.QFont("Consolas", 11))
        loss_label.setStyleSheet(f"color: {self.LABEL_COLOR}; border: none;")
        self.loss_value = QtWidgets.QLabel("0.0%")
        self.loss_value.setFont(QtGui.QFont("Consolas", 14, QtGui.QFont.Weight.Bold))
        self.loss_value.setStyleSheet(f"color: {self.SUCCESS_COLOR}; border: none;")
        loss_layout.addWidget(loss_label)
        loss_layout.addStretch()
        loss_layout.addWidget(self.loss_value)
        layout.addLayout(loss_layout)
        
        # Signal quality bar
        signal_label = QtWidgets.QLabel("Signal Quality:")
        signal_label.setFont(QtGui.QFont("Consolas", 11))
        signal_label.setStyleSheet(f"color: {self.LABEL_COLOR}; border: none;")
        layout.addWidget(signal_label)
        
        self.signal_bar = QtWidgets.QProgressBar()
        self.signal_bar.setRange(0, 100)
        self.signal_bar.setValue(95)
        self.signal_bar.setTextVisible(True)
        self.signal_bar.setFormat("%p%")
        self.signal_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid {self.ACCENT_COLOR};
                border-radius: 5px;
                text-align: center;
                background-color: {self.PANEL_BG};
                color: {self.TEXT_COLOR};
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background-color: {self.SUCCESS_COLOR};
            }}
        """)
        layout.addWidget(self.signal_bar)
        
        return card

    def create_gps_card(self):
        """Create GPS & navigation card"""
        card = QtWidgets.QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {self.CARD_BG};
                border: 1px solid {self.ACCENT_COLOR};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        layout = QtWidgets.QVBoxLayout(card)
        
        title = QtWidgets.QLabel("🌍 GPS & NAVIGATION")
        title.setFont(QtGui.QFont("Consolas", 16, QtGui.QFont.Weight.Bold))
        title.setStyleSheet(f"color: {self.ACCENT_COLOR}; border: none;")
        layout.addWidget(title)
        
        self.gps_labels = {}
        gps_params = [("Latitude", "°"), ("Longitude", "°"), ("Heading", "°")]
        
        for param, unit in gps_params:
            param_layout = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(f"{param}:")
            label.setFont(QtGui.QFont("Consolas", 11))
            label.setStyleSheet(f"color: {self.LABEL_COLOR}; border: none;")
            value = QtWidgets.QLabel(f"-- {unit}")
            value.setFont(QtGui.QFont("Consolas", 13, QtGui.QFont.Weight.Bold))
            value.setStyleSheet(f"color: {self.TEXT_COLOR}; border: none;")
            param_layout.addWidget(label)
            param_layout.addStretch()
            param_layout.addWidget(value)
            layout.addLayout(param_layout)
            self.gps_labels[param] = value
        
        return card

    def create_control_panel(self):
        """Create control buttons panel"""
        card = QtWidgets.QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {self.CARD_BG};
                border: 1px solid {self.ACCENT_COLOR};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        layout = QtWidgets.QVBoxLayout(card)
        
        title = QtWidgets.QLabel("⚙️ CONTROLS")
        title.setFont(QtGui.QFont("Consolas", 16, QtGui.QFont.Weight.Bold))
        title.setStyleSheet(f"color: {self.ACCENT_COLOR}; border: none;")
        layout.addWidget(title)
        
        button_style = f"""
            QPushButton {{
                background-color: {self.ACCENT_COLOR};
                color: {self.BG_COLOR};
                border: none;
                border-radius: 5px;
                padding: 12px;
                font-size: 13px;
                font-weight: bold;
                font-family: Consolas;
            }}
            QPushButton:hover {{
                background-color: {self.SUCCESS_COLOR};
            }}
            QPushButton:pressed {{
                background-color: {self.WARNING_COLOR};
            }}
            QPushButton:disabled {{
                background-color: {self.PANEL_BG};
                color: {self.LABEL_COLOR};
            }}
        """
        
        self.start_btn = QtWidgets.QPushButton("▶ START MONITORING")
        self.stop_btn = QtWidgets.QPushButton("⏸ PAUSE MONITORING")
        self.calib_btn = QtWidgets.QPushButton("🎯 CALIBRATE GYRO")
        self.export_btn = QtWidgets.QPushButton("💾 EXPORT DATA")
        
        for btn in [self.start_btn, self.stop_btn, self.calib_btn, self.export_btn]:
            btn.setStyleSheet(button_style)
            btn.setMinimumHeight(45)
            layout.addWidget(btn)
        
        self.start_btn.clicked.connect(lambda: self.update_timer.start(self.UPDATE_INTERVAL_MS))
        self.stop_btn.clicked.connect(lambda: self.update_timer.stop())
        self.calib_btn.clicked.connect(self.start_calibration)
        self.export_btn.clicked.connect(self.export_data)
        
        return card

    def create_graphs_panel(self):
        """Create optimized graphs panel with proper spacing"""
        panel = QtWidgets.QFrame()
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {self.PANEL_BG};
                border: 2px solid {self.ACCENT_COLOR};
                border-radius: 10px;
            }}
        """)
        
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"border: none; background-color: {self.PANEL_BG};")
        
        scroll_content = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(scroll_content)
        layout.setSpacing(20)  # Increased spacing
        layout.setContentsMargins(15, 15, 15, 15)
        
        self.graph_lines = {}
        self.graph_canvases = {}
        
        graph_configs = [
            ("Altitude (m)", 0, 0, "#00d9ff", ["Altitude (m)"]),
            ("Battery (V)", 0, 1, "#00ff88", ["Battery (V)"]),
            ("Temperature (°C)", 1, 0, "#ff3366", ["Temperature (°C)"]),
            ("Pressure (hPa)", 1, 1, "#ffaa00", ["Pressure (hPa)"]),
            ("Acceleration", 2, 0, None, ["Accel X (m/s²)", "Accel Y (m/s²)", "Accel Z (m/s²)"]),
            ("Gyroscope", 2, 1, None, ["Gyro X (°/s)", "Gyro Y (°/s)", "Gyro Z (°/s)"]),
            ("Magnetometer", 3, 0, None, ["Mag X (µT)", "Mag Y (µT)", "Mag Z (µT)"])
        ]
        
        for title, row, col, color, params in graph_configs:
            frame = self.create_graph_frame(title, color, params)
            if title == "Magnetometer":
                layout.addWidget(frame, row, col, 1, 2)  # Span 2 columns
            else:
                layout.addWidget(frame, row, col)
        
        layout.setRowStretch(4, 1)  # Add stretch at bottom
        scroll.setWidget(scroll_content)
        
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)
        
        return panel

    def create_graph_frame(self, title, color, params):
        """Create individual graph frame with professional styling"""
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {self.CARD_BG};
                border: 2px solid {self.ACCENT_COLOR};
                border-radius: 8px;
            }}
        """)
        frame.setMinimumHeight(280)  # Increased minimum height
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        
        try:
            fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=90)
            fig.patch.set_facecolor(self.CARD_BG)
            ax.set_facecolor(self.CARD_BG)
            
            # Styling
            ax.set_title(title, color=self.ACCENT_COLOR, fontsize=14, fontweight='bold', pad=10)
            ax.tick_params(colors=self.LABEL_COLOR, labelsize=10)
            ax.grid(True, alpha=0.2, color=self.LABEL_COLOR, linestyle='--')
            ax.spines['bottom'].set_color(self.ACCENT_COLOR)
            ax.spines['top'].set_color(self.ACCENT_COLOR)
            ax.spines['left'].set_color(self.ACCENT_COLOR)
            ax.spines['right'].set_color(self.ACCENT_COLOR)
            
            # Plot lines
            if len(params) == 1:
                line, = ax.plot([], [], color=color, linewidth=2.5, label=params[0])
                self.graph_lines[params[0]] = line
                ax.set_ylabel(params[0].split()[0], color=self.LABEL_COLOR, fontsize=11)
            else:
                colors_multi = ['#00d9ff', '#00ff88', '#ff3366']
                for i, param in enumerate(params):
                    line, = ax.plot([], [], color=colors_multi[i], linewidth=2, label=param.split()[1][0])
                    self.graph_lines[param] = line
                legend = ax.legend(facecolor=self.CARD_BG, labelcolor=self.TEXT_COLOR, 
                                 framealpha=0.9, loc='upper right', fontsize=9)
                legend.get_frame().set_edgecolor(self.ACCENT_COLOR)
            
            ax.set_xlabel('Time (s)', color=self.LABEL_COLOR, fontsize=10)
            
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: transparent;")
            layout.addWidget(canvas)
            self.graph_canvases[title] = (canvas, ax)
            
        except Exception as e:
            error_label = QtWidgets.QLabel(f"Graph Error: {str(e)}")
            error_label.setStyleSheet(f"color: {self.CRITICAL_COLOR};")
            layout.addWidget(error_label)
        
        return frame

    def create_footer(self):
        """Create footer with system info"""
        footer = QtWidgets.QFrame()
        footer.setStyleSheet(f"""
            QFrame {{
                background-color: {self.PANEL_BG};
                border: 1px solid {self.ACCENT_COLOR};
                border-radius: 5px;
                padding: 8px;
            }}
        """)
        layout = QtWidgets.QHBoxLayout(footer)
        
        system_time = QtWidgets.QLabel(f"System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        system_time.setFont(QtGui.QFont("Consolas", 10))
        system_time.setStyleSheet(f"color: {self.LABEL_COLOR}; border: none;")
        
        version = QtWidgets.QLabel("v2.0 Championship Edition")
        version.setFont(QtGui.QFont("Consolas", 10, QtGui.QFont.Weight.Bold))
        version.setStyleSheet(f"color: {self.ACCENT_COLOR}; border: none;")
        
        layout.addWidget(system_time)
        layout.addStretch()
        layout.addWidget(version)
        
        return footer

    def start_calibration(self):
        """Start gyro calibration with visual feedback"""
        if not self.calibrating:
            self.calibrating = True
            self.bias = {"gx": 0.0, "gy": 0.0, "gz": 0.0}
            self.bias_samples = {"gx": [], "gy": [], "gz": []}
            self.calib_btn.setText("⏳ CALIBRATING...")
            self.calib_btn.setEnabled(False)
            self.mission_status_label.setText("CALIBRATING")
            self.mission_status_label.setStyleSheet(f"""
                color: {self.WARNING_COLOR};
                background-color: rgba(255, 170, 0, 0.1);
                border: 2px solid {self.WARNING_COLOR};
                border-radius: 5px;
                padding: 10px;
            """)

    def start_simulation(self):
        """Start simulation thread"""
        if not self.simulating:
            self.simulating = True
            self.sim_thread = threading.Thread(target=self.simulate_data, daemon=True)
            self.sim_thread.start()

    def simulate_data(self):
        """Enhanced simulation with realistic data"""
        phase = "ascent"
        base_altitude = 0
        
        while self.simulating:
            try:
                current_time = time.time()
                dt = current_time - self.last_time if self.last_time > 0 else 0.1
                self.last_time = current_time
                self.time += dt

                if self.calibrating:
                    gyro = {"gx": random.uniform(-5, 5), "gy": random.uniform(-5, 5), "gz": random.uniform(-5, 5)}
                    for axis in ["gx", "gy", "gz"]:
                        self.bias_samples[axis].append(gyro[axis])
                        if len(self.bias_samples[axis]) >= 200:
                            self.bias[axis] = sum(self.bias_samples[axis]) / 200
                            self.calibrating = False
                    if not self.calibrating:
                        self.calib_btn.setText("🎯 CALIBRATE GYRO")
                        self.calib_btn.setEnabled(True)
                    self.data_queue.put(None)
                    time.sleep(0.1)
                    continue

                # Simulate mission phases
                if self.time < 30:
                    phase = "ascent"
                    base_altitude = 50 + (self.time * 3) + random.uniform(-2, 2)
                elif self.time < 60:
                    phase = "descent"
                    base_altitude = 140 - ((self.time - 30) * 2.5) + random.uniform(-3, 3)
                else:
                    phase = "landing"
                    base_altitude = max(0, 65 - (self.time - 60) * 1.5) + random.uniform(-1, 1)

                # Generate realistic sensor data
                data = {
                    "Altitude (m)": max(0, base_altitude),
                    "Battery (V)": max(6.0, 8.4 - (self.time * 0.005) + random.uniform(-0.1, 0.1)),
                    "Temperature (°C)": 25 + random.uniform(-2, 3) + (base_altitude * 0.01),
                    "Pressure (hPa)": 1013 - (base_altitude * 0.12) + random.uniform(-2, 2),
                    "Accel X (m/s²)": random.uniform(-2, 2) + (1 if phase == "ascent" else -0.5),
                    "Accel Y (m/s²)": random.uniform(-1.5, 1.5),
                    "Accel Z (m/s²)": 9.81 + random.uniform(-1, 1) + (2 if phase == "ascent" else -1),
                    "Gyro X (°/s)": random.uniform(-15, 15) - self.bias.get("gx", 0),
                    "Gyro Y (°/s)": random.uniform(-15, 15) - self.bias.get("gy", 0),
                    "Gyro Z (°/s)": random.uniform(-20, 20) - self.bias.get("gz", 0),
                    "Mag X (µT)": 30 + random.uniform(-5, 5),
                    "Mag Y (µT)": 20 + random.uniform(-5, 5),
                    "Mag Z (µT)": -40 + random.uniform(-5, 5)
                }

                # Simulate packet loss
                if random.random() > 0.95:
                    self.lost_packets += 1
                else:
                    self.received_packets += 1
                    self.data_queue.put(data)

                time.sleep(0.05)
            except Exception as e:
                print(f"Simulation error: {e}")
                time.sleep(0.1)

    def update_display(self):
        """Update all displays with latest data"""
        try:
            # Process queued data
            latest_data = None
            while not self.data_queue.empty():
                try:
                    latest_data = self.data_queue.get_nowait()
                except queue.Empty:
                    break

            if latest_data is None:
                return

            # Update graph data
            for param in self.params:
                if param in latest_data:
                    self.graph_data[param].append((self.time, latest_data[param]))
                    if len(self.graph_data[param]) > self.MAX_HISTORY:
                        self.graph_data[param].pop(0)

            # Update graphs
            for param, line in self.graph_lines.items():
                if self.graph_data[param]:
                    times, values = zip(*self.graph_data[param])
                    line.set_data(times, values)

            # Auto-scale graphs
            for title, (canvas, ax) in self.graph_canvases.items():
                ax.relim()
                ax.autoscale_view(True, True, True)
                canvas.draw_idle()

            # Update critical parameters
            if "Altitude (m)" in latest_data:
                self.critical_params["Altitude"].setText(f"{latest_data['Altitude (m)']:.1f} m")
                self.critical_params["Altitude"].setStyleSheet(
                    f"color: {self.get_status_color('Altitude (m)', latest_data['Altitude (m)'])}; border: none;"
                )

            if "Battery (V)" in latest_data:
                self.critical_params["Battery"].setText(f"{latest_data['Battery (V)']:.2f} V")
                self.critical_params["Battery"].setStyleSheet(
                    f"color: {self.get_status_color('Battery (V)', latest_data['Battery (V)'])}; border: none;"
                )

            if "Temperature (°C)" in latest_data:
                self.critical_params["Temperature"].setText(f"{latest_data['Temperature (°C)']:.1f} °C")
                self.critical_params["Temperature"].setStyleSheet(
                    f"color: {self.get_status_color('Temperature (°C)', latest_data['Temperature (°C)'])}; border: none;"
                )

            # Calculate speed from altitude change
            if len(self.graph_data["Altitude (m)"]) > 1:
                alt_data = self.graph_data["Altitude (m)"]
                speed = abs(alt_data[-1][1] - alt_data[-2][1]) / (alt_data[-1][0] - alt_data[-2][0])
                self.critical_params["Speed"].setText(f"{speed:.1f} m/s")

            # Update mission time
            hours = int(self.time // 3600)
            minutes = int((self.time % 3600) // 60)
            seconds = int(self.time % 60)
            self.time_value.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

            # Update telemetry health
            total_packets = self.received_packets + self.lost_packets
            if total_packets > 0:
                loss_rate = (self.lost_packets / total_packets) * 100
                self.packets_value.setText(f"{self.received_packets} / {total_packets}")
                self.loss_value.setText(f"{loss_rate:.1f}%")
                
                # Update signal quality
                signal_quality = max(0, 100 - (loss_rate * 10))
                self.signal_bar.setValue(int(signal_quality))
                
                # Color code based on quality
                if loss_rate < 5:
                    self.loss_value.setStyleSheet(f"color: {self.SUCCESS_COLOR}; border: none;")
                elif loss_rate < 15:
                    self.loss_value.setStyleSheet(f"color: {self.WARNING_COLOR}; border: none;")
                else:
                    self.loss_value.setStyleSheet(f"color: {self.CRITICAL_COLOR}; border: none;")

            # Update GPS (simulated)
            self.gps_labels["Latitude"].setText(f"{31.5204 + random.uniform(-0.001, 0.001):.5f}°")
            self.gps_labels["Longitude"].setText(f"{74.3587 + random.uniform(-0.001, 0.001):.5f}°")
            self.gps_labels["Heading"].setText(f"{random.uniform(0, 360):.1f}°")

            # Update mission status
            self.update_mission_status(latest_data)

            # Check connection health
            time_since_packet = time.time() - self.last_packet_time
            if time_since_packet > 2:
                self.connection_status = "SIGNAL WEAK"
                self.connection_label.setText("SIGNAL WEAK")
                self.connection_label.setStyleSheet(f"color: {self.WARNING_COLOR}; border: none;")
            else:
                self.connection_status = "CONNECTED"
                self.connection_label.setText("TELEMETRY ACTIVE")
                self.connection_label.setStyleSheet(f"color: {self.SUCCESS_COLOR}; border: none;")

            self.last_packet_time = time.time()

        except Exception as e:
            print(f"Display update error: {e}")

    def get_status_color(self, param, value):
        """Get color based on parameter value and thresholds"""
        if param == "Battery (V)":
            if value < self.thresholds["Battery (V)"]["critical"]:
                return self.CRITICAL_COLOR
            elif value < self.thresholds["Battery (V)"]["warning"]:
                return self.WARNING_COLOR
            return self.SUCCESS_COLOR
        elif param == "Temperature (°C)":
            if value > self.thresholds["Temperature (°C)"]["critical"]:
                return self.CRITICAL_COLOR
            elif value > self.thresholds["Temperature (°C)"]["warning"]:
                return self.WARNING_COLOR
            return self.SUCCESS_COLOR
        elif param == "Altitude (m)":
            if value > self.thresholds["Altitude (m)"]["max_safe"]:
                return self.WARNING_COLOR
            return self.SUCCESS_COLOR
        return self.TEXT_COLOR

    def update_mission_status(self, data):
        """Update overall mission status based on all parameters"""
        status = "NOMINAL"
        color = self.SUCCESS_COLOR
        bg_alpha = "0.1"

        # Check critical conditions
        if data.get("Battery (V)", 10) < self.thresholds["Battery (V)"]["critical"]:
            status = "CRITICAL - LOW BATTERY"
            color = self.CRITICAL_COLOR
        elif data.get("Temperature (°C)", 0) > self.thresholds["Temperature (°C)"]["critical"]:
            status = "CRITICAL - OVERHEAT"
            color = self.CRITICAL_COLOR
        elif data.get("Battery (V)", 10) < self.thresholds["Battery (V)"]["warning"]:
            status = "WARNING - BATTERY LOW"
            color = self.WARNING_COLOR
        elif data.get("Temperature (°C)", 0) > self.thresholds["Temperature (°C)"]["warning"]:
            status = "WARNING - HIGH TEMP"
            color = self.WARNING_COLOR
        elif data.get("Altitude (m)", 0) > self.thresholds["Altitude (m)"]["max_safe"]:
            status = "WARNING - HIGH ALTITUDE"
            color = self.WARNING_COLOR

        if not self.calibrating:
            self.mission_status_label.setText(status)
            self.mission_status_label.setStyleSheet(f"""
                color: {color};
                background-color: rgba({self._hex_to_rgb(color)}, {bg_alpha});
                border: 2px solid {color};
                border-radius: 5px;
                padding: 10px;
            """)

    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB string"""
        hex_color = hex_color.lstrip('#')
        return ', '.join(str(int(hex_color[i:i+2], 16)) for i in (0, 2, 4))

    def toggle_alert(self):
        """Toggle alert indicator for critical states"""
        self.alert_state = not self.alert_state
        if self.mission_status_label.text().startswith("CRITICAL"):
            if self.alert_state:
                self.live_indicator.setText("⬤ ALERT")
                self.live_indicator.setStyleSheet(f"color: {self.CRITICAL_COLOR}; border: none;")
            else:
                self.live_indicator.setText("● ALERT")
                self.live_indicator.setStyleSheet(f"color: {self.WARNING_COLOR}; border: none;")
        else:
            self.live_indicator.setText("● LIVE")
            self.live_indicator.setStyleSheet(f"color: {self.SUCCESS_COLOR}; border: none;")

    def export_data(self):
        """Export telemetry data to CSV"""
        try:
            filename = f"cansat_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['Time (s)'] + self.params)
                
                # Get all time points
                if self.graph_data[self.params[0]]:
                    times = [t for t, _ in self.graph_data[self.params[0]]]
                    
                    for time_point in times:
                        row = [time_point]
                        for param in self.params:
                            value = next((v for t, v in self.graph_data[param] if abs(t - time_point) < 0.01), None)
                            row.append(value if value is not None else '')
                        writer.writerow(row)
            
            # Show success message
            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
            msg.setWindowTitle("Export Successful")
            msg.setText(f"Data exported to:\n{filename}")
            msg.setStyleSheet(f"""
                QMessageBox {{
                    background-color: {self.CARD_BG};
                    color: {self.TEXT_COLOR};
                }}
                QPushButton {{
                    background-color: {self.ACCENT_COLOR};
                    color: {self.BG_COLOR};
                    border-radius: 5px;
                    padding: 8px 15px;
                    font-weight: bold;
                }}
            """)
            msg.exec()
            
        except Exception as e:
            # Show error message
            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            msg.setWindowTitle("Export Failed")
            msg.setText(f"Error exporting data:\n{str(e)}")
            msg.setStyleSheet(f"""
                QMessageBox {{
                    background-color: {self.CARD_BG};
                    color: {self.TEXT_COLOR};
                }}
                QPushButton {{
                    background-color: {self.CRITICAL_COLOR};
                    color: white;
                    border-radius: 5px;
                    padding: 8px 15px;
                    font-weight: bold;
                }}
            """)
            msg.exec()

    def closeEvent(self, event):
        """Clean shutdown"""
        self.simulating = False
        if self.sim_thread and self.sim_thread.is_alive():
            self.sim_thread.join(timeout=1)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set application-wide font
    font = QtGui.QFont("Consolas", 10)
    app.setFont(font)
    
    gui = CanSatGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()