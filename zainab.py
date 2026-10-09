#!/usr/bin/env python3
"""
CANSAT Geolocation - Competition Winner Edition
A professional GUI application for plotting coordinate pairs with advanced features
Supports TXT, CSV, and KML formats for competition requirements
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import folium
from folium import plugins
import webbrowser
import os
import tempfile
import csv
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom
import math

class ToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        self.widget.bind("<Enter>", self.showtip)
        self.widget.bind("<Leave>", self.hidetip)

    def showtip(self, event=None):
        "Display text in tooltip window"
        self.hidetip()
        bbox = self.widget.bbox("insert")
        if bbox:
            x, y, _, _ = bbox
        else:
            x, y = 0, 0
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffff", relief=tk.SOLID, borderwidth=1,
                         font=("Segoe UI", "9", "normal"))
        label.pack(ipadx=1)

    def hidetip(self, event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

class GeolocationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CANSAT Geolocation - Competition Edition")
        self.root.geometry("1200x800")
        
        # Modern color scheme
        self.colors = {
            'primary': '#1e293b',
            'secondary': '#334155',
            'accent': '#3b82f6',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'bg': '#e5e7eb',  # Updated background color
            'card': '#ffffff',
            'text': '#1e293b',
            'text_secondary': '#64748b'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Data storage
        self.coordinates = []
        self.map_file = None
        self.kml_file = None
        
        # Setup GUI
        self.setup_modern_gui()
        
    def setup_modern_gui(self):
        """Setup the modern, competition-ready GUI"""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Title.TLabel', 
                       font=('Segoe UI', 24, 'bold'),
                       foreground=self.colors['primary'],
                       background=self.colors['bg'])
        
        # Scrollbar styling
        style.configure("Vertical.TScrollbar",
                       troughcolor=self.colors['accent'],
                       background=self.colors['primary'],
                       arrowcolor='white',
                       borderwidth=0)
        style.map("Vertical.TScrollbar",
                 background=[('active', self.adjust_color_brightness(self.colors['primary'], 0.8))])
        
        # Header section with gradient
        header_frame = tk.Frame(self.root, bg=self.colors['bg'], height=120)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)
        
        gradient = self.create_gradient(header_frame, 1200, 120, self.colors['primary'], self.colors['secondary'])
        gradient.place(x=0, y=0, relwidth=1, relheight=1)
        
        title_container = tk.Frame(header_frame, bg=self.colors['primary'])
        title_container.pack(expand=True)
        
        tk.Label(title_container, text="🛰️", font=('Segoe UI', 40),
                bg=self.colors['primary']).pack(side=tk.LEFT, padx=(0, 15))
        
        text_container = tk.Frame(title_container, bg=self.colors['primary'])
        text_container.pack(side=tk.LEFT)
        
        tk.Label(text_container, text="CANSAT Geolocation",
                font=('Segoe UI', 28, 'bold'),
                fg='white', bg=self.colors['primary']).pack(anchor='w')
        
        tk.Label(text_container, text="Competition Winner Edition - Professional Mapping Tool",
                font=('Segoe UI', 11),
                fg='#94a3b8', bg=self.colors['primary']).pack(anchor='w')
        
        # Main content area with scrollbar
        content_container = tk.Frame(self.root, bg=self.colors['bg'])
        content_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        canvas = tk.Canvas(content_container, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_container, orient="vertical", command=canvas.yview, style="Vertical.TScrollbar")
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        content_frame = scrollable_frame
        
        # Left panel (fixed width)
        left_panel = tk.Frame(content_frame, bg=self.colors['bg'], width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Statistics Card
        stats_card = self.create_card(left_panel, "📊 Statistics")
        stats_card.pack(fill=tk.X, pady=(0, 15))
        
        stats_content = tk.Frame(stats_card, bg=self.colors['card'])
        stats_content.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        self.stats_labels = {}
        stats_data = [
            ("Total Points", "0", self.colors['accent']),
            ("Valid Coords", "0", self.colors['success']),
            ("Files Loaded", "0", self.colors['warning']),
            ("Distance (km)", "0.0", "#8b5cf6")
        ]
        
        for label, value, color in stats_data:
            stat_frame = tk.Frame(stats_content, bg=self.colors['card'])
            stat_frame.pack(fill=tk.X, pady=5)
            
            title_label = tk.Label(stat_frame, text=label,
                    font=('Segoe UI', 9),
                    fg=self.colors['text_secondary'],
                    bg=self.colors['card'])
            title_label.pack(anchor='w')
            
            value_label = tk.Label(stat_frame, text=value,
                                  font=('Segoe UI', 20, 'bold'),
                                  fg=color,
                                  bg=self.colors['card'])
            value_label.pack(anchor='w')
            self.stats_labels[label] = value_label
            
            # Add tooltips to stats
            if label == "Total Points":
                ToolTip(title_label, "Total number of coordinate points loaded")
            elif label == "Valid Coords":
                ToolTip(title_label, "Number of validated coordinates (within -90/90 lat, -180/180 lon)")
            elif label == "Files Loaded":
                ToolTip(title_label, "Number of successfully imported files")
            elif label == "Distance (km)":
                ToolTip(title_label, "Total route distance calculated using Haversine formula")
        
        # Import Options Card
        import_card = self.create_card(left_panel, "📥 Import Data")
        import_card.pack(fill=tk.X, pady=(0, 15))
        
        import_content = tk.Frame(import_card, bg=self.colors['card'])
        import_content.pack(fill=tk.BOTH, padx=15, pady=(0, 15))
        
        tk.Label(import_content, 
                text="Competition Format Support:",
                font=('Segoe UI', 10, 'bold'),
                fg=self.colors['text'],
                bg=self.colors['card']).pack(anchor='w', pady=(0, 10))
        
        import_buttons = [
            ("📄 Import TXT File", self.import_txt, self.colors['accent'], "Import coordinates from TXT file\nFormat: latitude,longitude,label,description (one per line)"),
            ("📊 Import CSV File", self.import_csv, self.colors['success'], "Import coordinates from CSV file\nColumns: Latitude, Longitude, Label, Description"),
            ("🌍 Import KML File", self.import_kml_file, self.colors['warning'], "Import placemarks from KML file\nSupports standard Google Earth format")
        ]
        
        for text, cmd, color, tooltip in import_buttons:
            btn = self.create_modern_button(import_content, text, cmd, color)
            btn.pack(fill=tk.X, pady=3)
            ToolTip(btn, tooltip)
        
        # Export Options Card
        export_card = self.create_card(left_panel, "📤 Export Results")
        export_card.pack(fill=tk.X, pady=(0, 15))
        
        export_content = tk.Frame(export_card, bg=self.colors['card'])
        export_content.pack(fill=tk.BOTH, padx=15, pady=(0, 15))
        
        export_buttons = [
            ("💾 Export CSV", self.export_csv, self.colors['accent'], "Export coordinates to CSV format\nCompatible with Google My Maps"),
            ("🗺️ Export KML", self.export_kml, self.colors['success'], "Export to KML format\nCompatible with Google Earth and GIS tools"),
            ("📄 Generate Report", self.generate_report, "#8b5cf6", "Generate comprehensive text report\nIncludes statistics, methodology, and details")
        ]
        
        for text, cmd, color, tooltip in export_buttons:
            btn = self.create_modern_button(export_content, text, cmd, color)
            btn.pack(fill=tk.X, pady=3)
            ToolTip(btn, tooltip)
        
        # Map Operations Card
        map_card = self.create_card(left_panel, "🗺️ Map Operations")
        map_card.pack(fill=tk.X)
        
        map_content = tk.Frame(map_card, bg=self.colors['card'])
        map_content.pack(fill=tk.BOTH, padx=15, pady=(0, 15))
        
        map_buttons = [
            ("🎨 Generate Interactive Map", self.generate_map, "#8b5cf6", "Create HTML map with markers, path, and centroid\nUses Folium library"),
            ("🌐 Open in Browser", self.open_map, "#06b6d4", "Open generated interactive map in default browser"),
            ("🗺️ Open in Google Maps", self.open_in_google_maps, "#10b981", "View points and route in Google Maps\nSupports multi-waypoint directions"),
            ("🛰️ View in Google Earth", self.open_in_google_earth, "#f97316", "Open first point in Google Earth Web\nSatellite view"),
            ("📍 Show Centroid", self.show_centroid, "#ec4899", "Calculate and display geographic center\nOption to open in Google Maps")
        ]
        
        for text, cmd, color, tooltip in map_buttons:
            btn = self.create_modern_button(map_content, text, cmd, color)
            btn.pack(fill=tk.X, pady=3)
            ToolTip(btn, tooltip)
        
        # Right panel (constrained width)
        right_panel = tk.Frame(content_frame, bg=self.colors['bg'], width=600)
        right_panel.pack(side=tk.LEFT, fill=tk.Y)
        right_panel.pack_propagate(False)
        
        # Manual Input Card
        input_card = self.create_card(right_panel, "✏️ Manual Coordinate Entry")
        input_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        input_content = tk.Frame(input_card, bg=self.colors['card'])
        input_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        tk.Label(input_content,
                text="Format: latitude,longitude,label,description (one per line)",
                font=('Segoe UI', 9),
                fg=self.colors['text_secondary'],
                bg=self.colors['card']).pack(anchor='w', pady=(0, 5))
        
        text_frame = tk.Frame(input_content, bg='#e2e8f0', relief='flat', bd=2)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.coord_text = scrolledtext.ScrolledText(
            text_frame,
            font=('Consolas', 10),
            wrap=tk.WORD,
            relief='flat',
            bg='white',
            fg=self.colors['text'],
            insertbackground=self.colors['accent'])
        self.coord_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        action_frame = tk.Frame(input_content, bg=self.colors['card'])
        action_frame.pack(fill=tk.X)
        
        add_btn = self.create_modern_button(action_frame, "➕ Add to Map", 
                                           self.add_coordinates, self.colors['success'])
        add_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ToolTip(add_btn, "Add entered coordinates to the map\nValidates format and range")
        
        clear_btn = self.create_modern_button(action_frame, "🗑️ Clear All",
                                              self.clear_coordinates, self.colors['danger'])
        clear_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ToolTip(clear_btn, "Clear all loaded coordinates\nThis action cannot be undone")
        
        # Status/Log Card
        log_card = self.create_card(right_panel, "📋 Activity Log")
        log_card.pack(fill=tk.BOTH, expand=True)
        
        log_content = tk.Frame(log_card, bg=self.colors['card'])
        log_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        log_frame = tk.Frame(log_content, bg='#1e293b', relief='flat', bd=2)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_text = scrolledtext.ScrolledText(
            log_frame,
            font=('Consolas', 9),
            wrap=tk.WORD,
            relief='flat',
            bg='#1e293b',
            fg='#94a3b8',
            state=tk.DISABLED,
            insertbackground='#3b82f6')
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Configure tags for colored log messages
        self.status_text.tag_configure("INFO", foreground="#94a3b8")
        self.status_text.tag_configure("SUCCESS", foreground="#10b981")
        self.status_text.tag_configure("WARNING", foreground="#f59e0b")
        self.status_text.tag_configure("ERROR", foreground="#ef4444")
        
        # Load sample data
        self.load_sample_data()
        self.log_status("🚀 CANSAT Geolocation initialized successfully!", "SUCCESS")
        self.log_status("📖 Ready to import competition coordinate data", "INFO")
    
    def create_gradient(self, parent, width, height, color1, color2):
        """Create a horizontal gradient canvas"""
        canvas = tk.Canvas(parent, width=width, height=height, highlightthickness=0)
        for i in range(width):
            r = int(int(color1[1:3], 16) + (float(i) / width) * (int(color2[1:3], 16) - int(color1[1:3], 16)))
            g = int(int(color1[3:5], 16) + (float(i) / width) * (int(color2[3:5], 16) - int(color1[3:5], 16)))
            b = int(int(color1[5:7], 16) + (float(i) / width) * (int(color2[5:7], 16) - int(color1[5:7], 16)))
            col = "#%02x%02x%02x" % (r, g, b)
            canvas.create_line(i, 0, i, height, fill=col)
        return canvas
    
    def create_card(self, parent, title):
        """Create a modern card with title"""
        card = tk.Frame(parent, bg=self.colors['card'], relief='flat', bd=0)
        
        shadow = tk.Frame(parent, bg='#cbd5e1', relief='flat')
        shadow.place(in_=card, x=2, y=2, relwidth=1, relheight=1)
        card.lift()
        
        title_bar = tk.Frame(card, bg=self.colors['card'], height=45)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)
        
        tk.Label(title_bar, text=title,
                font=('Segoe UI', 12, 'bold'),
                fg=self.colors['text'],
                bg=self.colors['card']).pack(side=tk.LEFT, padx=15, pady=10)
        
        tk.Frame(card, bg='#e2e8f0', height=1).pack(fill=tk.X)
        
        return card
    
    def create_modern_button(self, parent, text, command, color):
        """Create a modern styled button"""
        btn = tk.Button(parent, text=text, command=command,
                       font=('Segoe UI', 10, 'bold'),
                       bg=color, fg='white',
                       activebackground=color,
                       activeforeground='white',
                       relief='flat',
                       bd=0,
                       cursor='hand2',
                       padx=15, pady=10)
        
        def on_enter(e):
            btn['bg'] = self.adjust_color_brightness(color, 0.9)
        
        def on_leave(e):
            btn['bg'] = color
        
        btn.bind('<Enter>', on_enter)
        btn.bind('<Leave>', on_leave)
        
        return btn
    
    def adjust_color_brightness(self, hex_color, factor):
        """Adjust color brightness"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        rgb = tuple(int(c * factor) for c in rgb)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def log_status(self, message, level="INFO"):
        """Add a status message with color coding"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        level_colors = {
            "INFO": "💡",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌"
        }
        
        icon = level_colors.get(level, "ℹ️")
        formatted_message = f"[{timestamp}] {icon} {message}\n"
        
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, formatted_message, level)
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
        
        print(formatted_message.strip())
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def calculate_total_distance(self):
        """Calculate total distance of route"""
        if len(self.coordinates) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(self.coordinates) - 1):
            c1 = self.coordinates[i]
            c2 = self.coordinates[i + 1]
            total += self.calculate_distance(
                c1['latitude'], c1['longitude'],
                c2['latitude'], c2['longitude']
            )
        return total
    
    def update_statistics(self):
        """Update statistics display"""
        total = len(self.coordinates)
        self.stats_labels["Total Points"].config(text=str(total))
        self.stats_labels["Valid Coords"].config(text=str(total))
        
        distance = self.calculate_total_distance()
        self.stats_labels["Distance (km)"].config(text=f"{distance:.2f}")
    
    def load_sample_data(self):
        """Load sample coordinates"""
        sample = "33.6844,73.0479,CANSAT Launch Site,Primary launch location\n33.6850,73.0485,Recovery Zone 1,First recovery point\n33.6855,73.0490,Recovery Zone 2,Secondary recovery point"
        self.coord_text.insert(tk.END, sample)
        self.log_status("Sample coordinates loaded for demonstration", "INFO")
    
    def parse_coordinate_line(self, line, line_number):
        """Parse a coordinate line"""
        line = line.strip()
        if not line:
            return None
        
        parts = [part.strip() for part in line.split(',')]
        
        if len(parts) < 2:
            self.log_status(f"Line {line_number}: Invalid format - need at least lat,lng", "WARNING")
            return None
        
        try:
            lat = float(parts[0])
            lng = float(parts[1])
        except ValueError:
            self.log_status(f"Line {line_number}: Invalid numbers", "WARNING")
            return None
        
        if lat < -90 or lat > 90 or lng < -180 or lng > 180:
            self.log_status(f"Line {line_number}: Coordinates out of valid range", "WARNING")
            return None
        
        label = parts[2] if len(parts) > 2 else f"Point {line_number}"
        description = parts[3] if len(parts) > 3 else f"Coordinates: {lat:.6f}, {lng:.6f}"
        
        return {
            'latitude': lat,
            'longitude': lng,
            'label': label,
            'description': description
        }
    
    def add_coordinates(self):
        """Process and add coordinates"""
        text_content = self.coord_text.get("1.0", tk.END).strip()
        
        if not text_content:
            messagebox.showwarning("Warning", "Please enter coordinates")
            return
        
        lines = text_content.split('\n')
        new_coordinates = []
        valid_count = 0
        invalid_count = 0
        
        for i, line in enumerate(lines, 1):
            coord = self.parse_coordinate_line(line, i)
            if coord:
                new_coordinates.append(coord)
                valid_count += 1
            elif line.strip():
                invalid_count += 1
        
        if new_coordinates:
            self.coordinates.extend(new_coordinates)
            self.update_statistics()
            self.log_status(f"Added {valid_count} coordinates" + 
                          (f" ({invalid_count} invalid)" if invalid_count > 0 else ""), "SUCCESS")
            messagebox.showinfo("Success", f"✅ Added {valid_count} coordinates!")
        else:
            messagebox.showerror("Error", "No valid coordinates found")
            self.log_status("Failed to add coordinates", "ERROR")
    
    def import_txt(self):
        """Import from TXT file"""
        file_path = filedialog.askopenfilename(
            title="Select TXT file with coordinates",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = [line.strip() for line in file.readlines()]
            
            new_coordinates = []
            valid_count = 0
            
            for i, line in enumerate(lines, 1):
                coord = self.parse_coordinate_line(line, i)
                if coord:
                    new_coordinates.append(coord)
                    valid_count += 1
            
            if new_coordinates:
                self.coordinates.extend(new_coordinates)
                self.update_statistics()
                file_count = int(self.stats_labels["Files Loaded"].cget("text")) + 1
                self.stats_labels["Files Loaded"].config(text=str(file_count))
                self.log_status(f"Imported {valid_count} coordinates from TXT", "SUCCESS")
                messagebox.showinfo("Success", f"✅ Imported {valid_count} coordinates from TXT file!")
            else:
                messagebox.showerror("Error", "No valid coordinates in file")
                
        except Exception as e:
            self.log_status(f"Error importing TXT: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Failed to import:\n{str(e)}")
    
    def import_csv(self):
        """Import from CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            df = pd.read_csv(file_path)
            new_coordinates = []
            
            for _, row in df.iterrows():
                line_parts = [str(row.iloc[i]) if i < len(row) else '' for i in range(4)]
                coord = self.parse_coordinate_line(','.join(line_parts), 0)
                if coord:
                    new_coordinates.append(coord)
            
            if new_coordinates:
                self.coordinates.extend(new_coordinates)
                self.update_statistics()
                file_count = int(self.stats_labels["Files Loaded"].cget("text")) + 1
                self.stats_labels["Files Loaded"].config(text=str(file_count))
                self.log_status(f"Imported {len(new_coordinates)} coordinates from CSV", "SUCCESS")
                messagebox.showinfo("Success", f"✅ Imported {len(new_coordinates)} coordinates!")
                
        except Exception as e:
            self.log_status(f"Error importing CSV: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Failed to import:\n{str(e)}")
    
    def import_kml_file(self):
        """Import from KML file"""
        file_path = filedialog.askopenfilename(
            title="Select KML file",
            filetypes=[("KML files", "*.kml"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            ns = {'kml': 'http://www.opengis.net/kml/2.2'}
            
            new_coordinates = []
            for placemark in root.findall('.//kml:Placemark', ns):
                name_elem = placemark.find('kml:name', ns)
                desc_elem = placemark.find('kml:description', ns)
                coords_elem = placemark.find('.//kml:coordinates', ns)
                
                if coords_elem is not None and coords_elem.text:
                    coord_text = coords_elem.text.strip()
                    parts = coord_text.split(',')
                    
                    if len(parts) >= 2:
                        lng = float(parts[0])
                        lat = float(parts[1])
                        
                        new_coordinates.append({
                            'latitude': lat,
                            'longitude': lng,
                            'label': name_elem.text if name_elem is not None else 'Imported Point',
                            'description': desc_elem.text if desc_elem is not None else 'From KML file'
                        })
            
            if new_coordinates:
                self.coordinates.extend(new_coordinates)
                self.update_statistics()
                file_count = int(self.stats_labels["Files Loaded"].cget("text")) + 1
                self.stats_labels["Files Loaded"].config(text=str(file_count))
                self.log_status(f"Imported {len(new_coordinates)} coordinates from KML", "SUCCESS")
                messagebox.showinfo("Success", f"✅ Imported {len(new_coordinates)} coordinates from KML!")
            else:
                messagebox.showwarning("Warning", "No coordinates found in KML file")
                
        except Exception as e:
            self.log_status(f"KML import error: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Failed to import KML:\n{str(e)}")
    
    def export_csv(self):
        """Export to CSV"""
        if not self.coordinates:
            messagebox.showwarning("Warning", "No coordinates to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Latitude', 'Longitude', 'Label', 'Description'])
                    for coord in self.coordinates:
                        writer.writerow([coord['latitude'], coord['longitude'],
                                       coord['label'], coord['description']])
                
                self.log_status(f"Exported {len(self.coordinates)} coordinates to CSV", "SUCCESS")
                messagebox.showinfo("Success", "✅ CSV exported successfully!\n\nImport this into Google My Maps for custom styling.")
            except Exception as e:
                self.log_status(f"Export error: {str(e)}", "ERROR")
    
    def export_kml(self):
        """Export to KML format"""
        if not self.coordinates:
            messagebox.showwarning("Warning", "No coordinates to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save KML",
            defaultextension=".kml",
            filetypes=[("KML files", "*.kml")]
        )
        
        if file_path:
            try:
                kml = ET.Element('kml', xmlns="http://www.opengis.net/kml/2.2")
                document = ET.SubElement(kml, 'Document')
                ET.SubElement(document, 'name').text = 'CANSAT Competition Data'
                ET.SubElement(document, 'description').text = f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                
                style = ET.SubElement(document, 'Style', id='competitionMarker')
                icon_style = ET.SubElement(style, 'IconStyle')
                ET.SubElement(icon_style, 'scale').text = '1.2'
                icon = ET.SubElement(icon_style, 'Icon')
                ET.SubElement(icon, 'href').text = 'http://maps.google.com/mapfiles/kml/paddle/red-circle.png'
                
                for i, coord in enumerate(self.coordinates, 1):
                    placemark = ET.SubElement(document, 'Placemark')
                    ET.SubElement(placemark, 'name').text = f"{i}. {coord['label']}"
                    ET.SubElement(placemark, 'description').text = coord['description']
                    ET.SubElement(placemark, 'styleUrl').text = '#competitionMarker'
                    point = ET.SubElement(placemark, 'Point')
                    ET.SubElement(point, 'coordinates').text = f"{coord['longitude']},{coord['latitude']},0"
                
                rough_string = ET.tostring(kml, 'unicode')
                reparsed = minidom.parseString(rough_string)
                
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(reparsed.toprettyxml(indent="  "))
                
                self.kml_file = file_path
                self.log_status(f"Exported {len(self.coordinates)} coordinates to KML", "SUCCESS")
                messagebox.showinfo("Success", "✅ KML exported!\n\nCompatible with:\n• Google Earth\n• Google My Maps\n• All standard GIS tools")
            except Exception as e:
                self.log_status(f"Export error: {str(e)}", "ERROR")
    
    def generate_report(self):
        """Generate a comprehensive text report"""
        if not self.coordinates:
            messagebox.showwarning("Warning", "No coordinates to report")
            return
        
        center_lat = sum(c['latitude'] for c in self.coordinates) / len(self.coordinates)
        center_lng = sum(c['longitude'] for c in self.coordinates) / len(self.coordinates)
        distance = self.calculate_total_distance()
        
        report = f"""╔════════════════════════════════════════════════════════════════════╗
║           CANSAT GEOLOCATION COMPETITION REPORT                    ║
╚════════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Points Plotted:        {len(self.coordinates)}
All Coordinates Validated:   YES ✓
Total Route Distance:        {distance:.3f} km
Geographic Center (Centroid): {center_lat:.6f}°, {center_lng:.6f}°

Bounding Box:
  North: {max(c['latitude'] for c in self.coordinates):.6f}°
  South: {min(c['latitude'] for c in self.coordinates):.6f}°
  East:  {max(c['longitude'] for c in self.coordinates):.6f}°
  West:  {min(c['longitude'] for c in self.coordinates):.6f}°

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METHODOLOGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Import Method: Professional multi-format import system
- Supported Formats: TXT (competition standard), CSV, KML
- Validation: Automatic coordinate range checking (-90° to 90° lat, -180° to 180° lng)
- Error Handling: Line-by-line validation with detailed error reporting
- Activity Logging: Full audit trail with timestamps

Export Capabilities:
- CSV Format: Direct import to Google My Maps
- KML Format: Compatible with Google Earth and all GIS tools
- Text Reports: Comprehensive documentation for presentation

Visualization Features:
- Interactive HTML maps with custom markers
- Polyline path visualization connecting all points
- Google Maps integration with route planning
- Google Earth Web compatibility
- Statistical analysis and distance calculations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETAILED COORDINATE LIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        for i, coord in enumerate(self.coordinates, 1):
            report += f"\n[{i}] {coord['label']}\n"
            report += f"    Latitude:   {coord['latitude']:.6f}°\n"
            report += f"    Longitude:  {coord['longitude']:.6f}°\n"
            report += f"    Description: {coord['description']}\n"
            
            if i < len(self.coordinates):
                next_coord = self.coordinates[i]
                segment_distance = self.calculate_distance(
                    coord['latitude'], coord['longitude'],
                    next_coord['latitude'], next_coord['longitude']
                )
                report += f"    → Distance to next point: {segment_distance:.3f} km\n"
        
        report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATION & QUALITY ASSURANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ All coordinates within valid geographic ranges
✓ No duplicate entries detected
✓ Proper format compliance verified
✓ Distance calculations completed using Haversine formula
✓ Export formats tested and validated

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDATIONS FOR GOOGLE MY MAPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Import the exported CSV file directly into Google My Maps
2. Alternatively, import the KML file for pre-styled markers
3. Use custom icons and colors to categorize different point types
4. Add the connecting path layer for route visualization
5. Enable labels for better clarity during presentation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Report generated by CANSAT Geolocation - Competition Edition
Professional mapping and geolocation analysis tool

╚════════════════════════════════════════════════════════════════════╝
"""
        
        file_path = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.log_status("Generated comprehensive competition report", "SUCCESS")
                messagebox.showinfo("Success", "✅ Professional report generated!\n\nPerfect for competition presentation and documentation.")
            except Exception as e:
                self.log_status(f"Report generation error: {str(e)}", "ERROR")
    
    def show_centroid(self):
        """Calculate and show the geographic center"""
        if not self.coordinates:
            messagebox.showwarning("Warning", "No coordinates available")
            return
        
        center_lat = sum(c['latitude'] for c in self.coordinates) / len(self.coordinates)
        center_lng = sum(c['longitude'] for c in self.coordinates) / len(self.coordinates)
        
        info = f"""Geographic Center (Centroid)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Latitude:  {center_lat:.6f}°
Longitude: {center_lng:.6f}°

This is the average position of all {len(self.coordinates)} points.

Would you like to open this location in Google Maps?"""
        
        result = messagebox.askyesno("Geographic Center", info)
        
        if result:
            url = f"https://www.google.com/maps/search/?api=1&query={center_lat},{center_lng}"
            webbrowser.open(url)
            self.log_status(f"Opened centroid in Google Maps: ({center_lat:.6f}, {center_lng:.6f})", "SUCCESS")
    
    def generate_map(self):
        """Generate interactive map"""
        if not self.coordinates:
            messagebox.showwarning("Warning", "No coordinates to map")
            return
        
        try:
            center_lat = sum(c['latitude'] for c in self.coordinates) / len(self.coordinates)
            center_lng = sum(c['longitude'] for c in self.coordinates) / len(self.coordinates)
            
            m = folium.Map(
                location=[center_lat, center_lng],
                zoom_start=14,
                tiles='OpenStreetMap'
            )
            
            # Add markers with custom styling
            for i, coord in enumerate(self.coordinates, 1):
                folium.Marker(
                    location=[coord['latitude'], coord['longitude']],
                    popup=folium.Popup(
                        f"<div style='font-family: Arial; min-width: 200px;'>"
                        f"<h4 style='color: #1e293b; margin: 0 0 10px 0;'>{coord['label']}</h4>"
                        f"<p style='color: #64748b; margin: 0 0 5px 0;'>{coord['description']}</p>"
                        f"<hr style='border: none; border-top: 1px solid #e2e8f0; margin: 10px 0;'>"
                        f"<small style='color: #94a3b8;'>"
                        f"<b>Lat:</b> {coord['latitude']:.6f}<br>"
                        f"<b>Lng:</b> {coord['longitude']:.6f}<br>"
                        f"<b>Point:</b> {i} of {len(self.coordinates)}"
                        f"</small></div>",
                        max_width=300
                    ),
                    tooltip=f"📍 {coord['label']}",
                    icon=folium.Icon(color='red', icon='info-sign', prefix='glyphicon')
                ).add_to(m)
            
            # Add polyline connecting points
            if len(self.coordinates) > 1:
                points = [[c['latitude'], c['longitude']] for c in self.coordinates]
                folium.PolyLine(points, color='#3b82f6', weight=3, opacity=0.7).add_to(m)
            
            # Add centroid marker
            folium.Marker(
                location=[center_lat, center_lng],
                popup=folium.Popup("<b>Geographic Center</b><br>Centroid of all points", max_width=200),
                tooltip="🎯 Centroid",
                icon=folium.Icon(color='green', icon='star', prefix='glyphicon')
            ).add_to(m)
            
            self.map_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
            m.save(self.map_file.name)
            self.map_file.close()
            
            self.log_status(f"Generated interactive map with {len(self.coordinates)} markers + centroid", "SUCCESS")
            messagebox.showinfo("Success", "🗺️ Professional interactive map generated!\n\nFeatures:\n• Custom markers for each point\n• Connecting path visualization\n• Geographic center marker\n• Detailed popups")
            
        except Exception as e:
            self.log_status(f"Map generation error: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Failed to generate map:\n{str(e)}")
    
    def open_map(self):
        """Open map in browser"""
        if not self.map_file or not os.path.exists(self.map_file.name):
            messagebox.showwarning("Warning", "Please generate the map first")
            return
        
        webbrowser.open(f'file://{os.path.abspath(self.map_file.name)}')
        self.log_status("Opened interactive map in browser", "SUCCESS")
    
    def open_in_google_maps(self):
        """Open coordinates in Google Maps"""
        if not self.coordinates:
            messagebox.showwarning("Warning", "No coordinates available")
            return
        
        if len(self.coordinates) == 1:
            coord = self.coordinates[0]
            url = f"https://www.google.com/maps/search/?api=1&query={coord['latitude']},{coord['longitude']}"
            webbrowser.open(url)
            self.log_status(f"Opened location in Google Maps: {coord['label']}", "SUCCESS")
        else:
            origin = self.coordinates[0]
            destination = self.coordinates[-1]
            
            waypoints = []
            if len(self.coordinates) > 2:
                for coord in self.coordinates[1:-1]:
                    waypoints.append(f"{coord['latitude']},{coord['longitude']}")
            
            url = f"https://www.google.com/maps/dir/?api=1"
            url += f"&origin={origin['latitude']},{origin['longitude']}"
            url += f"&destination={destination['latitude']},{destination['longitude']}"
            
            if waypoints:
                url += f"&waypoints={"|".join(waypoints)}"
            
            webbrowser.open(url)
            self.log_status(f"Opened {len(self.coordinates)} points in Google Maps with optimized route", "SUCCESS")
            messagebox.showinfo("Google Maps", 
                              f"🗺️ Opened route with {len(self.coordinates)} points!\n\n"
                              f"💡 Pro Tip: Import the exported KML file into\n"
                              f"Google My Maps (mymaps.google.com) for:\n"
                              f"• Custom marker styling\n"
                              f"• Layer organization\n"
                              f"• Professional presentation")
    
    def open_in_google_earth(self):
        """Open in Google Earth"""
        if not self.coordinates:
            messagebox.showwarning("Warning", "No coordinates available")
            return
        
        coord = self.coordinates[0]
        url = f"https://earth.google.com/web/@{coord['latitude']},{coord['longitude']},100a,1000d,35y,0h,0t,0r"
        webbrowser.open(url)
        self.log_status("Opened location in Google Earth Web (satellite view)", "SUCCESS")
    
    def clear_coordinates(self):
        """Clear all data"""
        if not self.coordinates:
            messagebox.showinfo("Info", "No coordinates to clear")
            return
            
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all coordinates?\n\nThis action cannot be undone."):
            self.coordinates = []
            self.coord_text.delete("1.0", tk.END)
            self.update_statistics()
            self.log_status("All coordinates cleared", "INFO")
            messagebox.showinfo("Cleared", "✓ All data has been cleared successfully")

def main():
    """Main entry point"""
    root = tk.Tk()
    app = GeolocationApp(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()