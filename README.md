# CANSAT-PROJECT
A CanSat project involving embedded systems, sensors, integration, data collection and telementry for a student aerospace engineering project.
# CanSat Mission Control and Telemetry System 🚀

## Overview

This project contains a collection of Python-based tools developed for CanSat mission monitoring, flight data analysis, trajectory reconstruction, and geolocation.

The software provides graphical interfaces for visualizing telemetry data, analyzing sensor measurements, reconstructing movement, and displaying geographic coordinates.

## Project Features

* **Mission Control:** 3D trajectory visualization, orientation tracking, sensor data processing, and simulated or serial-based data ingestion.
* **Ground Station GUI:** Telemetry dashboard with flight parameters, status indicators, alerts, and graphical monitoring.
* **Geolocation Tool:** Coordinate mapping, route-distance calculations, interactive maps, and TXT, CSV, and KML import/export.
* **Flight Telemetry Dashboard:** Flight-phase analysis, synchronized plots, data-quality assessment, and report and plot exports.
* **3D Trajectory Reconstruction:** Sensor calibration, sensor fusion, motion analysis, drift correction, and trajectory validation.

## Technologies Used

* Python
* NumPy
* Matplotlib
* Tkinter and CustomTkinter
* PyQt6
* Pandas
* SciPy
* Folium

## Repository Structure

| File                    | Purpose                                         |
| ----------------------- | ----------------------------------------------- |
| `abeera(1).py`          | Mission control and 3D trajectory visualization |
| `ground_station_gui.py` | Ground station telemetry dashboard              |
| `zainab.py`             | CanSat geolocation and mapping tool             |
| `eight.py`              | 3D trajectory reconstruction and sensor fusion  |
| `hybrid_finale.py`      | Flight telemetry analysis dashboard             |

## Getting Started

1. Install Python 3.8 or later.
2. Install the required Python libraries for the application you want to run.
3. Run the corresponding Python script from your development environment.

Example:

```bash
python ground_station_gui.py
```

Each application may require additional libraries and a compatible desktop environment.

## Data and Testing

The tools support different aspects of telemetry visualization, coordinate mapping, and flight-data analysis. Some functionality uses simulated data, while hardware-dependent operation requires compatible data sources and configuration.

## Project Status

This repository organizes the software tools developed for the CanSat project. Further integration, testing, and documentation can be added as the project evolves.
