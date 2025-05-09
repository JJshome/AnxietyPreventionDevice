#!/usr/bin/env python3
"""
API Server for Anxiety Prevention Device

This module provides a Flask-based web server for monitoring and controlling 
the anxiety prevention system, as implemented in the patents:
- 10-2022-0007209 ("불안장애 예방장치")
- 10-2459338 ("저주파 자극기 제어장치")

The server provides:
- RESTful API endpoints for controlling stimulation devices
- WebSocket connection for real-time monitoring
- Web interface for visualization and interaction
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Union, Optional, Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import pymongo
from pymongo import MongoClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'anxiety_prevention_secret_key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Get MongoDB connection parameters from environment variables
MONGO_HOST = os.environ.get('MONGO_HOST', 'localhost')
MONGO_PORT = int(os.environ.get('MONGO_PORT', 27017))

# Connect to MongoDB
try:
    mongo_client = MongoClient(f'mongodb://{MONGO_HOST}:{MONGO_PORT}/')
    db = mongo_client['anxiety_prevention_db']
    logger.info(f'Connected to MongoDB at {MONGO_HOST}:{MONGO_PORT}')
except Exception as e:
    logger.error(f'Failed to connect to MongoDB: {str(e)}')
    db = None

# Data collections
if db:
    ecg_collection = db['ecg_data']
    hrv_collection = db['hrv_data']
    anxiety_collection = db['anxiety_data']
    stimulation_collection = db['stimulation_data']
    device_collection = db['device_data']

# In-memory data storage (fallback if MongoDB is not available)
in_memory_data = {
    'ecg_data': [],
    'hrv_data': [],
    'anxiety_data': [],
    'stimulation_data': [],
    'device_data': {}
}

# State variables
connected_clients = set()
connected_devices = {}
is_simulation_running = False

@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get the current status of the system."""
    status = {
        'connected_clients': len(connected_clients),
        'connected_devices': connected_devices,
        'is_simulation_running': is_simulation_running,
        'server_time': datetime.now().isoformat()
    }
    return jsonify(status)

@app.route('/api/ecg/recent', methods=['GET'])
def get_recent_ecg():
    """Get recent ECG data."""
    limit = int(request.args.get('limit', 1000))
    
    if db:
        # Get data from MongoDB
        ecg_documents = list(ecg_collection.find().sort('timestamp', -1).limit(limit))
        ecg_data = [doc['data'] for doc in ecg_documents]
        timestamps = [doc['timestamp'] for doc in ecg_documents]
    else:
        # Get data from in-memory storage
        ecg_data = in_memory_data['ecg_data'][-limit:]
        timestamps = [time.time() - (len(ecg_data) - i) * 0.01 for i in range(len(ecg_data))]
    
    return jsonify({
        'ecg_data': ecg_data,
        'timestamps': timestamps
    })

@app.route('/api/hrv/recent', methods=['GET'])
def get_recent_hrv():
    """Get recent HRV analysis data."""
    limit = int(request.args.get('limit', 100))
    
    if db:
        # Get data from MongoDB
        hrv_documents = list(hrv_collection.find().sort('timestamp', -1).limit(limit))
        hrv_data = [doc['data'] for doc in hrv_documents]
        timestamps = [doc['timestamp'] for doc in hrv_documents]
    else:
        # Get data from in-memory storage
        hrv_data = in_memory_data['hrv_data'][-limit:]
        timestamps = [time.time() - (len(hrv_data) - i) * 1.0 for i in range(len(hrv_data))]
    
    return jsonify({
        'hrv_data': hrv_data,
        'timestamps': timestamps
    })

@app.route('/api/anxiety/recent', methods=['GET'])
def get_recent_anxiety():
    """Get recent anxiety prediction data."""
    limit = int(request.args.get('limit', 100))
    
    if db:
        # Get data from MongoDB
        anxiety_documents = list(anxiety_collection.find().sort('timestamp', -1).limit(limit))
        anxiety_data = [doc['data'] for doc in anxiety_documents]
        timestamps = [doc['timestamp'] for doc in anxiety_documents]
    else:
        # Get data from in-memory storage
        anxiety_data = in_memory_data['anxiety_data'][-limit:]
        timestamps = [time.time() - (len(anxiety_data) - i) * 5.0 for i in range(len(anxiety_data))]
    
    return jsonify({
        'anxiety_data': anxiety_data,
        'timestamps': timestamps
    })

@app.route('/api/stimulation/status', methods=['GET'])
def get_stimulation_status():
    """Get the current stimulation status."""
    if db:
        # Get latest stimulation data from MongoDB
        latest_stim = stimulation_collection.find_one(sort=[('timestamp', -1)])
        stim_status = latest_stim['data'] if latest_stim else {'is_stimulating': False, 'session': None}
    else:
        # Get latest stimulation data from in-memory storage
        stim_data = in_memory_data['stimulation_data']
        stim_status = stim_data[-1] if stim_data else {'is_stimulating': False, 'session': None}
    
    return jsonify(stim_status)

@app.route('/api/devices', methods=['GET'])
def get_devices():
    """Get the list of connected devices."""
    return jsonify({'devices': connected_devices})

@app.route('/api/stimulation/start', methods=['POST'])
def start_stimulation():
    """Start stimulation with the specified parameters."""
    if not request.json:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    # Extract parameters from request
    anxiety_level = request.json.get('anxiety_level', 1)
    session_duration = request.json.get('session_duration', 20)
    waveform = request.json.get('waveform', 'sine')
    base_frequency = request.json.get('base_frequency', 0.5)
    phase_difference = request.json.get('phase_difference', 0.5)
    device_ids = request.json.get('device_ids', [])
    
    # Validate parameters
    if not isinstance(anxiety_level, int) or anxiety_level < 0 or anxiety_level > 3:
        return jsonify({'error': 'Invalid anxiety_level (must be 0-3)'}), 400
    
    if not isinstance(session_duration, int) or session_duration < 1 or session_duration > 60:
        return jsonify({'error': 'Invalid session_duration (must be 1-60 minutes)'}), 400
    
    if waveform not in ['sine', 'square', 'triangular', 'monophasic', 'biphasic']:
        return jsonify({'error': 'Invalid waveform'}), 400
    
    if not isinstance(base_frequency, (int, float)) or base_frequency < 0.1 or base_frequency > 10.0:
        return jsonify({'error': 'Invalid base_frequency (must be 0.1-10.0 Hz)'}), 400
    
    if not isinstance(phase_difference, (int, float)) or phase_difference < 0.0 or phase_difference > 5.0:
        return jsonify({'error': 'Invalid phase_difference (must be 0.0-5.0 seconds)'}), 400
    
    if not device_ids:
        return jsonify({'error': 'No devices specified'}), 400
    
    # Emit start stimulation event to connected clients
    stimulation_params = {
        'anxiety_level': anxiety_level,
        'session_duration': session_duration,
        'waveform': waveform,
        'base_frequency': base_frequency,
        'phase_difference': phase_difference,
        'device_ids': device_ids
    }
    
    socketio.emit('start_stimulation', stimulation_params)
    
    # Store stimulation data
    stim_status = {
        'is_stimulating': True,
        'session': {
            'start_time': time.time(),
            'duration': session_duration * 60,
            'waveform': waveform,
            'anxiety_level': anxiety_level,
            'devices': device_ids,
            'frequency': base_frequency,
            'phase_difference': phase_difference
        }
    }
    
    if db:
        stimulation_collection.insert_one({
            'timestamp': time.time(),
            'data': stim_status
        })
    else:
        in_memory_data['stimulation_data'].append(stim_status)
    
    return jsonify({'success': True, 'message': 'Stimulation started successfully'})

@app.route('/api/stimulation/stop', methods=['POST'])
def stop_stimulation():
    """Stop any active stimulation."""
    # Emit stop stimulation event to connected clients
    socketio.emit('stop_stimulation')
    
    # Update stimulation status
    stim_status = {
        'is_stimulating': False,
        'session': None
    }
    
    if db:
        stimulation_collection.insert_one({
            'timestamp': time.time(),
            'data': stim_status
        })
    else:
        in_memory_data['stimulation_data'].append(stim_status)
    
    return jsonify({'success': True, 'message': 'Stimulation stopped successfully'})

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    client_id = request.sid
    connected_clients.add(client_id)
    logger.info(f'Client connected: {client_id}')
    emit('connected', {'client_id': client_id})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    client_id = request.sid
    if client_id in connected_clients:
        connected_clients.remove(client_id)
    logger.info(f'Client disconnected: {client_id}')

@socketio.on('register_device')
def handle_register_device(data):
    """Register a device with the server."""
    device_id = data.get('device_id')
    device_type = data.get('device_type')
    device_name = data.get('device_name', 'Unknown Device')
    
    if not device_id:
        emit('error', {'message': 'No device_id provided'})
        return
    
    connected_devices[device_id] = {
        'device_id': device_id,
        'device_type': device_type,
        'device_name': device_name,
        'client_id': request.sid,
        'connected_at': time.time(),
        'status': 'connected'
    }
    
    if db:
        device_collection.update_one(
            {'device_id': device_id},
            {'$set': connected_devices[device_id]},
            upsert=True
        )
    else:
        in_memory_data['device_data'][device_id] = connected_devices[device_id]
    
    logger.info(f'Device registered: {device_id} ({device_name})')
    emit('device_registered', {'device_id': device_id})
    socketio.emit('device_update', {'devices': connected_devices})

@socketio.on('ecg_data')
def handle_ecg_data(data):
    """Handle incoming ECG data."""
    device_id = data.get('device_id')
    ecg_signal = data.get('ecg_signal', [])
    timestamp = data.get('timestamp', time.time())
    
    if not device_id or not ecg_signal:
        return
    
    # Store ECG data
    if db:
        ecg_collection.insert_one({
            'device_id': device_id,
            'timestamp': timestamp,
            'data': ecg_signal
        })
    else:
        in_memory_data['ecg_data'].extend(ecg_signal)
        if len(in_memory_data['ecg_data']) > 10000:
            in_memory_data['ecg_data'] = in_memory_data['ecg_data'][-10000:]
    
    # Broadcast to connected clients
    socketio.emit('ecg_update', {
        'device_id': device_id,
        'ecg_signal': ecg_signal,
        'timestamp': timestamp
    })

@socketio.on('hrv_data')
def handle_hrv_data(data):
    """Handle incoming HRV analysis data."""
    device_id = data.get('device_id')
    hrv_params = data.get('hrv_params', {})
    timestamp = data.get('timestamp', time.time())
    
    if not device_id or not hrv_params:
        return
    
    # Store HRV data
    if db:
        hrv_collection.insert_one({
            'device_id': device_id,
            'timestamp': timestamp,
            'data': hrv_params
        })
    else:
        in_memory_data['hrv_data'].append(hrv_params)
        if len(in_memory_data['hrv_data']) > 1000:
            in_memory_data['hrv_data'] = in_memory_data['hrv_data'][-1000:]
    
    # Broadcast to connected clients
    socketio.emit('hrv_update', {
        'device_id': device_id,
        'hrv_params': hrv_params,
        'timestamp': timestamp
    })

@socketio.on('anxiety_data')
def handle_anxiety_data(data):
    """Handle incoming anxiety prediction data."""
    device_id = data.get('device_id')
    prediction = data.get('prediction', {})
    timestamp = data.get('timestamp', time.time())
    
    if not device_id or not prediction:
        return
    
    # Store anxiety data
    if db:
        anxiety_collection.insert_one({
            'device_id': device_id,
            'timestamp': timestamp,
            'data': prediction
        })
    else:
        in_memory_data['anxiety_data'].append(prediction)
        if len(in_memory_data['anxiety_data']) > 1000:
            in_memory_data['anxiety_data'] = in_memory_data['anxiety_data'][-1000:]
    
    # Broadcast to connected clients
    socketio.emit('anxiety_update', {
        'device_id': device_id,
        'prediction': prediction,
        'timestamp': timestamp
    })

@socketio.on('stimulation_status')
def handle_stimulation_status(data):
    """Handle incoming stimulation status updates."""
    device_id = data.get('device_id')
    status = data.get('status', {})
    timestamp = data.get('timestamp', time.time())
    
    if not device_id or not status:
        return
    
    # Store stimulation data
    if db:
        stimulation_collection.insert_one({
            'device_id': device_id,
            'timestamp': timestamp,
            'data': status
        })
    else:
        in_memory_data['stimulation_data'].append(status)
        if len(in_memory_data['stimulation_data']) > 1000:
            in_memory_data['stimulation_data'] = in_memory_data['stimulation_data'][-1000:]
    
    # Broadcast to connected clients
    socketio.emit('stimulation_update', {
        'device_id': device_id,
        'status': status,
        'timestamp': timestamp
    })

@socketio.on('device_status')
def handle_device_status(data):
    """Handle device status updates."""
    device_id = data.get('device_id')
    status = data.get('status', 'unknown')
    battery_level = data.get('battery_level', 0)
    
    if not device_id:
        return
    
    if device_id in connected_devices:
        connected_devices[device_id]['status'] = status
        connected_devices[device_id]['battery_level'] = battery_level
        connected_devices[device_id]['last_update'] = time.time()
        
        if db:
            device_collection.update_one(
                {'device_id': device_id},
                {'$set': {
                    'status': status,
                    'battery_level': battery_level,
                    'last_update': time.time()
                }}
            )
        else:
            if device_id in in_memory_data['device_data']:
                in_memory_data['device_data'][device_id]['status'] = status
                in_memory_data['device_data'][device_id]['battery_level'] = battery_level
                in_memory_data['device_data'][device_id]['last_update'] = time.time()
        
        # Broadcast device update
        socketio.emit('device_update', {'devices': connected_devices})

def create_templates_folder():
    """Create templates folder and add index.html."""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    index_html_path = os.path.join(templates_dir, 'index.html')
    if not os.path.exists(index_html_path):
        with open(index_html_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anxiety Prevention Device - Dashboard</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.0/dist/css/bootstrap.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.5.0/dist/chart.min.js"></script>
    <script src="https://cdn.socket.io/4.1.2/socket.io.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
        }
        .header {
            background-color: #343a40;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .container {
            margin-top: 20px;
        }
        .card {
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .card-header {
            background-color: #343a40;
            color: white;
            border-radius: 10px 10px 0 0 !important;
        }
        .status-indicator {
            display: inline-block;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .status-connected {
            background-color: #28a745;
        }
        .status-disconnected {
            background-color: #dc3545;
        }
        .status-warning {
            background-color: #ffc107;
        }
        .chart-container {
            position: relative;
            height: 300px;
            width: 100%;
        }
        .device-card {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
        }
        .anxiety-level {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
        }
        .anxiety-level-0 {
            color: #28a745;
        }
        .anxiety-level-1 {
            color: #ffc107;
        }
        .anxiety-level-2 {
            color: #fd7e14;
        }
        .anxiety-level-3 {
            color: #dc3545;
        }
        .stimulation-controls {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Anxiety Prevention Device</h1>
        <p>Real-time Monitoring Dashboard</p>
    </div>

    <div class="container">
        <div class="row">
            <!-- System Status -->
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        System Status
                    </div>
                    <div class="card-body">
                        <p>
                            <span class="status-indicator" id="system-status-indicator"></span>
                            <span id="system-status-text">Connecting...</span>
                        </p>
                        <p>Connected Clients: <span id="connected-clients">0</span></p>
                        <p>Simulation Running: <span id="simulation-status">No</span></p>
                        <p>Server Time: <span id="server-time">-</span></p>
                    </div>
                </div>

                <!-- Connected Devices -->
                <div class="card">
                    <div class="card-header">
                        Connected Devices
                    </div>
                    <div class="card-body">
                        <div id="devices-container">
                            <p>No devices connected</p>
                        </div>
                    </div>
                </div>

                <!-- Current Anxiety Status -->
                <div class="card">
                    <div class="card-header">
                        Current Anxiety Status
                    </div>
                    <div class="card-body">
                        <div class="anxiety-level anxiety-level-0" id="anxiety-level">0</div>
                        <p class="text-center" id="anxiety-category">Low</p>
                        <div class="progress">
                            <div id="anxiety-progress" class="progress-bar bg-success" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
                        </div>
                        <p class="mt-2">Confidence: <span id="anxiety-confidence">0.00</span></p>
                    </div>
                </div>

                <!-- Stimulation Controls -->
                <div class="card">
                    <div class="card-header">
                        Stimulation Controls
                    </div>
                    <div class="card-body">
                        <div id="stimulation-status-container">
                            <p>Status: <span id="stimulation-status-text">Not Active</span></p>
                            <div id="stimulation-active-container" style="display: none;">
                                <p>Duration: <span id="stimulation-duration">0:00</span></p>
                                <p>Remaining: <span id="stimulation-remaining">0:00</span></p>
                                <div class="progress">
                                    <div id="stimulation-progress" class="progress-bar" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
                                </div>
                                <button id="stop-stimulation-btn" class="btn btn-danger mt-3">Stop Stimulation</button>
                            </div>
                        </div>
                        <div id="stimulation-control-container">
                            <h5 class="mt-3">Manual Stimulation</h5>
                            <form id="start-stimulation-form">
                                <div class="mb-3">
                                    <label for="anxiety-level-select" class="form-label">Anxiety Level</label>
                                    <select class="form-select" id="anxiety-level-select">
                                        <option value="0">Low (0)</option>
                                        <option value="1">Moderate (1)</option>
                                        <option value="2">High (2)</option>
                                        <option value="3">Very High (3)</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label for="duration-input" class="form-label">Duration (minutes)</label>
                                    <input type="number" class="form-control" id="duration-input" min="1" max="60" value="20">
                                </div>
                                <div class="mb-3">
                                    <label for="waveform-select" class="form-label">Waveform</label>
                                    <select class="form-select" id="waveform-select">
                                        <option value="sine">Sine</option>
                                        <option value="square">Square</option>
                                        <option value="triangular">Triangular</option>
                                        <option value="monophasic">Monophasic</option>
                                        <option value="biphasic">Biphasic</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label for="frequency-input" class="form-label">Frequency (Hz)</label>
                                    <input type="number" class="form-control" id="frequency-input" min="0.1" max="10" step="0.1" value="0.5">
                                </div>
                                <div class="mb-3">
                                    <label for="phase-diff-input" class="form-label">Phase Difference (s)</label>
                                    <input type="number" class="form-control" id="phase-diff-input" min="0" max="5" step="0.1" value="0.5">
                                </div>
                                <div id="device-checkboxes" class="mb-3">
                                    <!-- Device checkboxes will be added here dynamically -->
                                </div>
                                <button type="submit" class="btn btn-primary">Start Stimulation</button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Charts -->
            <div class="col-md-8">
                <!-- ECG Signal -->
                <div class="card">
                    <div class="card-header">
                        ECG Signal
                    </div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="ecg-chart"></canvas>
                        </div>
                    </div>
                </div>

                <!-- HRV Analysis -->
                <div class="card">
                    <div class="card-header">
                        HRV Analysis
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <canvas id="rmssd-chart"></canvas>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <canvas id="lf-hf-chart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Anxiety Prediction -->
                <div class="card">
                    <div class="card-header">
                        Anxiety Prediction
                    </div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="anxiety-chart"></canvas>
                        </div>
                    </div>
                </div>

                <!-- Stimulation Visualization -->
                <div class="card">
                    <div class="card-header">
                        Stimulation Visualization
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <canvas id="stim1-chart"></canvas>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <canvas id="stim2-chart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Socket.IO connection
        const socket = io();
        
        // Charts
        let ecgChart, rmssdChart, lfHfChart, anxietyChart, stim1Chart, stim2Chart;
        
        // Data storage
        let ecgData = [];
        let hrvData = [];
        let anxietyData = [];
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
            initializeSocketListeners();
            initializeUIControls();
            
            // Fetch initial data
            fetchSystemStatus();
            fetchDevices();
            fetchRecentECG();
            fetchRecentHRV();
            fetchRecentAnxiety();
            fetchStimulationStatus();
            
            // Set up periodic updates
            setInterval(fetchSystemStatus, 5000);
            setInterval(fetchStimulationStatus, 2000);
        });
        
        // Initialize charts
        function initializeCharts() {
            // ECG Chart
            const ecgCtx = document.getElementById('ecg-chart').getContext('2d');
            ecgChart = new Chart(ecgCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'ECG Signal',
                        data: [],
                        borderColor: 'rgba(40, 167, 69, 1)',
                        borderWidth: 1,
                        pointRadius: 0,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Time (s)'
                            }
                        },
                        y: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Amplitude (mV)'
                            }
                        }
                    },
                    animation: false
                }
            });
            
            // RMSSD Chart
            const rmssdCtx = document.getElementById('rmssd-chart').getContext('2d');
            rmssdChart = new Chart(rmssdCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'RMSSD',
                        data: [],
                        borderColor: 'rgba(0, 123, 255, 1)',
                        borderWidth: 1,
                        pointRadius: 2,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Time (min)'
                            }
                        },
                        y: {
                            display: true,
                            title: {
                                display: true,
                                text: 'RMSSD (ms)'
                            }
                        }
                    }
                }
            });
            
            // LF/HF Chart
            const lfHfCtx = document.getElementById('lf-hf-chart').getContext('2d');
            lfHfChart = new Chart(lfHfCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'LF/HF Ratio',
                        data: [],
                        borderColor: 'rgba(220, 53, 69, 1)',
                        borderWidth: 1,
                        pointRadius: 2,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Time (min)'
                            }
                        },
                        y: {
                            display: true,
                            title: {
                                display: true,
                                text: 'LF/HF Ratio'
                            }
                        }
                    }
                }
            });
            
            // Anxiety Chart
            const anxietyCtx = document.getElementById('anxiety-chart').getContext('2d');
            anxietyChart = new Chart(anxietyCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Anxiety Level',
                        data: [],
                        borderColor: 'rgba(108, 117, 125, 1)',
                        backgroundColor: 'rgba(108, 117, 125, 0.5)',
                        borderWidth: 2,
                        pointRadius: 3,
                        pointBackgroundColor: 'rgba(220, 53, 69, 1)',
                        fill: false,
                        stepped: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Time (min)'
                            }
                        },
                        y: {
                            display: true,
                            min: -0.1,
                            max: 3.1,
                            ticks: {
                                stepSize: 1,
                                callback: function(value) {
                                    if (value === 0) return 'Low';
                                    if (value === 1) return 'Moderate';
                                    if (value === 2) return 'High';
                                    if (value === 3) return 'Very High';
                                    return '';
                                }
                            },
                            title: {
                                display: true,
                                text: 'Anxiety Level'
                            }
                        }
                    },
                    plugins: {
                        annotation: {
                            annotations: {
                                lowLine: {
                                    type: 'line',
                                    yMin: 0.5,
                                    yMax: 0.5,
                                    borderColor: 'rgba(40, 167, 69, 0.5)',
                                    borderWidth: 1,
                                    borderDash: [5, 5]
                                },
                                moderateLine: {
                                    type: 'line',
                                    yMin: 1.5,
                                    yMax: 1.5,
                                    borderColor: 'rgba(255, 193, 7, 0.5)',
                                    borderWidth: 1,
                                    borderDash: [5, 5]
                                },
                                highLine: {
                                    type: 'line',
                                    yMin: 2.5,
                                    yMax: 2.5,
                                    borderColor: 'rgba(220, 53, 69, 0.5)',
                                    borderWidth: 1,
                                    borderDash: [5, 5]
                                }
                            }
                        }
                    }
                }
            });
            
            // Stimulation Charts
            const stim1Ctx = document.getElementById('stim1-chart').getContext('2d');
            stim1Chart = new Chart(stim1Ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Left Stimulator',
                        data: [],
                        borderColor: 'rgba(0, 123, 255, 1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Time (s)'
                            }
                        },
                        y: {
                            display: true,
                            min: -1.1,
                            max: 1.1,
                            title: {
                                display: true,
                                text: 'Amplitude'
                            }
                        }
                    },
                    animation: false
                }
            });
            
            const stim2Ctx = document.getElementById('stim2-chart').getContext('2d');
            stim2Chart = new Chart(stim2Ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Right Stimulator',
                        data: [],
                        borderColor: 'rgba(220, 53, 69, 1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Time (s)'
                            }
                        },
                        y: {
                            display: true,
                            min: -1.1,
                            max: 1.1,
                            title: {
                                display: true,
                                text: 'Amplitude'
                            }
                        }
                    },
                    animation: false
                }
            });
        }
        
        // Initialize Socket.IO listeners
        function initializeSocketListeners() {
            socket.on('connect', function() {
                updateSystemStatus('Connected', 'connected');
            });
            
            socket.on('disconnect', function() {
                updateSystemStatus('Disconnected', 'disconnected');
            });
            
            socket.on('ecg_update', function(data) {
                updateECGChart(data.ecg_signal);
            });
            
            socket.on('hrv_update', function(data) {
                updateHRVCharts(data.hrv_params);
            });
            
            socket.on('anxiety_update', function(data) {
                updateAnxietyChart(data.prediction);
                updateAnxietyStatus(data.prediction);
            });
            
            socket.on('stimulation_update', function(data) {
                updateStimulationStatus(data.status);
            });
            
            socket.on('device_update', function(data) {
                updateDevicesList(data.devices);
            });
        }
        
        // Initialize UI controls
        function initializeUIControls() {
            // Start stimulation form
            document.getElementById('start-stimulation-form').addEventListener('submit', function(e) {
                e.preventDefault();
                startStimulation();
            });
            
            // Stop stimulation button
            document.getElementById('stop-stimulation-btn').addEventListener('click', function() {
                stopStimulation();
            });
        }
        
        // Fetch system status
        function fetchSystemStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('connected-clients').textContent = data.connected_clients;
                    document.getElementById('simulation-status').textContent = data.is_simulation_running ? 'Yes' : 'No';
                    document.getElementById('server-time').textContent = new Date(data.server_time).toLocaleString();
                })
                .catch(error => {
                    console.error('Error fetching system status:', error);
                    updateSystemStatus('Connection Error', 'disconnected');
                });
        }
        
        // Fetch devices
        function fetchDevices() {
            fetch('/api/devices')
                .then(response => response.json())
                .then(data => {
                    updateDevicesList(data.devices);
                })
                .catch(error => {
                    console.error('Error fetching devices:', error);
                });
        }
        
        // Fetch recent ECG data
        function fetchRecentECG() {
            fetch('/api/ecg/recent')
                .then(response => response.json())
                .then(data => {
                    if (data.ecg_data && data.ecg_data.length > 0) {
                        updateECGChart(data.ecg_data);
                    }
                })
                .catch(error => {
                    console.error('Error fetching ECG data:', error);
                });
        }
        
        // Fetch recent HRV data
        function fetchRecentHRV() {
            fetch('/api/hrv/recent')
                .then(response => response.json())
                .then(data => {
                    if (data.hrv_data && data.hrv_data.length > 0) {
                        data.hrv_data.forEach(hrv => {
                            updateHRVCharts(hrv);
                        });
                    }
                })
                .catch(error => {
                    console.error('Error fetching HRV data:', error);
                });
        }
        
        // Fetch recent anxiety data
        function fetchRecentAnxiety() {
            fetch('/api/anxiety/recent')
                .then(response => response.json())
                .then(data => {
                    if (data.anxiety_data && data.anxiety_data.length > 0) {
                        data.anxiety_data.forEach(anxiety => {
                            updateAnxietyChart(anxiety);
                        });
                        
                        // Update current anxiety status with latest data
                        updateAnxietyStatus(data.anxiety_data[data.anxiety_data.length - 1]);
                    }
                })
                .catch(error => {
                    console.error('Error fetching anxiety data:', error);
                });
        }
        
        // Fetch stimulation status
        function fetchStimulationStatus() {
            fetch('/api/stimulation/status')
                .then(response => response.json())
                .then(data => {
                    updateStimulationStatus(data);
                })
                .catch(error => {
                    console.error('Error fetching stimulation status:', error);
                });
        }
        
        // Update system status
        function updateSystemStatus(status, statusClass) {
            document.getElementById('system-status-text').textContent = status;
            const indicator = document.getElementById('system-status-indicator');
            indicator.className = 'status-indicator';
            
            if (statusClass === 'connected') {
                indicator.classList.add('status-connected');
            } else if (statusClass === 'disconnected') {
                indicator.classList.add('status-disconnected');
            } else if (statusClass === 'warning') {
                indicator.classList.add('status-warning');
            }
        }
        
        // Update ECG chart
        function updateECGChart(ecgSignal) {
            if (!ecgSignal || ecgSignal.length === 0) return;
            
            // Add new data
            ecgData = ecgData.concat(ecgSignal);
            
            // Keep only the last 10 seconds of data (assuming 250 Hz sample rate)
            const maxSamples = 10 * 250;
            if (ecgData.length > maxSamples) {
                ecgData = ecgData.slice(-maxSamples);
            }
            
            // Create time axis
            const timeAxis = Array.from({ length: ecgData.length }, (_, i) => i / 250);
            
            // Update chart
            ecgChart.data.labels = timeAxis;
            ecgChart.data.datasets[0].data = ecgData;
            ecgChart.update();
        }
        
        // Update HRV charts
        function updateHRVCharts(hrvParams) {
            if (!hrvParams) return;
            
            // Add timestamp
            const timestamp = new Date().getTime() / 60000; // Convert to minutes
            
            // Update RMSSD chart
            if (hrvParams.RMSSD !== undefined) {
                rmssdChart.data.labels.push(timestamp);
                rmssdChart.data.datasets[0].data.push(hrvParams.RMSSD);
                
                // Keep only last 30 points
                if (rmssdChart.data.labels.length > 30) {
                    rmssdChart.data.labels.shift();
                    rmssdChart.data.datasets[0].data.shift();
                }
                
                rmssdChart.update();
            }
            
            // Update LF/HF chart
            if (hrvParams.LF_HF_ratio !== undefined) {
                lfHfChart.data.labels.push(timestamp);
                lfHfChart.data.datasets[0].data.push(hrvParams.LF_HF_ratio);
                
                // Keep only last 30 points
                if (lfHfChart.data.labels.length > 30) {
                    lfHfChart.data.labels.shift();
                    lfHfChart.data.datasets[0].data.shift();
                }
                
                lfHfChart.update();
            }
        }
        
        // Update anxiety chart
        function updateAnxietyChart(prediction) {
            if (!prediction || prediction.risk_level === undefined) return;
            
            // Add timestamp
            const timestamp = new Date().getTime() / 60000; // Convert to minutes
            
            // Add data point
            anxietyChart.data.labels.push(timestamp);
            anxietyChart.data.datasets[0].data.push(prediction.risk_level);
            
            // Keep only last 30 points
            if (anxietyChart.data.labels.length > 30) {
                anxietyChart.data.labels.shift();
                anxietyChart.data.datasets[0].data.shift();
            }
            
            anxietyChart.update();
        }
        
        // Update anxiety status display
        function updateAnxietyStatus(prediction) {
            if (!prediction || prediction.risk_level === undefined) return;
            
            const anxietyLevel = document.getElementById('anxiety-level');
            const anxietyCategory = document.getElementById('anxiety-category');
            const anxietyProgress = document.getElementById('anxiety-progress');
            const anxietyConfidence = document.getElementById('anxiety-confidence');
            
            // Update level and category
            anxietyLevel.textContent = prediction.risk_level;
            anxietyCategory.textContent = prediction.risk_category;
            
            // Update confidence
            anxietyConfidence.textContent = prediction.confidence.toFixed(2);
            
            // Update progress bar
            const progressPercent = (prediction.risk_level / 3) * 100;
            anxietyProgress.style.width = `${progressPercent}%`;
            anxietyProgress.setAttribute('aria-valuenow', progressPercent);
            
            // Update color
            anxietyLevel.className = 'anxiety-level';
            anxietyProgress.className = 'progress-bar';
            
            if (prediction.risk_level === 0) {
                anxietyLevel.classList.add('anxiety-level-0');
                anxietyProgress.classList.add('bg-success');
            } else if (prediction.risk_level === 1) {
                anxietyLevel.classList.add('anxiety-level-1');
                anxietyProgress.classList.add('bg-warning');
            } else if (prediction.risk_level === 2) {
                anxietyLevel.classList.add('anxiety-level-2');
                anxietyProgress.classList.add('bg-orange');
            } else if (prediction.risk_level === 3) {
                anxietyLevel.classList.add('anxiety-level-3');
                anxietyProgress.classList.add('bg-danger');
            }
        }
        
        // Update stimulation status
        function updateStimulationStatus(status) {
            if (!status) return;
            
            const stimStatusText = document.getElementById('stimulation-status-text');
            const stimActiveContainer = document.getElementById('stimulation-active-container');
            const stimControlContainer = document.getElementById('stimulation-control-container');
            const stimDuration = document.getElementById('stimulation-duration');
            const stimRemaining = document.getElementById('stimulation-remaining');
            const stimProgress = document.getElementById('stimulation-progress');
            
            if (status.is_stimulating && status.session) {
                // Stimulation is active
                stimStatusText.textContent = 'Active';
                stimActiveContainer.style.display = 'block';
                stimControlContainer.style.display = 'none';
                
                // Update duration
                const totalDuration = status.session.duration;
                const durationMins = Math.floor(totalDuration / 60);
                const durationSecs = Math.floor(totalDuration % 60);
                stimDuration.textContent = `${durationMins}:${durationSecs.toString().padStart(2, '0')}`;
                
                // Calculate remaining time
                const elapsedTime = Date.now() / 1000 - status.session.start_time;
                const remainingTime = Math.max(0, totalDuration - elapsedTime);
                const remainingMins = Math.floor(remainingTime / 60);
                const remainingSecs = Math.floor(remainingTime % 60);
                stimRemaining.textContent = `${remainingMins}:${remainingSecs.toString().padStart(2, '0')}`;
                
                // Update progress bar
                const progressPercent = Math.min(100, (elapsedTime / totalDuration) * 100);
                stimProgress.style.width = `${progressPercent}%`;
                stimProgress.setAttribute('aria-valuenow', progressPercent);
                
                // Update stimulation waveform visualization
                updateStimulationWaveforms(status.session);
            } else {
                // Stimulation is not active
                stimStatusText.textContent = 'Not Active';
                stimActiveContainer.style.display = 'none';
                stimControlContainer.style.display = 'block';
                
                // Clear stimulation waveforms
                clearStimulationWaveforms();
            }
        }
        
        // Update stimulation waveforms
        function updateStimulationWaveforms(session) {
            if (!session) return;
            
            const frequency = session.frequency || 0.5;
            const waveform = session.waveform || 'sine';
            const phaseOffset = session.phase_difference || 0.5;
            
            // Generate time points (2 seconds of data)
            const t = Array.from({ length: 200 }, (_, i) => i * 0.01);
            
            // Generate waveforms based on type
            let wave1 = [], wave2 = [];
            
            if (waveform === 'sine') {
                wave1 = t.map(time => Math.sin(2 * Math.PI * frequency * time));
                wave2 = t.map(time => Math.sin(2 * Math.PI * frequency * (time - phaseOffset)));
            } else if (waveform === 'square') {
                wave1 = t.map(time => Math.sign(Math.sin(2 * Math.PI * frequency * time)));
                wave2 = t.map(time => Math.sign(Math.sin(2 * Math.PI * frequency * (time - phaseOffset))));
            } else if (waveform === 'triangular') {
                wave1 = t.map(time => {
                    const phase = (time * frequency) % 1;
                    return phase < 0.5 ? 4 * phase - 1 : 3 - 4 * phase;
                });
                wave2 = t.map(time => {
                    const phase = ((time - phaseOffset) * frequency) % 1;
                    return phase < 0.5 ? 4 * phase - 1 : 3 - 4 * phase;
                });
            } else if (waveform === 'monophasic') {
                wave1 = t.map(time => Math.max(0, Math.sin(2 * Math.PI * frequency * time)));
                wave2 = t.map(time => Math.max(0, Math.sin(2 * Math.PI * frequency * (time - phaseOffset))));
            } else { // biphasic
                wave1 = t.map(time => Math.sin(2 * Math.PI * frequency * time));
                wave2 = t.map(time => Math.sin(2 * Math.PI * frequency * (time - phaseOffset)));
            }
            
            // Apply intensity if available
            if (session.intensities) {
                const deviceIds = Object.keys(session.intensities);
                if (deviceIds.length > 0 && deviceIds[0]) {
                    wave1 = wave1.map(v => v * session.intensities[deviceIds[0]]);
                }
                if (deviceIds.length > 1 && deviceIds[1]) {
                    wave2 = wave2.map(v => v * session.intensities[deviceIds[1]]);
                }
            }
            
            // Update charts
            stim1Chart.data.labels = t;
            stim1Chart.data.datasets[0].data = wave1;
            stim1Chart.update();
            
            stim2Chart.data.labels = t;
            stim2Chart.data.datasets[0].data = wave2;
            stim2Chart.update();
        }
        
        // Clear stimulation waveforms
        function clearStimulationWaveforms() {
            stim1Chart.data.labels = [];
            stim1Chart.data.datasets[0].data = [];
            stim1Chart.update();
            
            stim2Chart.data.labels = [];
            stim2Chart.data.datasets[0].data = [];
            stim2Chart.update();
        }
        
        // Update devices list
        function updateDevicesList(devices) {
            const devicesContainer = document.getElementById('devices-container');
            const deviceCheckboxes = document.getElementById('device-checkboxes');
            
            if (!devices || Object.keys(devices).length === 0) {
                devicesContainer.innerHTML = '<p>No devices connected</p>';
                deviceCheckboxes.innerHTML = '<p>No devices available</p>';
                return;
            }
            
            // Clear containers
            devicesContainer.innerHTML = '';
            deviceCheckboxes.innerHTML = '<label class="form-label">Select Devices</label>';
            
            // Add devices
            Object.entries(devices).forEach(([deviceId, device]) => {
                // Add to devices list
                const deviceCard = document.createElement('div');
                deviceCard.className = 'device-card';
                
                const statusClass = device.status === 'connected' ? 'status-connected' : 'status-disconnected';
                
                deviceCard.innerHTML = `
                    <div>
                        <span class="status-indicator ${statusClass}"></span>
                        <strong>${device.device_name}</strong> (${deviceId})
                    </div>
                    <div>Type: ${device.device_type || 'Unknown'}</div>
                    <div>Status: ${device.status || 'Unknown'}</div>
                    ${device.battery_level !== undefined ? `<div>Battery: ${device.battery_level}%</div>` : ''}
                `;
                
                devicesContainer.appendChild(deviceCard);
                
                // Add to device checkboxes
                const checkbox = document.createElement('div');
                checkbox.className = 'form-check';
                checkbox.innerHTML = `
                    <input class="form-check-input" type="checkbox" value="${deviceId}" id="device-${deviceId}" checked>
                    <label class="form-check-label" for="device-${deviceId}">
                        ${device.device_name} (${deviceId})
                    </label>
                `;
                
                deviceCheckboxes.appendChild(checkbox);
            });
        }
        
        // Start stimulation
        function startStimulation() {
            // Get form values
            const anxietyLevel = parseInt(document.getElementById('anxiety-level-select').value);
            const duration = parseInt(document.getElementById('duration-input').value);
            const waveform = document.getElementById('waveform-select').value;
            const frequency = parseFloat(document.getElementById('frequency-input').value);
            const phaseDiff = parseFloat(document.getElementById('phase-diff-input').value);
            
            // Get selected devices
            const deviceCheckboxes = document.querySelectorAll('#device-checkboxes input[type="checkbox"]:checked');
            const deviceIds = Array.from(deviceCheckboxes).map(cb => cb.value);
            
            if (deviceIds.length === 0) {
                alert('Please select at least one device.');
                return;
            }
            
            // Prepare request
            const requestData = {
                anxiety_level: anxietyLevel,
                session_duration: duration,
                waveform: waveform,
                base_frequency: frequency,
                phase_difference: phaseDiff,
                device_ids: deviceIds
            };
            
            // Send request
            fetch('/api/stimulation/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log('Stimulation started successfully');
                } else {
                    console.error('Failed to start stimulation:', data.error);
                    alert(`Failed to start stimulation: ${data.error}`);
                }
            })
            .catch(error => {
                console.error('Error starting stimulation:', error);
                alert('Error starting stimulation. See console for details.');
            });
        }
        
        // Stop stimulation
        function stopStimulation() {
            fetch('/api/stimulation/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log('Stimulation stopped successfully');
                } else {
                    console.error('Failed to stop stimulation:', data.error);
                    alert(`Failed to stop stimulation: ${data.error}`);
                }
            })
            .catch(error => {
                console.error('Error stopping stimulation:', error);
                alert('Error stopping stimulation. See console for details.');
            });
        }
    </script>
</body>
</html>""")
    
    # Create static folder
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)

if __name__ == '__main__':
    # Create templates folder and index.html
    create_templates_folder()
    
    # Run the server
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f'Starting API server on {host}:{port}')
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False)
