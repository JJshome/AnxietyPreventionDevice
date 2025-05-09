"""
Bluetooth Manager for ECG Sensor

This module provides Bluetooth connection management for ECG sensors.
It handles device discovery, connection, and data transmission.
Implements the necessary functionality for both classic Bluetooth and BLE.
"""

import logging
import threading
import time
from typing import Dict, List, Tuple, Callable, Optional, Union
import queue

# Import platform-specific Bluetooth libraries conditionally to support multiple platforms
try:
    import bluetooth  # PyBluez for classic Bluetooth
    CLASSIC_BT_AVAILABLE = True
except ImportError:
    CLASSIC_BT_AVAILABLE = False
    logging.warning("PyBluez not available, classic Bluetooth functionality will be limited")

try:
    from bluepy import btle  # BLE for Linux
    BLE_AVAILABLE = True
except ImportError:
    BLE_AVAILABLE = False
    logging.warning("Bluepy not available, BLE functionality will be limited")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# UUID constants for BLE services and characteristics
ECG_SERVICE_UUID = "0000180D-0000-1000-8000-00805f9b34fb"  # Heart Rate Service UUID
ECG_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"    # Heart Rate Measurement Characteristic UUID


class BluetoothManager:
    """
    Manages Bluetooth connections for ECG sensors.
    Supports both classic Bluetooth and Bluetooth Low Energy (BLE).
    """
    
    def __init__(self):
        """Initialize the Bluetooth manager."""
        self.devices = {}                    # Discovered devices {address: name}
        self.connected_devices = {}          # Currently connected devices {address: connection}
        self.scanning = False                # Flag to track scanning state
        self.scan_thread = None              # Thread for background scanning
        self.data_threads = {}               # Threads for receiving data from devices
        self.data_buffers = {}               # Buffers for incoming data
        self.callbacks = {}                  # Callbacks for data handling
        self.stop_events = {}                # Events to signal threads to stop
        
    def scan_for_devices(self, duration: int = 10, ble: bool = True) -> Dict[str, str]:
        """
        Scan for available Bluetooth devices.
        
        Args:
            duration: Scan duration in seconds
            ble: Whether to scan for BLE devices (otherwise, scan for classic devices)
            
        Returns:
            Dictionary mapping device addresses to names
        """
        self.devices = {}
        
        if ble:
            if not BLE_AVAILABLE:
                logger.error("BLE scanning requested but BLE libraries not available")
                return {}
            
            try:
                logger.info(f"Scanning for BLE devices for {duration} seconds...")
                scanner = btle.Scanner()
                devices = scanner.scan(duration)
                
                for dev in devices:
                    name = None
                    for (adtype, desc, value) in dev.getScanData():
                        if desc == "Complete Local Name":
                            name = value
                            break
                    
                    if name is None:
                        name = f"Unknown BLE Device ({dev.addr})"
                    
                    self.devices[dev.addr] = name
                    logger.info(f"Found BLE device: {name} ({dev.addr})")
            
            except Exception as e:
                logger.error(f"Error scanning for BLE devices: {str(e)}")
        
        else:
            if not CLASSIC_BT_AVAILABLE:
                logger.error("Classic Bluetooth scanning requested but libraries not available")
                return {}
            
            try:
                logger.info(f"Scanning for classic Bluetooth devices for {duration} seconds...")
                nearby_devices = bluetooth.discover_devices(
                    duration=duration, 
                    lookup_names=True,
                    flush_cache=True
                )
                
                for addr, name in nearby_devices:
                    if name is None:
                        name = f"Unknown Device ({addr})"
                    
                    self.devices[addr] = name
                    logger.info(f"Found classic Bluetooth device: {name} ({addr})")
            
            except Exception as e:
                logger.error(f"Error scanning for classic Bluetooth devices: {str(e)}")
        
        return self.devices
    
    def start_background_scan(self, interval: int = 30, ble: bool = True):
        """
        Start background scanning for devices at regular intervals.
        
        Args:
            interval: Scan interval in seconds
            ble: Whether to scan for BLE devices
        """
        if self.scanning:
            logger.warning("Background scanning is already active")
            return
        
        self.scanning = True
        self.scan_thread = threading.Thread(
            target=self._background_scan_task,
            args=(interval, ble),
            daemon=True
        )
        self.scan_thread.start()
    
    def _background_scan_task(self, interval: int, ble: bool):
        """Background task for periodic device scanning."""
        while self.scanning:
            self.scan_for_devices(duration=5, ble=ble)
            time.sleep(interval - 5)
    
    def stop_background_scan(self):
        """Stop background scanning for devices."""
        self.scanning = False
        if self.scan_thread and self.scan_thread.is_alive():
            self.scan_thread.join(timeout=1.0)
    
    def connect_to_device(self, address: str, ble: bool = True) -> bool:
        """
        Connect to a Bluetooth device.
        
        Args:
            address: Bluetooth address of the device
            ble: Whether the device is BLE
            
        Returns:
            True if connection successful, False otherwise
        """
        if address in self.connected_devices:
            logger.warning(f"Already connected to device {address}")
            return True
        
        try:
            if ble:
                if not BLE_AVAILABLE:
                    logger.error("BLE connection requested but BLE libraries not available")
                    return False
                
                logger.info(f"Connecting to BLE device: {address}")
                peripheral = btle.Peripheral(address)
                self.connected_devices[address] = peripheral
                
                # Create data buffer and stop event for this device
                self.data_buffers[address] = queue.Queue()
                self.stop_events[address] = threading.Event()
                
                # Start data reception thread
                self.data_threads[address] = threading.Thread(
                    target=self._ble_data_reception_task,
                    args=(address, peripheral),
                    daemon=True
                )
                self.data_threads[address].start()
                
                logger.info(f"Connected to BLE device: {address}")
                return True
            
            else:
                if not CLASSIC_BT_AVAILABLE:
                    logger.error("Classic Bluetooth connection requested but libraries not available")
                    return False
                
                logger.info(f"Connecting to classic Bluetooth device: {address}")
                rfcomm_services = bluetooth.find_service(address=address)
                
                if not rfcomm_services:
                    logger.error(f"No RFCOMM services found on device: {address}")
                    return False
                
                # Try to connect to the first available service
                service = rfcomm_services[0]
                port = service["port"]
                sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                sock.connect((address, port))
                self.connected_devices[address] = sock
                
                # Create data buffer and stop event for this device
                self.data_buffers[address] = queue.Queue()
                self.stop_events[address] = threading.Event()
                
                # Start data reception thread
                self.data_threads[address] = threading.Thread(
                    target=self._classic_data_reception_task,
                    args=(address, sock),
                    daemon=True
                )
                self.data_threads[address].start()
                
                logger.info(f"Connected to classic Bluetooth device: {address}")
                return True
        
        except Exception as e:
            logger.error(f"Error connecting to device {address}: {str(e)}")
            return False
    
    def disconnect_from_device(self, address: str) -> bool:
        """
        Disconnect from a Bluetooth device.
        
        Args:
            address: Bluetooth address of the device
            
        Returns:
            True if disconnection successful, False otherwise
        """
        if address not in self.connected_devices:
            logger.warning(f"Not connected to device {address}")
            return False
        
        try:
            # Signal data reception thread to stop
            if address in self.stop_events:
                self.stop_events[address].set()
            
            # Wait for thread to complete
            if address in self.data_threads and self.data_threads[address].is_alive():
                self.data_threads[address].join(timeout=2.0)
            
            # Close connection
            connection = self.connected_devices[address]
            if isinstance(connection, btle.Peripheral):
                connection.disconnect()
            else:
                connection.close()
            
            # Clean up
            del self.connected_devices[address]
            if address in self.data_buffers:
                del self.data_buffers[address]
            if address in self.stop_events:
                del self.stop_events[address]
            if address in self.data_threads:
                del self.data_threads[address]
            if address in self.callbacks:
                del self.callbacks[address]
            
            logger.info(f"Disconnected from device: {address}")
            return True
        
        except Exception as e:
            logger.error(f"Error disconnecting from device {address}: {str(e)}")
            return False
    
    def register_data_callback(self, address: str, callback: Callable[[bytes], None]):
        """
        Register a callback function to process data from a specific device.
        
        Args:
            address: Bluetooth address of the device
            callback: Function to call with received data
        """
        self.callbacks[address] = callback
    
    def _ble_data_reception_task(self, address: str, peripheral: 'btle.Peripheral'):
        """Task for receiving data from a BLE device."""
        try:
            # Find ECG service and characteristic
            service = peripheral.getServiceByUUID(ECG_SERVICE_UUID)
            char = service.getCharacteristics(ECG_CHAR_UUID)[0]
            
            # Set up notifications
            peripheral.setDelegate(BLEDelegate(self.data_buffers[address]))
            peripheral.writeCharacteristic(
                char.valHandle + 1,  # Client Characteristic Configuration Descriptor
                b"\x01\x00"         # Enable notifications
            )
            
            logger.info(f"BLE notifications enabled for device: {address}")
            
            # Main notification loop
            while not self.stop_events[address].is_set():
                if peripheral.waitForNotifications(1.0):
                    # Handle notifications (delegate will put data into queue)
                    pass
                
                # Process data from buffer
                while not self.data_buffers[address].empty():
                    data = self.data_buffers[address].get_nowait()
                    
                    # Call registered callback if exists
                    if address in self.callbacks:
                        self.callbacks[address](data)
        
        except Exception as e:
            logger.error(f"Error in BLE data reception for device {address}: {str(e)}")
        
        finally:
            # Ensure device is disconnected
            try:
                if address in self.connected_devices:
                    peripheral.disconnect()
                    logger.info(f"Disconnected from BLE device: {address}")
            except Exception as e:
                logger.error(f"Error disconnecting from BLE device {address}: {str(e)}")
    
    def _classic_data_reception_task(self, address: str, sock: 'bluetooth.BluetoothSocket'):
        """Task for receiving data from a classic Bluetooth device."""
        try:
            # Set socket timeout for non-blocking reads
            sock.settimeout(1.0)
            
            # Main reception loop
            while not self.stop_events[address].is_set():
                try:
                    # Try to receive data
                    data = sock.recv(1024)
                    
                    if data:
                        # Put data in buffer
                        self.data_buffers[address].put(data)
                        
                        # Call registered callback if exists
                        if address in self.callbacks:
                            self.callbacks[address](data)
                
                except bluetooth.btcommon.BluetoothError as e:
                    # Timeout or other error
                    if "timed out" not in str(e).lower():
                        logger.error(f"Bluetooth error: {str(e)}")
                
                # Brief pause to prevent CPU overuse
                time.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Error in classic Bluetooth data reception for device {address}: {str(e)}")
        
        finally:
            # Ensure socket is closed
            try:
                if address in self.connected_devices:
                    sock.close()
                    logger.info(f"Disconnected from classic Bluetooth device: {address}")
            except Exception as e:
                logger.error(f"Error disconnecting from classic Bluetooth device {address}: {str(e)}")
    
    def send_data(self, address: str, data: bytes) -> bool:
        """
        Send data to a connected Bluetooth device.
        
        Args:
            address: Bluetooth address of the device
            data: Bytes to send
            
        Returns:
            True if data sent successfully, False otherwise
        """
        if address not in self.connected_devices:
            logger.error(f"Not connected to device {address}")
            return False
        
        try:
            connection = self.connected_devices[address]
            
            if isinstance(connection, btle.Peripheral):
                # Find the write characteristic (this would depend on the specific device)
                # For demonstration, we're using a simple approach
                services = connection.getServices()
                for service in services:
                    for char in service.getCharacteristics():
                        if char.supportsWrite():
                            char.write(data)
                            return True
                
                logger.error(f"No writable characteristic found for BLE device: {address}")
                return False
            
            else:
                # Classic Bluetooth - just write to the socket
                connection.send(data)
                return True
        
        except Exception as e:
            logger.error(f"Error sending data to device {address}: {str(e)}")
            return False


class BLEDelegate(btle.DefaultDelegate):
    """Delegate for handling BLE notifications."""
    
    def __init__(self, data_queue: queue.Queue):
        """
        Initialize the BLE delegate.
        
        Args:
            data_queue: Queue to store received data
        """
        btle.DefaultDelegate.__init__(self)
        self.data_queue = data_queue
    
    def handleNotification(self, handle: int, data: bytes):
        """Handle incoming BLE notifications."""
        self.data_queue.put(data)


# Example usage
if __name__ == "__main__":
    # Simple demonstration of the BluetoothManager
    bt_manager = BluetoothManager()
    
    # Scan for devices
    print("Scanning for Bluetooth devices...")
    devices = bt_manager.scan_for_devices(duration=5, ble=True)
    
    if not devices:
        print("No devices found.")
        exit()
    
    # Print found devices
    print("Found devices:")
    for addr, name in devices.items():
        print(f"  {name} ({addr})")
    
    # Ask user to select a device
    print("\nEnter the address of the device to connect to:")
    addr = input("> ")
    
    if addr not in devices:
        print("Invalid device address.")
        exit()
    
    # Connect to selected device
    print(f"Connecting to {devices[addr]}...")
    if bt_manager.connect_to_device(addr, ble=True):
        print("Connected!")
        
        # Define callback for received data
        def data_callback(data):
            print(f"Received data: {data}")
        
        # Register callback
        bt_manager.register_data_callback(addr, data_callback)
        
        # Keep the connection active for a while
        print("Receiving data for 30 seconds...")
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("Interrupted by user.")
        
        # Disconnect
        print("Disconnecting...")
        bt_manager.disconnect_from_device(addr)
        print("Disconnected.")
    
    else:
        print("Failed to connect.")
