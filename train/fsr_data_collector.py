import serial
import time
import threading

class FSRDataCollector:
    def __init__(self, port='/dev/tty.usbmodem1301', baudrate=115200):
        self.ser = serial.Serial(port, baudrate)
        self.ser.flushInput()
        self.threshold = 50
        self.ser_lock = threading.Lock()

    def get_current_data(self):
        with self.ser_lock:
            if self.ser.in_waiting > 0:
                raw_data = self.ser.readline()
            else:
                return None  # No data available

        try:
            data = raw_data.decode('utf-8').strip()
        except UnicodeDecodeError:
            return None  # Error in decoding

        if data.startswith("DATA:"):
            data_values = data[5:]  # Extract the data after "DATA:"
            try:
                force_A0, force_A1, force_A2 = map(float, data_values.split(','))
                return (force_A0, force_A1, force_A2)  # Return the current data
            except ValueError:
                return None  # Error in value conversion

    def get_data(self):
        return self.get_current_data()  # Return current data instead of accumulated data

    def close(self):
        self.ser.close()