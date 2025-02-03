import serial
import time
import threading
import matplotlib.pyplot as plt

class FSRSerialReader:
    def __init__(self, port, baudrate=115200, threshold=50):
        self.ser = serial.Serial(port, baudrate)
        self.ser.flushInput()
        
        self.force_data = {"A0": [], "A1": [], "A2": []}
        self.binary_data = {"A0": [], "A1": [], "A2": []}
        self.time_data = []
        self.threshold = threshold
        self.stop_collecting = False
        self.ser_lock = threading.Lock()
        self.data_thread = None

    def collect_data(self):
        start_time = time.time()
        while not self.stop_collecting:
            with self.ser_lock:
                if self.ser.in_waiting > 0:
                    raw_data = self.ser.readline()
                else:
                    continue
            try:
                data = raw_data.decode('utf-8').strip()
            except UnicodeDecodeError:
                continue
            
            if data.startswith("DATA:"):
                try:
                    force_A0, force_A1, force_A2 = map(float, data[5:].split(','))
                    current_time = time.time() - start_time
                    
                    self.force_data["A0"].append(force_A0)
                    self.force_data["A1"].append(force_A1)
                    self.force_data["A2"].append(force_A2)
                    self.time_data.append(current_time)
                    
                    self.binary_data["A0"].append(1 if force_A0 > self.threshold else 0)
                    self.binary_data["A1"].append(1 if force_A1 > self.threshold else 0)
                    self.binary_data["A2"].append(1 if force_A2 > self.threshold else 0)
                except ValueError:
                    continue
            else:
                print(f"Arduino: {data}")

    def get_fsr(self):
        return (
            self.force_data["A0"][-1] if self.force_data["A0"] else None,
            self.force_data["A1"][-1] if self.force_data["A1"] else None,
            self.force_data["A2"][-1] if self.force_data["A2"] else None
        )

    def start_collection(self):
        self.stop_collecting = False
        self.data_thread = threading.Thread(target=self.collect_data)
        self.data_thread.start()
    
    def stop_collection(self):
        self.stop_collecting = True
        if self.data_thread:
            self.data_thread.join()
        self.ser.close()
    
    def send_position(self, pos):
        if 0 <= pos <= 180:
            with self.ser_lock:
                self.ser.write(f"{pos}\n".encode('utf-8'))
        else:
            print("Invalid position. Please enter a value between 0 and 180.")
    
    def plot_data(self):
        plt.figure(1)
        for key in ["A0", "A1", "A2"]:
            plt.plot(self.time_data, self.force_data[key], label=f'FSR {key}')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.title('Force-Time Curves for FSR A0, A1, and A2')
        plt.legend()
        plt.grid(True)

        plt.figure(2)
        for key in ["A0", "A1", "A2"]:
            plt.step(self.time_data, self.binary_data[key], where='post', label=f'FSR {key}')
        plt.xlabel('Time (s)')
        plt.ylabel('Binary Signal')
        plt.title('Binary Signal Plot (Force Threshold)')
        plt.legend()
        plt.grid(True)

        plt.show()