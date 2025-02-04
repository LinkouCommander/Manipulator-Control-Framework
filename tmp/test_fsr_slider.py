import serial
import time
import matplotlib.pyplot as plt
import threading

# Setup serial communication with the specified port and 115200 baud rate
ser = serial.Serial('COM5', 115200)  # Adjust to your port
ser.flushInput()

# Initialize data collection for three FSRs
force_data_A0 = []
force_data_A1 = []
force_data_A2 = []
time_data = []
binary_data_A0 = []
binary_data_A1 = []
binary_data_A2 = []
threshold = 50  # Example threshold to create binary signals
stop_collecting = False

ser_lock = threading.Lock()  # Lock for serial port access

# Function to collect data from the serial port
def collect_data():
    global stop_collecting
    start_time = time.time()
    
    while not stop_collecting:
        with ser_lock:
            if ser.in_waiting > 0:
                raw_data = ser.readline()
            else:
                continue
        try:
            data = raw_data.decode('utf-8').strip()
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError: {e}")
            print(f"Raw data: {raw_data}")
            continue  # Skip this iteration and continue with the next one
        
        try:
            if data.startswith("DATA:"):
                data_values = data[5:]  # Remove "DATA:" prefix
                # Parse force values from three FSRs connected to A0, A1, A2
                force_A0, force_A1, force_A2 = map(float, data_values.split(','))  
                current_time = time.time() - start_time  # Record elapsed time

                # Append data to lists for each FSR
                force_data_A0.append(force_A0)
                force_data_A1.append(force_A1)
                force_data_A2.append(force_A2)
                time_data.append(current_time)

                # Generate binary signals based on the threshold for each FSR
                binary_data_A0.append(1 if force_A0 > threshold else 0)
                binary_data_A1.append(1 if force_A1 > threshold else 0)
                binary_data_A2.append(1 if force_A2 > threshold else 0)

                # Print force readings for all three FSRs
              #  print(f"Time: {current_time:.2f}s, Force A0: {force_A0}N, Force A1: {force_A1}N, Force A2: {force_A2}N")
            else:
                # Print other messages from Arduino
                print(f"Arduino: {data}")
        except ValueError as ve:
            print(f"ValueError: {ve}")
            print(f"Invalid data received: {data}")
            continue  # Skip invalid data


# Start data collection in a separate thread
data_thread = threading.Thread(target=collect_data)
data_thread.start()

# Main loop to accept user input
try:
    while True:
        user_input = input("Enter a number (75-145) to move to that position, 'exit' to quit, or 'stop' to stop data collection: ")
        if user_input.lower() == 'exit':
            stop_collecting = True
            break
        elif user_input.lower() == 'stop':
            stop_collecting = True
        elif user_input.isdigit():
            pos = int(user_input)
            if 75 <= pos <= 145:
                with ser_lock:
                    ser.write(f"{pos}\n".encode('utf-8'))  # Send position to Arduino
            else:
                print("Invalid position. Please enter a value between 75 and 145.")
        else:
            print("Invalid command.")
except KeyboardInterrupt:
    stop_collecting = True

# Wait for data collection to stop
data_thread.join()

# Close serial connection
ser.close()

# Plot the force-time curves for all three FSRs
plt.figure(1)
plt.plot(time_data, force_data_A0, label='FSR A0')
plt.plot(time_data, force_data_A1, label='FSR A1')
plt.plot(time_data, force_data_A2, label='FSR A2')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Force-Time Curves for FSR A0, A1, and A2')
plt.legend()
plt.grid(True)

# Plot the binary signal curves for all three FSRs
plt.figure(2)
plt.step(time_data, binary_data_A0, where='post', label='FSR A0')
plt.step(time_data, binary_data_A1, where='post', label='FSR A1')
plt.step(time_data, binary_data_A2, where='post', label='FSR A2')
plt.xlabel('Time (s)')
plt.ylabel('Binary Signal')
plt.title('Binary Signal Plot (Force Threshold)')
plt.legend()
plt.grid(True)

# Show both plots
plt.show()