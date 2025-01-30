import serial
import time

def initialize_serial(port='COM3', baud_rate=9600):
    ser = serial.Serial(port, baud_rate)
    #time.sleep(2)  # Wait for the connection to initialize
    return ser

def send_command(ser, command):
    ser.write(command.encode() + b'\n')
    print(f"Command sent to Arduino: {command}")
   # time.sleep(1)  # Give time for the command to be processed
    while ser.in_waiting > 0:
        response = ser.readline().decode('utf-8').strip()
        print(f"Arduino response: {response}")

try:
    ser = initialize_serial(port='COM3', baud_rate=9600)
    print("Serial connection established")

    while True:
        user_input = input("Enter a number (0-180) to move to that position, or 'exit' to quit: ")
        if user_input.lower() == 'exit':
            break
        elif user_input.isdigit():
            pos = int(user_input)
            if 0 <= pos <= 180:
                send_command(ser, f"{pos}")
            else:
                print("Invalid position. Please enter a value between 0 and 180.")
        else:
            print("Invalid command. Please enter a number between 0 and 180.")

    ser.close()
    print("Serial connection closed")

except serial.SerialException as e:
    print(f"Error: {e}")
    print("Make sure the port is available and not in use by another application.")
    print("Try closing other applications that might be using the port, or restarting your computer.")