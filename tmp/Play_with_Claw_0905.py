import time
from dynamixel_sdk import *  # Uses Dynamixel SDK library

# Dynamixel motor control setup
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000
DEVICENAME = 'COM4'  # COM port for claw control

# Motor IDs for the 9 claws
DXL_IDs = [10, 11, 12, 20, 21, 22, 30, 31, 32]

# Initialize PortHandler and PacketHandler
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if not portHandler.openPort():
    print("Failed to open the port")
    quit()

# Set port baudrate
if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to change the baudrate")
    quit()

# Enable torque for all motors
def enable_torque(portHandler, packetHandler, DXL_IDs):
    for DXL_ID in DXL_IDs:
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, 1)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
        else:
            print(f"Torque enabled for ID: {DXL_ID}")

# Disable torque for all motors
def disable_torque(portHandler, packetHandler, DXL_IDs):
    for DXL_ID in DXL_IDs:
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, 0)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
        else:
            print(f"Torque disabled for ID: {DXL_ID}")

# Move motor to specific position with feedback and delay
def move_to_position(DXL_ID, target_position):
    packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, target_position)
    
    # Wait for the motor to reach the target position
    while True:
        current_position, _, _ = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
        if abs(current_position - target_position) < 10:  # Tolerance of 10 units
            break
        time.sleep(0.1)  # Small delay to allow the motor to move

    print(f"Motor {DXL_ID} moved to position {target_position} (current position: {current_position})")

# Read position of all motors
def read_positions(DXL_IDs):
    positions = {}
    for DXL_ID in DXL_IDs:
        position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
        else:
            positions[DXL_ID] = position
    return positions

# Enable torque
enable_torque(portHandler, packetHandler, DXL_IDs)

# Command loop
try:
    while True:
        command = input("Enter motor ID to move (10-32) or 'quit' to exit: ").strip()
        if command.lower() == 'quit':
            break
        elif command.isdigit() and int(command) in DXL_IDs:
            motor_id = int(command)
            position = int(input(f"Enter position value for motor {motor_id} (0-4095): "))
            move_to_position(motor_id, position)
            positions = read_positions(DXL_IDs)
            print("Current motor positions:", positions)
        else:
            print("Invalid motor ID. Please enter a value between 10 and 32.")

finally:
    disable_torque(portHandler, packetHandler, DXL_IDs)
    portHandler.closePort()
    print("Torque disabled, and port closed.")


# {10: 1012, 11: 1538, 12: 2561, 20: 1029, 21: 1538, 22: 2559, 30: 1024, 31: 1539, 32: 2559}