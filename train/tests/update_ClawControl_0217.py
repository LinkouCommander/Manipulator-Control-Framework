import time
from dynamixel_sdk import *  # Uses Dynamixel SDK library

# Dynamixel motor control setup
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_TEMPERATURE = 146
PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000
DEVICENAME = 'COM3'  # COM port for claw control

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

# Move motor to specific position with timeout mechanism
def move_to_position(DXL_ID, target_position):
    packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, target_position)
    
    start_time = time.time()
    while True:
        current_position, _, _ = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
        if abs(current_position - target_position) < 10:  # Tolerance of 10 units
            print(f"Motor {DXL_ID} moved to position {target_position} (current position: {current_position})")
            return
        
        if time.time() - start_time > 3:  # Timeout of 3 seconds
            fallback_position = 1024 if DXL_ID in [10, 20, 30] else 2048
            print(f"Motor {DXL_ID} could not reach position {target_position}. Moving back to {fallback_position}.")
            packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, fallback_position)  # Move back to fallback position
            
            # Wait for motor to reach fallback position
            while True:
                current_position, _, _ = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
                if abs(current_position - fallback_position) < 10:
                    print(f"Position cannot be reached. Motor moved back to {fallback_position} (current position: {current_position}).")
                    break
                time.sleep(0.1)  # Small delay to allow the motor to move
            return  # Ensure the function exits cleanly after fallback
        
        time.sleep(0.1)  # Small delay to allow the motor to move

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

# Read temperature of all motors
def read_temperatures(DXL_IDs):
    temperatures = {}
    for DXL_ID in DXL_IDs:
        temperature, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(portHandler, DXL_ID, ADDR_TEMPERATURE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
        else:
            temperatures[DXL_ID] = temperature
    print("Motor temperatures:", temperatures)

# Enable torque
enable_torque(portHandler, packetHandler, DXL_IDs)

# Report initial positions
initial_positions = read_positions(DXL_IDs)
print("Initial motor positions:", initial_positions)

# Command loop
try:
    while True:
        command = input("Enter motor ID to move (10-32), 'temp' to check temperatures, or 'quit' to exit: ").strip()
        if command.lower() == 'quit':
            break
        elif command.lower() == 'temp':
            read_temperatures(DXL_IDs)
        elif command.isdigit() and int(command) in DXL_IDs:
            motor_id = int(command)
            position = int(input(f"Enter position value for motor {motor_id} (0-4095): "))
            move_to_position(motor_id, position)
            positions = read_positions(DXL_IDs)
            print("Current motor positions:", positions)
        else:
            print("Invalid command. Please enter a valid motor ID, 'temp' to check temperatures, or 'quit' to exit.")

finally:
    disable_torque(portHandler, packetHandler, DXL_IDs)
    portHandler.closePort()
    print("Torque disabled, and port closed.")
