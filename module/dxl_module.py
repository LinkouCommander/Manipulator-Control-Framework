import time
from dynamixel_sdk import *  # Uses Dynamixel SDK library
import numpy as np

class DXLHandler:
    """
    A handler class for controlling multiple Dynamixel motors using the Dynamixel SDK.
    Provides methods for initializing communication, enabling/disabling torque, 
    setting motor velocity, reading position/temperature, and commanding motor movements.
    """

    # Control table addresses (based on Dynamixel X-series)
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    ADDR_PROFILE_VELOCITY = 112
    ADDR_TEMPERATURE = 146

    # Protocol version for communication
    PROTOCOL_VERSION = 2.0

    # Motor position limits
    DXL_MIN_LIMIT = 0
    DXL_MAX_LIMIT = 4096

    # IDs and initial positions of 9 Dynamixel motors
    DXL_IDs = [10, 11, 12, 20, 21, 22, 30, 31, 32]
    DXL_INIT_POS = [1024, 1536, 2560, 1024, 1536, 2560, 1024, 1536, 2560]

    def __init__(self, device_name = 'COM4', baudrate = 1000000):
        self.desired_velocity = 65
        self.device_name = device_name
        self.baudrate = baudrate

        # Create PortHandler and PacketHandler instances
        self.portHandler = PortHandler(self.device_name)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        self._open_port()
        self._set_baudrate()
        
    def start_dxl(self):
        """
        Enable torque and set velocity for all motors.
        """
        self.enable_torque(self.DXL_IDs)
        self.set_velocity(self.DXL_IDs)

    def stop_dxl(self):
        """
        Disable torque and close the communication port.
        """
        self.disable_torque(self.DXL_IDs)
        self.portHandler.closePort()

# ==============================================================================================
# Motor Control Functions
# ==============================================================================================

    def enable_torque(self, ids):
        """
        Enables torque for specified Dynamixel motor IDs.

        Args:
            ids (list): List of motor IDs.
        """
        for id in ids:
            self._write_byte(1, id, self.ADDR_TORQUE_ENABLE, 1)

    def disable_torque(self, ids):
        """
        Disables torque for specified Dynamixel motor IDs.

        Args:
            ids (list): List of motor IDs.
        """
        for id in ids:
            self._write_byte(1, id, self.ADDR_TORQUE_ENABLE, 0)

    def set_velocity(self, ids):
        """
        Sets the velocity profile for specified Dynamixel motor IDs.

        Args:
            ids (list): List of motor IDs.
        """
        for id in ids:
            self._write_byte(4, id, self.ADDR_PROFILE_VELOCITY, self.desired_velocity)

    def read_positions(self, ids):
        """
        Reads the current positions of the specified motors.

        Args:
            ids (list): List of motor IDs.

        Returns:
            position_list (list): List of current positions.
        """
        position_list = []
        for id in ids:
            position_list.append(self._read_byte(4, id, self.ADDR_PRESENT_POSITION))
            time.sleep(0.01)
        return position_list

    # read temperature of motor
    def read_temperature(self, ids=DXL_IDs):
        """
        Reads the temperature of the specified motors.

        Args:
            ids (list, optional): List of motor IDs. Defaults to all motors.

        Returns:
            temperature_list (list): List of temperature readings for each motor.
        """
        temperature_list = []
        for id in ids:
            temperature_list.append(self._read_byte(1, id, self.ADDR_TEMPERATURE))
        return temperature_list
    
    # move motors
    def move_to_position(self, ids, destinations):
        """
        Commands motors to move to target positions.

        Args:
            ids (list): List of motor IDs to control.
            destinations (list): Target positions for each motor.

        Returns:
            tuple:
                int: 1 if movement successful, 0 if timeout, -1 if invalid input.
                list: Final positions of motors when condition met or timed out.
        """
        for i, id in enumerate(ids):
            if destinations[i] < self.DXL_MIN_LIMIT or destinations[i] > self.DXL_MAX_LIMIT or id not in self.DXL_IDs:
                return -1
            self._write_byte(4, id, self.ADDR_GOAL_POSITION, destinations[i])
        
        start_time = time.time()
        while True:
            current_positions = self.read_positions(ids)
            # print(current_positions, destinations)
            diff = np.abs(np.array(current_positions) - np.array(destinations))
            if np.all(diff < 50) and np.any(np.array(current_positions) != -1):
                # print(f"(destination: {destinations}), (current: {current_positions})")
                return 1, current_positions

            if time.time() - start_time > 2:
                return 0, current_positions
            
            time.sleep(0.1)


# ==============================================================================================
# Communication Functions
# ==============================================================================================


    def _open_port(self):
        if not self.portHandler.openPort():
            raise Exception("[DXL] Failed to open the port")

    def _set_baudrate(self):
        if not self.portHandler.setBaudRate(self.baudrate):
            raise Exception("[DXL] Failed to change the baudrate")

    def _write_byte(self, byteNum, dxl_id, address, value):
        """
        Writes a value to the specified address of a Dynamixel motor.

        Args:
            byteNum (int): Number of bytes to write (1 or 4).
            dxl_id (int): ID of the motor.
            address (int): Register address.
            value (int): Value to write.
        """
        if byteNum == 1:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, address, value)
        else:
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, address, value)

        if dxl_comm_result != COMM_SUCCESS:
            print(f"[DXL] Comm failed (write) - ID: {dxl_id}, Addr: {address}, Value: {value}, Msg: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"[DXL] DXL error (write) - ID: {dxl_id}, Addr: {address}, Value: {value}, Msg: {self.packetHandler.getRxPacketError(dxl_error)}")
        # else:
        #     print(f"Motor {dxl_id}: Wrote {value} to {address}")

    def _read_byte(self, byteNum, dxl_id, address):
        """
        Reads a value from the specified address of a Dynamixel motor.

        Args:
            byteNum (int): Number of bytes to read (1 or 4).
            dxl_id (int): ID of the motor.
            address (int): Register address.

        Returns:
            int: Read value or -1 on failure.
        """
        if byteNum == 1:
            value, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(self.portHandler, dxl_id, address)
        else:
            value, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, dxl_id, self.ADDR_PRESENT_POSITION)

        if dxl_comm_result != COMM_SUCCESS:
            # print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            print(f"[DXL] Comm failed (read) - ID: {dxl_id}, Addr: {address}, Msg: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            return -1
        elif dxl_error != 0:
            # print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            print(f"[DXL] DXL error (read) - ID: {dxl_id}, Addr: {address}, Msg: {self.packetHandler.getRxPacketError(dxl_error)}")
            return -1
        else:
            return value