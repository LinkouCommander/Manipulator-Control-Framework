import time
from dynamixel_sdk import *  # Uses Dynamixel SDK library
import numpy as np

class DXLHandler:
    # Dynamixel motor control setup
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    ADDR_PROFILE_VELOCITY = 112
    ADDR_TEMPERATURE = 146
    PROTOCOL_VERSION = 2.0
    DXL_MIN_LIMIT = 0
    DXL_MAX_LIMIT = 4096

    # Motor IDs for the 9 claws
    DXL_IDs = [10, 11, 12, 20, 21, 22, 30, 31, 32]
    DXL_INIT_POS = [1024, 1536, 2560, 1024, 1536, 2560, 1024, 1536, 2560]

    def __init__(self, device_name = 'COM4', baudrate = 1000000):
        self.desired_velocity = 65
        self.device_name = device_name
        self.baudrate = baudrate

        # Initialize PortHandler and PacketHandler
        self.portHandler = PortHandler(self.device_name)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        self._open_port()
        self._set_baudrate()
        
    def start_dxl(self):
        self.enable_torque(self.DXL_IDs)
        self.set_velocity(self.DXL_IDs)

    def stop_dxl(self):
        self.disable_torque(self.DXL_IDs)
        self.portHandler.closePort()

################################################################################################
# Main Functions
################################################################################################

    # Enable motor torque
    def enable_torque(self, ids):
        for id in ids:
            self._write_byte(1, id, self.ADDR_TORQUE_ENABLE, 1)

    #   Disable motor torque
    def disable_torque(self, ids):
        for id in ids:
            self._write_byte(1, id, self.ADDR_TORQUE_ENABLE, 0)

    # Set the desired velocity for motors
    def set_velocity(self, ids):
        for id in ids:
            self._write_byte(4, id, self.ADDR_PROFILE_VELOCITY, self.desired_velocity)

    # read position of motor
    def read_positions(self, ids):
        position_list = []
        for id in ids:
            position_list.append(self._read_byte(4, id, self.ADDR_PRESENT_POSITION))
            time.sleep(0.01)
        return position_list

    # read temperature of motor
    def read_temperature(self, ids=DXL_IDs):
        temperature_list = []
        for id in ids:
            temperature_list.append(self._read_byte(1, id, self.ADDR_TEMPERATURE))
        return temperature_list
    
    # move motors
    def move_to_position(self, ids, destinations):
        """
        Move motors to specified positions.

        :param ids: List of motor IDs to move.
        :param destinations: List of target positions for the motors.
        :return: 1 if successful, 0 if timed out, -1 if invalid input.
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
            if np.all(diff < 10) or np.any(np.array(current_positions) == -1):
                # print(f"(destination: {destinations}), (current: {current_positions})")
                return 1

            if time.time() - start_time > 2:
                return 0
            
            time.sleep(0.1)


################################################################################################
# Communication Functions
################################################################################################

    def _open_port(self):
        if not self.portHandler.openPort():
            raise Exception("[DXL] Failed to open the port")

    def _set_baudrate(self):
        if not self.portHandler.setBaudRate(self.baudrate):
            raise Exception("[DXL] Failed to change the baudrate")

    def _write_byte(self, byteNum, dxl_id, address, value):
        if byteNum == 1:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, address, value)
        else:
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, address, value)

        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        # else:
        #     print(f"Motor {dxl_id}: Wrote {value} to {address}")

    def _read_byte(self, byteNum, dxl_id, address):
        if byteNum == 1:
            value, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(self.portHandler, dxl_id, address)
        else:
            value, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, dxl_id, self.ADDR_PRESENT_POSITION)

        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            return -1
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            return -1
        else:
            return value