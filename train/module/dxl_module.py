import time
from dynamixel_sdk import *  # Uses Dynamixel SDK library

class DXLHandler:
    # Dynamixel motor control setup
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    ADDR_PROFILE_VELOCITY = 112
    ADDR_TEMPERATURE = 146
    PROTOCOL_VERSION = 2.0

    # Motor IDs for the 9 claws
    DXL_IDs = [10, 11, 12, 20, 21, 22, 30, 31, 32]

    def __init__(self, device_name, baudrate):
        self.desired_velocity = 65
        self.device_name = device_name
        self.baudrate = baudrate

        # Initialize PortHandler and PacketHandler
        self.portHandler = PortHandler(self.device_name)
        self.packetHandler = PacketHandler(self.baudrate)

        self._open_port()
        self._set_baudrate()
        
    def start_dxl(self):
        self.enable_torque()
        self.set_velocity()

################################################################################################
# 
################################################################################################

    # Enable torque for all motors
    def enable_torque(self):
        for dxl_id in self.DXL_IDs:
            self._write_byte(1, dxl_id, self.ADDR_TORQUE_ENABLE, 1)

    #   Disable torque for all motors
    def disable_torque(self):
        for dxl_id in self.DXL_IDs:
            self._write_byte(1, dxl_id, self.ADDR_TORQUE_ENABLE, 0)

    # Set the desired velocity for all motors
    def set_velocity(self):
        for dxl_id in self.DXL_IDs:
            self._write_byte(4, dxl_id, self.ADDR_PROFILE_VELOCITY, self.desired_velocity)

    def read_positions(self):
        position_list = {}
        for dxl_id in self.DXL_IDs:
            position_list[dxl_id] = self._read_byte(4, dxl_id, self.ADDR_PRESENT_POSITION)
        return position_list

    def read_temperature(self):
        temperature_list = {}
        for dxl_id in self.DXL_IDs:
            temperature_list[dxl_id] = self._read_byte(1, dxl_id, self.ADDR_TEMPERATURE)
        return temperature_list
    
    def move_to_position(self, dxl_id, destination):
        

################################################################################################
# 
################################################################################################

    def _open_port(self):
        if not self.port_handler.openPort():
            raise Exception("Failed to open the port")

    def _set_baudrate(self):
        if not self.port_handler.setBaudRate(self.baudrate):
            raise Exception("Failed to change the baudrate")

    def _write_byte(self, byteNum, dxl_id, address, value):
        if byteNum == 1:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, address, value)
        else:
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, address, value)

        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print(f"Motor {dxl_id}: Wrote {value} to {address}")

    def _read_byte(self, byteNum, dxl_id, address):
        value = -1
        if byteNum == 1:
            value, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTx(self.portHandler, dxl_id, address)
        else:
            value, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTx(self.portHandler, dxl_id, address)

        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            return -1
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            return -1
        else:
            return value