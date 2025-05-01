# coding:UTF-8
import time
import struct
import bleak
import asyncio
import threading
import math

lock_event = threading.Event()
error_code = True

class BLEIMUHandler:
    """
    A handler for managing Bluetooth Low Energy (BLE) IMU devices.
    Handles scanning, connection, data retrieval, and safe shutdown.
    """
    def __init__(self, target_device="C2:CE:D3:39:47:43"):
        """
        Initializes the BLEIMUHandler.

        Args:
            target_device (str): The MAC address of the target IMU device.
        """
        self.devices = []
        self.target_device = None
        self.user_input = target_device # set target MAC address
        self.imu = None
        self.data_thread = None

    # Scan Bluetooth devices and filter names
    async def scan(self):
        """
        Scans for IMU devices until the target device is found.
        Sets `self.target_device` when a match is found.
        """
        print("Searching for Bluetooth devices......")
        try:
            while True:
                self.devices = await bleak.BleakScanner.discover()
                print("Searching IMU...")
                for d in self.devices:
                    if d.name is not None and d.address == self.user_input:
                        self.target_device = self.user_input
                        print(self.user_input, "is found!")
                        return

                    # Debug option to print matching names
                    # if d.name is not None and "WT" in d.name:
                    #     print(d)
                    #     return

                await asyncio.sleep(2)  # Delay before next scan attempt
        except Exception as ex:
            print("Error during Bluetooth scanning:")


    def start_imu(self):
        """
        Starts the BLE IMU by scanning and initializing the connection in a background thread.

        Raises:
            Exception: If device not found or fails to open.
        """
        # Search Device
        asyncio.run(self.scan())
        time.sleep(1) # Allow scan time to finalize

        if self.target_device is not None:
            # Create BLE device object
            self.imu = DeviceModel("MyBle5.0", self.target_device)

            # Run device setup in background thread
            self.data_thread = threading.Thread(target=self._run_device)
            self.data_thread.start()

            # Wait until device is ready
            lock_event.wait()
            if(error_code):
                raise Exception("[IMU] Failed to run openDevice()")
        else:
            raise Exception("[IMU] No Bluetooth device corresponding to Mac address found!!")

    def _run_device(self):
        """
        Private helper method that starts the device loop in a separate thread.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.imu.openDevice())

    def updateIMUData(self):
        """
        Retrieves the current IMU data from the BLE device.

        Returns:
            tuple: Angular velocity in radians for axes (AsX, AsY, AsZ).
                   Returns (0, 0, 0) if no data found.
        """
        if self.imu:
            return (
                self.imu.get("AsX") * (math.pi / 180) if self.imu.get("AsX") else 0,
                self.imu.get("AsY") * (math.pi / 180) if self.imu.get("AsY") else 0,
                self.imu.get("AsZ") * (math.pi / 180) if self.imu.get("AsZ") else 0
            )
        else:
            return (0, 0, 0)  # Return 0 if no IMU
        
    def stop_imu(self):
        """
        Safely stops the IMU device and joins the background data thread.
        """
        if self.imu is not None:
            self.imu.closeDevice()
        if self.data_thread:
            self.data_thread.join()

# Device instance
class DeviceModel:
    # region Attributes
    # Device name
    deviceName = "My Device"

    # Device data dictionary
    deviceData = {}

    # Is the device open
    isOpen = False

    # Temporary array
    TempBytes = []

    # endregion

    def __init__(self, deviceName, mac):
        print("Initializing device model")
        # Device Name (custom)
        self.deviceName = deviceName
        self.mac = mac
        self.client = None
        self.writer_characteristic = None
        self.isOpen = False
        self.deviceData = {}

    # region Obtain device data
    # Set device data
    def set(self, key, value):
        # Save device data to key values
        self.deviceData[key] = value

    # Obtain device data
    def get(self, key):
        # Get data from key values, return None if not found
        if key in self.deviceData:
            return self.deviceData[key]
        else:
            return None

    # Delete device data
    def remove(self, key):
        # Delete device key value
        del self.deviceData[key]

    # endregion

    # Open device
    async def openDevice(self):
        global error_code
        print("Opening device......")
        try:
            # Obtain the services and characteristic of the device
            async with bleak.BleakClient(self.mac) as client:
                self.client = client
                self.isOpen = True
                # Device UUID constant
                target_service_uuid = "0000ffe5-0000-1000-8000-00805f9a34fb"
                target_characteristic_uuid_read = "0000ffe4-0000-1000-8000-00805f9a34fb"
                target_characteristic_uuid_write = "0000ffe9-0000-1000-8000-00805f9a34fb"
                notify_characteristic = None

                print("Matching services......")
                for service in client.services:
                    if service.uuid == target_service_uuid:
                        print(f"Service: {service}")
                        print("Matching characteristic......")
                        for characteristic in service.characteristics:
                            if characteristic.uuid == target_characteristic_uuid_read:
                                notify_characteristic = characteristic
                            if characteristic.uuid == target_characteristic_uuid_write:
                                self.writer_characteristic = characteristic
                        if notify_characteristic:
                            break

                if notify_characteristic:
                    print(f"Characteristic: {notify_characteristic}")
                    error_code = False
                    lock_event.set()  # Notify start_imu() can return

                    # Set up notifications to receive data
                    await client.start_notify(notify_characteristic.uuid, self.onDataReceived)

                    # Keep connected and open
                    try:
                        while self.isOpen:
                            await asyncio.sleep(1)
                    except asyncio.CancelledError:
                        pass
                    finally:
                        # Stop notification on exit
                        await client.stop_notify(notify_characteristic.uuid)
                else:
                    print("No matching services or characteristic found")
                    lock_event.set()  # Notify start_imu() can return
        except Exception as ex:
            print(f"Failed to open device: {ex}")
            # add terminate policy
            lock_event.set()

    # Close device
    def closeDevice(self):
        self.isOpen = False
        print("The device is turned off")

    # region Data analysis
    # Serial port data processing
    def onDataReceived(self, sender, data):
        tempdata = bytes.fromhex(data.hex())
        for var in tempdata:
            self.TempBytes.append(var)
            if len(self.TempBytes) == 2 and (self.TempBytes[0] != 0x55 or self.TempBytes[1] != 0x61):
                del self.TempBytes[0]
                continue
            if len(self.TempBytes) == 20:
                self.processData(self.TempBytes[2:])
                self.TempBytes.clear()

    # Data analysis
    def processData(self, Bytes):
        Ax = self.getSignInt16(Bytes[1] << 8 | Bytes[0]) / 32768 * 16
        Ay = self.getSignInt16(Bytes[3] << 8 | Bytes[2]) / 32768 * 16
        Az = self.getSignInt16(Bytes[5] << 8 | Bytes[4]) / 32768 * 16
        Gx = self.getSignInt16(Bytes[7] << 8 | Bytes[6]) / 32768 * 2000
        Gy = self.getSignInt16(Bytes[9] << 8 | Bytes[8]) / 32768 * 2000
        Gz = self.getSignInt16(Bytes[11] << 8 | Bytes[10]) / 32768 * 2000
        AngX = self.getSignInt16(Bytes[13] << 8 | Bytes[12]) / 32768 * 180
        AngY = self.getSignInt16(Bytes[15] << 8 | Bytes[14]) / 32768 * 180
        AngZ = self.getSignInt16(Bytes[17] << 8 | Bytes[16]) / 32768 * 180
        self.set("AccX", round(Ax, 3))
        self.set("AccY", round(Ay, 3))
        self.set("AccZ", round(Az, 3))
        self.set("AsX", round(Gx, 3))
        self.set("AsY", round(Gy, 3))
        self.set("AsZ", round(Gz, 3))
        self.set("AngX", round(AngX, 3))
        self.set("AngY", round(AngY, 3))
        self.set("AngZ", round(AngZ, 3))

    # Obtain int16 signed number
    @staticmethod
    def getSignInt16(num):
        if num >= pow(2, 15):
            num -= pow(2, 16)
        return num

    # endregion

    # Sending serial port data
    def sendData(self, data):
        try:
            if self.client is not None and self.writer_characteristic is not None:
                self.client.write_value(self.writer_characteristic.uuid, data)
        except Exception as ex:
            print(ex)

    # Read register
    def readReg(self, regAddr):
        # Encapsulate read instructions and send data to the serial port
        self.sendData(self.get_readBytes(regAddr))

    # Write Register
    def writeReg(self, regAddr, sValue):
        # Unlock
        self.unlock()
        # Delay 100ms
        time.sleep(0.1)
        # Encapsulate write instructions and send data to the serial port
        self.sendData(self.get_writeBytes(regAddr, sValue))
        # Delay 100ms
        time.sleep(0.1)
        # Save
        self.save()

    # Read instruction encapsulation
    @staticmethod
    def get_readBytes(regAddr):
        # Initialize
        tempBytes = [None] * 5
        tempBytes[0] = 0xff
        tempBytes[1] = 0xaa
        tempBytes[2] = 0x01
        tempBytes[3] = regAddr
        tempBytes[4] = tempBytes[0] + tempBytes[1] + tempBytes[2] + tempBytes[3]
        return bytes(tempBytes)

    # Write instruction encapsulation
    @staticmethod
    def get_writeBytes(regAddr, sValue):
        # Initialize
        tempBytes = [None] * 6
        tempBytes[0] = 0xff
        tempBytes[1] = 0xaa
        tempBytes[2] = 0x02
        tempBytes[3] = regAddr
        tempBytes[4] = sValue
        tempBytes[5] = tempBytes[0] + tempBytes[1] + tempBytes[2] + tempBytes[3] + tempBytes[4]
        return bytes(tempBytes)

    # Unlock
    @staticmethod
    def unlock():
        pass

    # Save
    def save(self):
        cmd = self.get_writeBytes(0x00, 0x0000)
        self.sendData(cmd)