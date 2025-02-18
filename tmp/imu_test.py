import asyncio
import bleak
from imu_module import BLEIMUHandler
import threading
import time

# 測試
if __name__ == "__main__":
    handler = BLEIMUHandler()
    debug_code = handler.start_imu()
    if debug_code < 0:
        exit()
    
    try:
        while True:
            time.sleep(1)  # 每秒獲取一次數據
            rX, rY, rZ = handler.updateData()
            print(f"IMU Data: AsX={rX}, AsY={rY}, AsZ={rZ}")
    except KeyboardInterrupt:
        print("Interrupt")

    handler.stop_imu()