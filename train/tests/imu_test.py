import asyncio
import bleak
import threading
import time
import test_env
from module.imu_module import BLEIMUHandler

# 測試
if __name__ == "__main__":
    handler = BLEIMUHandler()
    debug_code = handler.start_imu()
    if debug_code < 0:
        exit()
    
    try:
        while True:
            time.sleep(0.2)  # 每秒獲取一次數據
            rX, rY, rZ = handler.updateIMUData()
            print(f"IMU Data: AsX={rX}, AsY={rY}, AsZ={rZ}")
    except KeyboardInterrupt:
        print("Interrupt")

    handler.stop_imu()