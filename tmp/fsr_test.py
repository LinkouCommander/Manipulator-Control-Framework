import time
from fsr_class import FSRSerialReader

# 初始化 FSRSerialReader
fsr = FSRSerialReader(port='COM4', baudrate=115200, threshold=50)

# 啟動數據收集
fsr.start_collection()

try:
    for _ in range(10):  # grab data 10 times
        time.sleep(1)  # wait 1 sec
        force_A0, force_A1, force_A2 = fsr.get_fsr()
        print(f"A0: {force_A0}, A1: {force_A1}, A2: {force_A2}")
except KeyboardInterrupt:
    print("Interrupt")

# 停止數據收集
fsr.stop_collection()

# 畫圖
fsr.plot_data()
