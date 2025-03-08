import time
import test_env
from module.fsr_slider_module import FSRSerialReader

# 初始化 FSRSerialReader
fsr = FSRSerialReader(port='COM5', baudrate=115200, threshold=50)

# 啟動數據收集
fsr.start_collection()

try:
    while True:  # grab data 10 times
        time.sleep(1)  # wait 1 sec
        force_A0, force_A1, force_A2 = fsr.get_fsr()
        print(f"A0: {force_A0}, A1: {force_A1}, A2: {force_A2}")
        # user_input = input("Enter a number (75-145) to move to that position: ")
        # respond = fsr.send_slider_position(user_input)
        # print(respond)
except KeyboardInterrupt:
    print("Interrupt")

# 停止數據收集
fsr.stop_collection()

# 畫圖
fsr.plot_data()
