import time
import threading
import test_env

from module.dxl_module import DXLHandler

if __name__ == "__main__":
    try:
        dxl = DXLHandler()
        dxl.start_dxl()

        while True:
            command = input("Commands:\n<motor_id> <position>, \n't'   Check temperatures, \n'-1'   Exit: ").strip()

            if command.strip() == "-1":
                break
            elif command.lower() == 't':
                print(f"temperature: {dxl.read_temperature()}")
            else:
                inputs = command.split()
                if len(inputs) != 2:
                    print("Invalid input")
                    continue

                motor_id, position = int(inputs[0]), int(inputs[1])
                if motor_id not in dxl.DXL_IDs:
                    print("Invalid motor id")
                    continue
                if position < 0 or position > 4000:
                    print("Invalid position")
                    continue

                pos_code = dxl.move_to_position([motor_id], [position])
                if pos_code <= 0:
                    reset_code = dxl.move_to_position([dxl.DXL_IDs], [dxl.DXL_INIT_POS])
                    if reset_code <= 0:
                        raise Exception("[DXL] DXL is stuck")
                print(f"(destination: {position}), (current: {dxl.read_positions([motor_id])})") 
    except Exception as e:
        print(f"An error occurred: {e}")
    except KeyboardInterrupt:
        print("Program interrupted.")
    finally:
        dxl.stop_dxl()
