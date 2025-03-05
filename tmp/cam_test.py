# from imutils.video import VideoStream
import cv2
import time
import threading
from cam_module import BallTracker

def collect_rewards(tracker):
    while True:
        time.sleep(1)
        total_rewards, lifting_rewards, rotation_rewards = tracker.get_rewards()
        print("Reward:", total_rewards, ", Lifting Reward:", lifting_rewards, ", Rotation Reward:", rotation_rewards)

def main():
    tracker = BallTracker(buffer_size=64, height_threshold=300, alpha=0.2)
    tracker.start_cam()
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)  # Allow the camera to warm up

    # reward_threading = threading.Thread(target=collect_rewards, args=(tracker,), daemon=True)
    # reward_threading.start()

    try:
        while True:
            frame = tracker.get_frame()
            cv2.imshow("Ball Tracking", frame)

            l, r = tracker.get_rewards()

            print("Reward:", l, r)

            # frame = vs.read()
            # frame, lifting_reward, velocity = tracker.track_ball(frame)  # Process the frame with the tracker

            # cv2.imshow("Ball Tracking", frame)
            # print("Lifting Reward:", lifting_reward, ", Velocity:", velocity)  # Print the lifting reward

            # total_reward = tracker.get_rewards()
            # print("Reward:", total_reward)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # Press 'q' to exit
                break
    finally:
        cv2.destroyAllWindows()
        tracker.stop_cam()
        # vs.stop()

if __name__ == "__main__":
    main()