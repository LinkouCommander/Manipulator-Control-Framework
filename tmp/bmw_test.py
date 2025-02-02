from imutils.video import VideoStream
import cv2
import time
from bmw_class import BallTracker

def main():
    tracker = BallTracker(buffer_size=64, height_threshold=300, alpha=0.2)
    vs = VideoStream(src=0).start()
    time.sleep(2.0)  # Allow the camera to warm up

    try:
        while True:
            frame = vs.read()
            frame, lifting_reward = tracker.track_ball(frame)  # Process the frame with the tracker

            cv2.imshow("Ball Tracking", frame)
            print("Lifting Reward:", lifting_reward)  # Print the lifting reward

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # Press 'q' to exit
                break
    finally:
        cv2.destroyAllWindows()
        vs.stop()

if __name__ == "__main__":
    main()