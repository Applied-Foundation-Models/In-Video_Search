import cv2
import numpy as np
import argparse

def extract_keyframes(video_path, threshold):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Cannot read video file")
        return

    frame_idx = 0
    keyframe_idx = 0

    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frames to grayscale to simplify the difference calculation
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference between current frame and previous frame
        diff = cv2.absdiff(gray_frame, gray_prev)
        # Normalize to [0,1] range
        non_zero_count = np.sum(diff > 50) / diff.size

        # If change between frames is greater than threshold, save the frame as a keyframe
        if non_zero_count > threshold:
            keyframe_path = f"keyframe_{keyframe_idx}.jpg"
            cv2.imwrite(keyframe_path, frame)
            print(f"Keyframe saved at {keyframe_path}")
            keyframe_idx += 1

        # Update previous frame
        prev_frame = frame
        frame_idx += 1

    cap.release()
    print("Done extracting keyframes")

def main():
    parser = argparse.ArgumentParser(description="Extract keyframes from video based on frame differences.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--threshold", type=float, default=0.2, help="Threshold for detecting keyframe (default: 0.2).")

    args = parser.parse_args()

    extract_keyframes(args.video_path, args.threshold)

if __name__ == "__main__":
    main()
