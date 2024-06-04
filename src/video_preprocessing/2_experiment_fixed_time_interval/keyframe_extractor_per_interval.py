import argparse
import os
from loguru import logger
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file at specified intervals."
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the video file."
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        default=10,
        help="Timestamp in seconds where the keyframe should be extracted. Default is 10 seconds.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Interval in seconds between frames to extract. Default is 10 seconds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Get the current directory and construct the video path
    current_directory = os.getcwd() + "/dataset"
    video_path = os.path.join(current_directory, args.video_path)

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if camera opened successfully
    if not cap.isOpened():
        logger.error("Error: Could not open video.")
        exit()

    # Get the total number of seconds in the video
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))

    # Loop through the video at the specified interval
    for timestamp in range(args.timestamp, video_length, args.interval):
        cap.set(
            cv2.CAP_PROP_POS_MSEC, timestamp * 1000
        )  # Set the position in milliseconds
        ret, frame = cap.read()
        if ret:
            # Save the frame as an image file
            cv2.imwrite(f"frame_at_{timestamp}_seconds.jpg", frame)
            logger.info(f"Frame at {timestamp} seconds saved successfully.")
        else:
            logger.error(f"Error: Could not read frame at {timestamp} seconds.")

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
