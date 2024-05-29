import os

import cv2

# Set the path to the video file

# get current directory
current_directory = os.getcwd() + "/dataset"

video_path = os.path.join(current_directory, "anatomy_vid.mp4")

# Set the timestamp (in seconds) where the keyframe should be extracted
timestamp = 10  # Example: Extract frame at 10 seconds

# Interval in seconds between frames to extract
interval = 10

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the total number of seconds in the video
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))

# # Loop through the video at intervals of 10 seconds
# for timestamp in range(0, video_length, interval):
#     # Check if frame was successfully read
#     if ret:
#         # Save the frame as an image file
#         cv2.imwrite(f"frame_at_{timestamp}_seconds.jpg", frame)
#         print(f"Frame at {timestamp} seconds saved successfully.")
#     else:
#         print(f"Error: Could not read frame at {timestamp} seconds.")

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
