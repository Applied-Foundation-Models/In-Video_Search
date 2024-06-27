import argparse
import os
from subprocess import call

from loguru import logger

# pip install --upgrade scenedetect[opencv]
# https://pyscenedetect.readthedocs.io/en/latest/download/


def detect_scenes(file_path):
    """
    Detects scenes in video files located in the specified directory.

    This function walks through the directory specified by `file_path` and searches for video files with the ".mp4" extension.
    For each video file found, it runs the `scenedetect` command-line tool to detect scenes in the video.
    The detected scenes are saved as separate images and the video is split into multiple segments based on the detected scenes.

    Note:
    - The `scenedetect` command-line tool must be installed and accessible in the system's PATH.
    - The `file_path` variable should be set to the directory containing the video files.

    Returns:
    None
    """
    for dirpath, subdirs, files in os.walk(file_path):
        for file in files:
            if file.endswith(".mp4"):
                logger.info("Found file")
                name = file
                video_path = os.path.join(file_path, file)

                logger.info(f"Name:{name},dirname:{video_path}")
                try:
                    logger.info("Running scene_detection:")
                    run_scene = "scenedetect -i {video_path} -o {scenes}  -s {video_path_wo_suffix}.stats.csv list-scenes detect-content -t {treshhold} save-images -n {image_number} split-video -o {video_output_dir}".format(
                        video_path=video_path,
                        scenes=os.path.join(file_path, "extracted_keyframes"),
                        video_path_wo_suffix=file[:-4],
                        treshhold=5,
                        image_number=1,
                        video_output_dir=os.path.join(file_path, "scene_snippets"),
                    )
                    call(run_scene, shell=True)

                except Exception as e:
                    logger.exception(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-file_path",
        "--file_path",
        help="base folder path of dataset and csv",
        default="/data/",
    )  # TODO
    args = parser.parse_args()

    file_path = args.file_path
    detect_scenes(file_path)
