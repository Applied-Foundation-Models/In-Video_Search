# Imports
from __future__ import unicode_literals

import argparse
import os

import moviepy.editor as mp
from loguru import logger
from src.video_preprocessing.download_videos.download_utils import split_video
from yt_dlp import YoutubeDL


def preprocess_video(
    download,
    url,
    aud_opts,
    vid_opts,
    name,
    audio_file,
    input_file,
    output,
    split_length=None,
    uploaded_vid=None,
):
    """
    Preprocesses a video by downloading it from YouTube or using a local clip,
    splitting it into chunks if specified, and storing the results in the appropriate directories.

    Args:
        download (bool): Whether to download the video from YouTube or use a local clip.
        url (str): The URL of the YouTube video to download.
        aud_opts (dict): Options for downloading the audio using youtube-dl.
        vid_opts (dict): Options for downloading the video using youtube-dl.
        name (str): The name of the video.
        audio_file (str): The filename of the downloaded audio file.
        input_file (str): The filename of the downloaded video file.
        output (str): The output directory for storing the results.
        split_length (int, optional): The length (in seconds) to split the video into chunks. Defaults to None.
        uploaded_vid (str, optional): The path to the local video clip to use if not downloading from YouTube. Defaults to None.

    Returns:
        str: The path to the directory where the video and audio chunks, as well as transcriptions, are stored.
    """
    basepath = os.getcwd()
    path_to_data = os.path.join(basepath, "data/raw", name)
    try:
        os.makedirs(path_to_data, exist_ok=True)
        logger.info("Starting AutoCaptioning...")
        logger.info(f"Results will be stored in data/raw/{name}")
        video_chunks_dir = os.path.join(path_to_data, "video_chunks")
        audio_chunks_dir = os.path.join(path_to_data, "audio_chunks")
        transcriptions_dir = os.path.join(path_to_data, "transcriptions")

        os.makedirs(video_chunks_dir, exist_ok=True)
        os.makedirs(audio_chunks_dir, exist_ok=True)
        os.makedirs(transcriptions_dir, exist_ok=True)
        logger.info("Created chunks folders")

    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        return

    # Use audio and video options for youtube-dl if downloading from youtube
    vid_opts["outtmpl"] = f"{path_to_data}/{input_file}"
    aud_opts["outtmpl"] = f"{path_to_data}/{audio_file}"

    URLS = [url]
    if download:
        # with YoutubeDL(aud_opts) as ydl:
        #     ydl.download(url)
        with YoutubeDL(vid_opts) as ydl:
            ydl.download(URLS)
    else:
        # Use local clip if not downloading from youtube
        my_clip = mp.VideoFileClip(uploaded_vid)
        my_clip.write_videofile(f"{path_to_data}/{input_file}")
        # my_clip.audio.write_audiofile(f"{path_to_data}/{audio_file}")

    if split_length is not None:
        logger.info("Splitting starts:")
        split_video(
            filename=f"{path_to_data}/{input_file}",
            split_length=split_length,
            output_dir=video_chunks_dir,
            vcodec="copy",
            acodec="copy",
        )
        # extract_and_store_audio(video_chunks_dir, audio_chunks_dir)
    else:
        logger.info("Video is not splitted:")

        # extract_and_store_audio(path_to_data, audio_chunks_dir)

    logger.info("Video downloaded successfully!")

    return path_to_data


def main():
    parser = argparse.ArgumentParser(
        description="Download and process a video from YouTube."
    )
    parser.add_argument(
        "-n", "--name", type=str, required=True, help="Name of the video."
    )
    parser.add_argument(
        "-url",
        "--youtube_url",
        type=str,
        required=True,
        help="YouTube URL of the video.",
    )
    parser.add_argument(
        "-ch",
        "--chunks",
        type=int,
        default=None,
        help="Split size in seconds if not specified, will not be splitted",
    )

    args = parser.parse_args()

    opts_aud = {"format": "mp3/bestaudio/best", "keep-video": True}
    opts_vid = {"format": "mp4/bestvideo/best"}

    # INSERT video name here
    name = args.name
    url = args.youtube_url
    chunks = args.chunks

    _ = preprocess_video(
        download=True,
        uploaded_vid="dune.mp4",  # path to local file
        url=url,
        name=name,
        aud_opts=opts_aud,
        vid_opts=opts_vid,  # Video download settings
        audio_file=name + ".mp3",
        input_file=name + ".mp4",
        output="output.mp4",
        split_length=chunks,
    )


if __name__ == "__main__":
    main()
