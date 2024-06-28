from __future__ import print_function

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
from multiprocessing import Pool

import pandas as pd
from loguru import logger
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from tqdm import tqdm

import whisper

# The rest of the functions remain unchanged

# Removed the main() and if __name__ == "__main__" block


def split_by_manifest(
    filename,
    manifest,
    output_dir,
    vcodec="copy",
    acodec="copy",
    extra="",
    **kwargs,
):
    """Split video into segments based on the given manifest file.

    Arguments:
        filename (str)      - Location of the video.
        manifest (str)      - Location of the manifest file.
        vcodec (str)        - Controls the video codec for the ffmpeg video
                            output.
        acodec (str)        - Controls the audio codec for the ffmpeg video
                            output.
        extra (str)         - Extra options for ffmpeg.
    """
    if not os.path.exists(manifest):
        logger.error("File does not exist: %s" % manifest)
        raise SystemExit

    with open(manifest) as manifest_file:
        manifest_type = manifest.split(".")[-1]
        if manifest_type == "json":
            config = json.load(manifest_file)
        elif manifest_type == "csv":
            config = csv.DictReader(manifest_file)
        else:
            logger.error("Format not supported. File must be a csv or json file")
            raise SystemExit

        split_cmd = [
            "ffmpeg",
            "-i",
            filename,
            "-vcodec",
            vcodec,
            "-acodec",
            acodec,
            "-y",
        ] + shlex.split(extra)
        try:
            fileext = filename.split(".")[-1]
        except IndexError as e:
            raise IndexError("No . in filename. Error: " + str(e))
        for video_config in config:
            split_args = []
            try:
                split_start = video_config["start_time"]
                split_length = video_config.get("end_time", None)
                if not split_length:
                    split_length = video_config["length"]
                filebase = video_config["rename_to"]
                if fileext in filebase:
                    filebase = ".".join(filebase.split(".")[:-1])

                split_args += [
                    "-ss",
                    str(split_start),
                    "-t",
                    str(split_length),
                    filebase + "." + fileext,
                ]
                logger.info("########################################################")
                logger.info("About to run: " + " ".join(split_cmd + split_args))
                logger.info("########################################################")
                subprocess.check_output(split_cmd + split_args)
            except KeyError as e:
                logger.error("############# Incorrect format ##############")
                if manifest_type == "json":
                    logger.error("The format of each json array should be:")
                    logger.error(
                        "{start_time: <int>, length: <int>, rename_to: <string>}"
                    )
                elif manifest_type == "csv":
                    logger.error(
                        "start_time,length,rename_to should be the first line "
                    )
                    logger.error("in the csv file.")
                logger.error(e)
                raise SystemExit


def get_video_length(filename):
    output = subprocess.check_output(
        (
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filename,
        )
    ).strip()
    video_length = int(float(output))
    logger.info("Video length in seconds: " + str(video_length))

    return video_length


def ceildiv(a, b):
    return int(math.ceil(a / float(b)))


def split_by_seconds(
    filename,
    split_length,
    output_dir,
    vcodec="copy",
    acodec="copy",
    extra="",
    video_length=None,
    **kwargs,
):
    if split_length and split_length <= 0:
        logger.info("Split length can't be 0")
        raise SystemExit

    if not video_length:
        video_length = get_video_length(filename)
    split_count = ceildiv(video_length, split_length)
    if split_count == 1:
        logger.info("Video length is less then the target split length.")
        raise SystemExit

    split_cmd = [
        "ffmpeg",
        "-i",
        filename,
        "-vcodec",
        vcodec,
        "-acodec",
        acodec,
    ] + shlex.split(extra)
    # Ensure the output directory exists

    try:
        filebase = filename.split("/")[-1].split(".")[0]
        filebase = os.path.join(output_dir, filebase)
        fileext = filename.split(".")[-1]
    except IndexError as e:
        raise IndexError("No . in filename. Error: " + str(e))

    for n in range(0, split_count):
        split_args = []
        if n == 0:
            split_start = 0
        else:
            split_start = split_length * n

        output_filename = f"{filebase}-{n + 1}-of-{split_count}.{fileext}"
        output_filepath = os.path.join(output_dir, output_filename)

        split_args += [
            "-ss",
            str(split_start),
            "-t",
            str(split_length),
            output_filepath,
        ]
        logger.info("About to run: " + " ".join(split_cmd + split_args))
        subprocess.check_output(split_cmd + split_args)


def split_video(filename, split_length=None, manifest=None, output_dir=None, **kwargs):
    """
    Entry function to split video either by seconds or using a manifest file.
    Args:
        filename (str): Path to the video file.
        split_length (int, optional): Length of each segment in seconds.
        manifest (str, optional): Path to a JSON or CSV manifest file.
        output_dir (str, optional): Path to put the chunks in.
        **kwargs: Additional keyword arguments for video codec settings.
    """
    if manifest:
        split_by_manifest(filename, manifest, output_dir, **kwargs)
    else:
        if not split_length:
            video_length = get_video_length(filename)
            split_by_seconds(
                filename, split_length, output_dir, video_length=video_length, **kwargs
            )
        else:
            split_by_seconds(filename, split_length, output_dir, **kwargs)


def split_mp3(input_file, output_dir, split_length):
    """
    Split an MP3 file into multiple snippets of a specified length.

    Args:
        input_file (str): The path to the input MP3 file.
        output_dir (str): The directory where the snippets will be saved.
        split_length (int): The length of each snippet in seconds.

    Returns:
        None
    """

    logger.info(f"Output_dir_audio: {output_dir}")
    # os.chmod(output_dir, 0o777)
    # os.makedirs(output_dir, exist_ok=True)

    # Load the input MP3 file
    audio = AudioSegment.from_mp3(input_file)

    # Calculate the length of snippets in milliseconds
    snippet_length_ms = split_length * 1000

    # Initialize the starting and ending positions for slicing
    start_time = 0
    end_time = snippet_length_ms

    logger.info(f"Audio length: {len(audio)}")
    logger.info(f"Audio duration: {audio.duration_seconds}")

    logger.info(f"End time: {end_time}")

    # Initialize snippet count
    snippet_count = 1

    while end_time <= len(audio):
        # Extract the snippet
        snippet = audio[start_time:end_time]
        logger.info(f"Snippet: {snippet}")

        # Define the output file name
        output_file = os.path.join(output_dir, f"snippet_{snippet_count}.mp3")

        # Export the snippet as an MP3 file
        snippet.export(output_file, format="mp3")

        # Update start and end times for the next snippet
        start_time = end_time
        logger.info(f"Start time:{start_time}")
        end_time = start_time + snippet_length_ms
        logger.info(f"Start time:{end_time}")

        # Increment the snippet count
        snippet_count += 1

    logger.info(f"Split {snippet_count - 1} snippets from {input_file}.")


def extract_and_store_audio(video_dir, audio_dir):
    """
    Extract audio from each video in the video_dir and store it in audio_dir.
    """
    os.makedirs(audio_dir, exist_ok=True)  # Ensure the audio directory exists
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

    for video_file in tqdm(video_files):
        video_path = os.path.join(video_dir, video_file)
        audio_path = os.path.join(audio_dir, video_file.replace(".mp4", ".wav"))

        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec="pcm_s16le", fps=44100)
        logger.info(f"Audio extracted and saved as {audio_path}")


# Speed up using hugging_face-whisper: https://www.reddit.com/r/MachineLearning/comments/14xxg6i/d_what_is_the_most_efficient_version_of_openai/
# def transcribe_audio_files(
#     audio_dir, transcriptions_dir, model_type="small", lang="en"
# ):
#     """
#     Transcribe each audio file in the audio_dir and save transcriptions to transcriptions_dir.
#     """
#     model = whisper.load_model(model_type)
#     os.makedirs(transcriptions_dir, exist_ok=True)
#     audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

#     for audio_file in tqdm(audio_files):
#         audio_path = os.path.join(audio_dir, audio_file)
#         result = model.transcribe(audio_path, task="translate", language=lang)
#         # Create Subtitle dataframe and save it
#         dict1 = {"start": [], "end": [], "text": []}
#         for segment in result["segments"]:
#             dict1["start"].append(int(segment["start"]))
#             dict1["end"].append(int(segment["end"]))
#             dict1["text"].append(segment["text"])
#         df = pd.DataFrame.from_dict(dict1)
#         df.to_csv(os.path.join(transcriptions_dir, audio_file.replace(".wav", ".csv")))
#         print(f"Transcription for {audio_file} saved in {transcriptions_dir}")
#         audio_path = os.path.join(audio_dir, audio_file)
#         result = model.transcribe(audio_path, task="translate", language=lang)

#         # Create Subtitle dataframe and save it
#         dict1 = {"start": [], "end": [], "text": []}
#         for segment in result["segments"]:
#             dict1["start"].append(int(segment["start"]))
#             dict1["end"].append(int(segment["end"]))
#             dict1["text"].append(segment["text"])

#         df = pd.DataFrame.from_dict(dict1)
#         df.to_csv(os.path.join(transcriptions_dir, audio_file.replace(".wav", ".csv")))

#         print(f"Transcription for {audio_file} saved in {transcriptions_dir}")


def transcribe_single_file(args):
    audio_dir, audio_file, transcriptions_dir, model_type, lang = args
    model = whisper.load_model(model_type)
    audio_path = os.path.join(audio_dir, audio_file)
    result = model.transcribe(audio_path, task="translate", language=lang)

    dict1 = {"start": [], "end": [], "text": []}
    for segment in result["segments"]:
        dict1["start"].append(int(segment["start"]))
        dict1["end"].append(int(segment["end"]))
        dict1["text"].append(segment["text"])

    df = pd.DataFrame.from_dict(dict1)
    df.to_csv(
        os.path.join(transcriptions_dir, audio_file.replace(".wav", ".csv")),
        index=False,
    )
    logger.info(f"Transcription for {audio_file} saved in {transcriptions_dir}")


def transcribe_audio_files(
    audio_dir, transcriptions_dir, model_type="small", lang="en"
):
    os.makedirs(transcriptions_dir, exist_ok=True)
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

    # Tuple of arguments for each file
    tasks = [
        (audio_dir, audio_file, transcriptions_dir, model_type, lang)
        for audio_file in audio_files
    ]

    # Create a pool of processes and map the tasks
    logger.info("Starting pooling:")
    with Pool(processes=os.cpu_count()) as pool:
        list(tqdm(pool.imap(transcribe_single_file, tasks), total=len(tasks)))


def transcription_to_text(keyframe, transcription_file_path, timestamp_file_path):
    # Load the CSV file into a DataFrame
    df_timestamps = pd.read_csv(timestamp_file_path, skiprows=1)

    timestamps = df_timestamps.set_index("Scene Number")[
        ["Start Time (seconds)", "End Time (seconds)"]
    ].T.to_dict("list")

    df_transcription = pd.read_csv(transcription_file_path)
    if not df_transcription.empty:
        transcription = " ".join(df_transcription["text"].astype(str))
    else:
        transcription = ""
    return transcription, timestamps[keyframe]


# def hugging_face_whisper():
#     # device = "mps" if torch.backends.mps.is_available() else "cpu"
#     device = "cpu"
#     pipe = pipeline(
#         "automatic-speech-recognition",
#         model="openai/whisper-tiny",
#         chunk_length_s=30,
#         device=device,
#     )
#     ds = load_dataset(
#         "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
#     )
#     sample = ds[0]["audio"]
#     logger.info(f"Sample:{sample}")
#     filepath = "/Users/magic-rabbit/Documents/AFM/afm-vlm/data/raw/biology_chapter_3_3/audio_chunks/biology_chapter_3_3-Scene-001.wav"
#     audio_data, sample_rate = sf.read(filepath)

#     # Prepare the audio dictionary expected by the Whisper pipeline
#     sample = {"array": audio_data, "sampling_rate": sample_rate}
#     logger.info(f"WAV sample: {sample}")

#     prediction = pipe(sample.copy(), batch_size=8, return_timestamps=True)["chunks"]
#     logger.info(prediction)


def create_metadata(
    keyframe_num,
    image_path,
    timestamps,
    transcription,
    ocr_extracted_text,
    llava_results,
    clip_llm_summary,
    extensive_summary,
    clip_text_embedding,
    clip_image_embedding,
    standard_text_embedding,
    extensive_text_embedding
):
    video_metadata = {
        "img_path": image_path,
        "timestamps": timestamps,
        "transcription": transcription,
        "ocr_extracted_text": ocr_extracted_text,
        "llava_result": llava_results,
        "clip_text": clip_llm_summary,
        "llm_long_summary": extensive_summary,
        "clip_text_embedding": clip_text_embedding,
        "clip_image_embedding": clip_image_embedding,
        "standard_text_embedding": standard_text_embedding,
        "extensive_text_embedding": extensive_text_embedding,
        "long_description_embedding": "",
    }

    return keyframe_num, video_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run components of the download utils")
    parser.add_argument(
        "--transcribe_audio",
        action="store_true",
        help="Transcribe audio files",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--transcriptions_dir",
        type=str,
        help="Directory to save transcriptions",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model for transcription",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language for transcription",
    )
    parser.add_argument(
        "--run_hugging_face",
        action="store_true",
        help="Run the Hugging Face Whisper function",
    )
    args = parser.parse_args()

    if args.transcribe_audio:
        transcribe_audio_files(
            args.audio_dir, args.transcriptions_dir, args.model, args.lang
        )

    # if args.run_hugging_face:
    #     hugging_face_whisper()
