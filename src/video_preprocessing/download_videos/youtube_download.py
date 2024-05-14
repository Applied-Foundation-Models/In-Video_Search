# Imports
from __future__ import unicode_literals

import os

import cv2
import moviepy.editor as mp
import pandas as pd
import yt_dlp
from moviepy.editor import *
from moviepy.editor import VideoFileClip
from moviepy.video.tools.subtitles import SubtitlesClip
from yt_dlp import YoutubeDL

import whisper


def subtitle_video(
    download,
    url,
    aud_opts,
    vid_opts,
    model_type,
    name,
    audio_file,
    input_file,
    output,
    lang,
    uploaded_vid=None,
):
    # ------------------------------------------------------------------------------------------------------------------------------
    #     Params:
    # ------------------------------------------------------------------------------------------------------------------------------
    #     download:      bool, this tells your function if you are downloading a youtube video
    #     url: str,      str, the URL of youtube video to download if download is True
    #     aud_opts:      dict, audio file youtube-dl options
    #     vid_opts:      dict, video file youtube-dl options
    #     model_type:    str, which pretrained model to download. Options are:
    #                    ['tiny', 'small', 'base', 'medium','large','tiny.en', 'small.en', 'base.en', 'medium.en']
    #                    More details about model_types can be found in table in original repo here:
    #                    https://github.com/openai/whisper#Available-models-and-languages
    # .    name:          str, name of directory to store files in in experiments_download folder
    #     audio_file:    str, path to extracted audio file for Whisper
    #     input_file:    str, path to video file for MoviePy to caption
    #     output:        str, destination of final output video file
    #     uploaded_vid:  str, path to uploaded video file if download is False
    #
    # --------------------------------------------------------------------------------------------------------------------------------
    #     Returns:       An annotated video with translated captions into english, saved to name/output
    # --------------------------------------------------------------------------------------------------------------------------------

    # First, this checks if your expermiment name is taken. If not, it will create the directory.
    # Otherwise, we will be prompted to retry with a new name
    basepath = os.getcwd()
    path_to_data = os.path.join(basepath, "data", name)
    try:
        os.makedirs(path_to_data, exist_ok=True)
        print("Starting AutoCaptioning...")
        print(f"Results will be stored in data/{name}")

    except Exception as e:
        return print(e)

    # Use audio and video options for youtube-dl if downloading from youtube
    vid_opts["outtmpl"] = f"{path_to_data}/{input_file}"
    aud_opts["outtmpl"] = f"{path_to_data}/{audio_file}"

    URLS = [url]
    if download:
        with YoutubeDL(aud_opts) as ydl:
            ydl.download(url)
        with YoutubeDL(vid_opts) as ydl:
            ydl.download(URLS)
    else:
        # Use local clip if not downloading from youtube
        my_clip = mp.VideoFileClip(uploaded_vid)
        my_clip.write_videofile(f"{path_to_data}/{input_file}")
        my_clip.audio.write_audiofile(f"{path_to_data}/{audio_file}")

    # Instantiate whisper model using model_type variable
    model = whisper.load_model(model_type)

    # Get text from speech for subtitles from audio file
    result = model.transcribe(
        f"""{path_to_data}/{audio_file}""", task="translate", language=lang
    )

    print(f"Result output format{result}")

    # create Subtitle dataframe, and save it
    dict1 = {"start": [], "end": [], "text": []}
    for i in result["segments"]:
        dict1["start"].append(int(i["start"]))
        dict1["end"].append(int(i["end"]))
        dict1["text"].append(i["text"])

    print(f"Transcription in dict format: {dict1}")
    df = pd.DataFrame.from_dict(dict1)
    df.to_csv(f"{path_to_data}/subs.csv")
    vidcap = cv2.VideoCapture(f"""{path_to_data}/{input_file}""")
    success, image = vidcap.read()
    height = image.shape[0]
    width = image.shape[1]

    # Instantiate MoviePy subtitle generator with TextClip, subtitles, and SubtitlesClip
    def generator(txt):
        return TextClip(
            txt,
            font="P052-Bold",
            fontsize=width / 50,
            stroke_width=0.7,
            color="white",
            stroke_color="black",
            size=(width, height * 0.25),
            method="caption",
        )

    # generator = lambda txt: TextClip(txt, color='white', fontsize=20, font='Georgia-Regular',stroke_width=3, method='caption', align='south', size=video.size)
    subs = tuple(
        zip(tuple(zip(df["start"].values, df["end"].values)), df["text"].values)
    )
    subtitles = SubtitlesClip(subs, generator)

    # Ff the file was on youtube, add the captions to the downloaded video
    if download:
        video = VideoFileClip(f"{path_to_data}/{input_file}")
        final = CompositeVideoClip([video, subtitles.set_pos(("center", "bottom"))])
        final.write_videofile(
            f"{path_to_data}/{output}",
            fps=video.fps,
            remove_temp=True,
            codec="libx264",
            audio_codec="aac",
        )
    else:
        # If the file was a local upload:
        video = VideoFileClip(uploaded_vid)
        final = CompositeVideoClip([video, subtitles.set_pos(("center", "bottom"))])
        final.write_videofile(
            f"{path_to_data}/{output}",
            fps=video.fps,
            remove_temp=True,
            codec="libx264",
            audio_codec="aac",
        )


def main():
    # Options for youtube download to ensure we get a high quality audio file extraction.
    # This is key, as extracting from the video in the same download seemed to significantly affect caption Word Error Rate in our experiments_download.
    # Only modify these if needed. Lowered audio quality may inhibit the transcription's word error rate.
    opts_aud = {"format": "mp3/bestaudio/best", "keep-video": True}
    # Options for youtube video to get right video file for final output
    opts_vid = {"format": "mp4/bestvideo/best"}

    # INSERT Youtube URL
    URL = "https://www.youtube.com/watch?v=xFqpUWtuRZM"

    # INSERT video name here
    name = "your-filename"
    subtitle_video(
        download=True,
        uploaded_vid="dune.mp4",  # path to local file
        url=URL,
        name=name,
        aud_opts=opts_aud,
        vid_opts=opts_vid,  # Video download settings
        model_type="medium.en",  # change to 'large' if you want more accurate results,
        # change to 'medium.en' or 'large.en' for all english language tasks,
        # and change to 'small' or 'base' for faster inference
        audio_file="audio.mp3",
        input_file="dune.mp4", # TODO: Edit file_name here
        output="output.mp4",
        lang="english",
    )


if __name__ == "__main__":
    main()
