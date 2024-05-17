# Imports
from __future__ import unicode_literals

import os

import moviepy.editor as mp
from download_utils import extract_and_store_audio, split_video, transcribe_audio_files
from yt_dlp import YoutubeDL

import whisper


def preprocess_video(
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
    path_to_data = os.path.join(basepath, "data/raw", name)
    try:
        os.makedirs(path_to_data, exist_ok=True)
        print("Starting AutoCaptioning...")
        print(f"Results will be stored in data/{name}")
        video_chunks_dir = os.path.join(path_to_data, "video_chunks")
        audio_chunks_dir = os.path.join(path_to_data, "audio_chunks")
        transcriptions_dir = os.path.join(path_to_data, "transcriptions")

        os.makedirs(video_chunks_dir, exist_ok=True)
        os.makedirs(audio_chunks_dir, exist_ok=True)
        os.makedirs(transcriptions_dir, exist_ok=True)
        print("Created chunks folders")

    except Exception as e:
        return print(e)

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

    print("Splitting starts:")

    # Call the split function
    split_length = 30

    split_video(
        filename=f"{path_to_data}/{input_file}",
        split_length=split_length,
        output_dir=video_chunks_dir,
        vcodec="copy",
        acodec="copy",
    )
    extract_and_store_audio(video_chunks_dir, audio_chunks_dir)
    print("Transcriptions starts")

    # Instantiate whisper model using model_type variable

    model = whisper.load_model(model_type)

    transcribe_audio_files(audio_chunks_dir, transcriptions_dir, model)


def main():
    opts_aud = {"format": "mp3/bestaudio/best", "keep-video": True}
    opts_vid = {"format": "mp4/bestvideo/best"}

    # INSERT Youtube URL
    URL = "https://www.youtube.com/watch?v=r11Lr4FILX8"

    # INSERT video name here
    name = "hitched"

    preprocess_video(
        download=True,
        uploaded_vid="dune.mp4",  # path to local file
        url=URL,
        name=name,
        aud_opts=opts_aud,
        vid_opts=opts_vid,  # Video download settings
        model_type="small",  # change to 'large' if you want more accurate results,
        # change to 'medium.en' or 'large.en' for all english language tasks,
        # and change to 'small' or 'base' for faster inference
        audio_file=name + ".mp3",
        input_file=name + ".mp4",
        output="output.mp4",
        lang="english",
    )


if __name__ == "__main__":
    main()
