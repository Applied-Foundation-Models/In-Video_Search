import argparse
import os
from subprocess import call

# pip install --upgrade scenedetect[opencv]
# https://pyscenedetect.readthedocs.io/en/latest/download/

parser = argparse.ArgumentParser()
parser.add_argument(
    "-base_path",
    "--base_path",
    help="base folder path of dataset and csv",
    default="/data/",
)  # TODO
args = parser.parse_args()


BASE_PATH = args.base_path

for dirpath, subdirs, files in os.walk(BASE_PATH):
    for file in files:
        if file.endswith(".mp4"):
            print("Found file")
            name = file
            video_path = os.path.join(BASE_PATH, file)

            print(f"name:{name},dirname:{video_path}")
            try:
                run_scene = "scenedetect -i {video_path} -o {scenes}  -s {video_path_wo_suffix}.stats.csv list-scenes detect-content -t {treshhold} save-images -n {image_number} split-video -o {video_output_dir}".format(
                    video_path=video_path,
                    scenes=os.path.join(BASE_PATH, "scenes"),
                    video_path_wo_suffix=file[:-4],
                    treshhold=2,
                    image_number=1,
                    video_output_dir=os.path.join(BASE_PATH, "videos"),
                )
                res1 = call(run_scene, shell=True)
            except Exception as e:
                print(e)
