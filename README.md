# afm-vlm

--------------------
Project Organization
--------------------

    ├── data
    │   ├── input          <- Ordner für Eingabedaten
    │   ├── metadata       <- Ordner für zwischengespeicherte Metadaten.
    │   └── output         <- Ordner für Ausgabedaten.
    │
    ├── docker             <- Hier werden die Docker Container abgelegt.
    │
    ├── docs               <- Unsere Dokumenation von TOS.
    │
    ├── models             <- Von TOS genutzte Modelle.
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── src                <- Source code, um TOS zu benutzen.
    │   ├── setup.py       <- Macht das Projekt durch pip installierbar (pip install -e .).
    │   ├── scripts        <- Scripts die nicht zu TOS gehören.
    │   │   └── app               <- Streamlit Webseite von TOS.
    │   │
    │   └── tos            <- Das ist TOS!
    │       ├── data              <- Module zum bearbeiten, analysieren oder generieren von Daten.
    │       ├── features          <- Module, um aus Rohdaten Features für Modelle zu generieren.
    │       ├── models            <- Module, um Modelle zu trainieren und einzusetzen.
    │       └── visualization     <- Module für die Visualisierung von Ergebnissen.
    │
    ├── tests              <- Ordner für Tests und Testdaten
    │
    ├── pyproject.toml     <- Für die Entwicklungsumgebung erforderliche Module.
    │
    └── README.md          <- Dieses Dokument.

--------

Getting started
---------------

```shell
pip install poetry                          // only for the first execution
poetry lock                                 // only for the first execution

.venv\Scripts\activate

poetry config virtualenvs.in-project true   // only for the first execution
poetry install                              // only for the first execution

streamlit run src/scripts/app/Startseite.py
```


## Downloading Videos from YouTube

This pipeline downloads a video from YouTube, segments it into specified chunks, and generates transcriptions for each segment. The results are saved in the data folder. Below are the available command line arguments and their explanations:

- **`-n`** or **`--name`**: Specifies the name of the output files.
- **`-ch`** or **`--chunk`**: Defines the length of each video chunk in seconds. If this is not defined, the video will not be split.
- **`-url`**: The URL of the YouTube video to download.

### Example Command

Use the following command to download, chunk, and transcribe a video:

```shell
python src/video_preprocessing/download_videos/youtube_download.py -n hitched -ch 30 -url https://www.youtube.com/watch\?v\=r11Lr4FILX8
```
