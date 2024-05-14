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
