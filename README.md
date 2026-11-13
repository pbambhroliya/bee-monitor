# 🐝 Bee monitoring

Find and follow insects (bees, bumblebees, hoverflies, moths …) in field videos of flower plots and count
their **visits**. Everything runs on your own computer in a simple app with three pages:

| Page | What it does |
|---|---|
| **Results** | Visits per video, activity over time, where the insects spend time (heatmap), annotated video |
| **Track a video** | Choose a model and a video, press Start, and watch the detection and tracking **live** |
| **Hard negatives** | Help make the model better: it shows you its boxes, you delete the wrong ones (see [CONTRIBUTING.md](CONTRIBUTING.md)) |

## Start in 3 steps

You need **Python 3.10 or newer** ([python.org](https://www.python.org/downloads/); on Windows tick
*"Add python.exe to PATH"* during the installation).

1. **Get the code.** `git clone <this repository's URL>`, or on GitHub: *Code → Download ZIP* and unzip it.
2. **Download the model** → <!-- MODEL LINK: replace the next line with your Google Drive link -->
   **[yolo11m_buzzset.pt (Google Drive)](https://drive.google.com/drive/folders/1Lv1iggqYYJiTYJS4M9TcgNSGLdNa0p8H?usp=sharing)** and put it into the `models` folder.
3. **Start the app.**
   - Windows: double-click **`run.bat`**
   - Linux / Mac: open a terminal in the folder and run `bash run.sh`

   The first start installs everything (5–15 minutes). Then the app opens in your browser
   (if not: <http://localhost:8501>). Stop it with `Ctrl+C` in the terminal window.

## The videos

The 2024 field videos are on Sciebo in `video_records_2024/` (about 2.7 GB each). All of them are listed in
**[videos.csv](videos.csv)**, each with a unique name made of its folder and file name, because
`RIMG0001.MOV` exists in several plot folders.

| column | meaning |
|---|---|
| `video` | unique name, e.g. `20240724_WM2_plot18_RIMG0001` |
| `sciebo_folder`, `file` | where to download it |
| `status` | `todo` free · `taken` someone works on it · `done` hard negatives collected · `test` kept aside to measure the model (never used for training) · `skip` |
| `taken_by` | who works on it |

Put downloaded videos into `videos/<Sciebo folder name>/`, e.g. `videos/20240724_WM2_plot18/RIMG0001.MOV`,
and the app finds them. A video somewhere else also works: choose *Another file on this computer…* or
**Browse…**. Files are always used where they are and never copied.

## The model

`yolo11m_buzzset.pt`: YOLO11m (Ultralytics) trained on the public **BuzzSet** dataset (5 types: bee,
bumblebee, hoverfly, other_insect, moth) at 1280 px. On the BuzzSet validation images it reaches
mAP50 0.50 and mAP50-95 0.33 (precision 0.55, recall 0.50).

Its typical mistakes in our videos are flower centres, buds, leaves and the orange stake. Those are
exactly the **hard negatives** we collect to train the next version.

On a normal laptop (no NVIDIA card) it needs about 3 seconds per analysed frame at 1280 px. On the *Track a
video* page choose **Finish within …** (e.g. 30 min) and the app picks how many frames it can analyse in
that time, and says how exact the result will be. A 25-minute video in 30 minutes means about one frame
every 3 seconds: good for an overview, but short visits are missed. For exact counts use a computer with
an NVIDIA card, which can analyse every frame.

## Help with hard negatives

See **[CONTRIBUTING.md](CONTRIBUTING.md)**. In short: take a free video, let the model look at it, delete
its wrong boxes (about 20 minutes), download the file and send it.

The file is a normal YOLO dataset (`images/`, `labels/`, `data.yaml`), so it works with any YOLO training,
and YOLO dataset zips from other review tools can be added the same way.

## What is in this folder

```
app.py              the app (start: run.bat / run.sh, or  streamlit run app.py)
app_pages/          its three pages
beemon/             the engine: detection, tracking (BeeTrack), filters, statistics, training tools
configs/            settings (pipeline.yaml = detection/tracking/filter settings)
videos.csv          list of all 2024 videos and their status
docs/TRAINING.md    for the person who trains new models
tests/              automatic tests (pytest)
── made on your computer, not on GitHub ──
models/             model files (.pt)
videos/             downloaded videos
results/            one folder per processed video
contributions/      your hard-negative work and the files you download
```

## For developers

```bash
python -m beemon --help                                  # all commands
python -m beemon process videos/20240724_WM2_plot18/RIMG0001.MOV --weights models/yolo11m_buzzset.pt
python -m pytest -q                                      # tests
```

How tracking works: detections are linked from frame to frame by **BeeTrack**, our tracker built on
ByteTrack's two-stage matching with a Kalman filter. Tracks are then filtered: too short, standing still
for seconds (flower centres, the stake), or in a spot where something is detected all the time. Only the
remaining tracks count as visits. Training a new model: [docs/TRAINING.md](docs/TRAINING.md).
