# Training a new model

For the person who trains. You need a computer with an **NVIDIA GPU**; on a laptop CPU, training takes
days. Install as in the README, but with the CUDA version of PyTorch:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision          # CUDA build (or the one your GPU machine recommends)
pip install -r requirements.txt && pip install -e .
# a server without a screen has no libGL: use the headless OpenCV
pip uninstall -y opencv-python && pip install opencv-python-headless
```

## 1. Base model on BuzzSet (done: `yolo11m_buzzset.pt`)

```bash
# set `path:` in configs/buzzset_data.yaml to your BuzzSet copy first
python -m beemon train --preset pretrain --weights yolo11m.pt --data configs/buzzset_data.yaml \
       --imgsz 1280 --name m_buzzset
```

The result is in `runs/m_buzzset/weights/best.pt`.

## 2. Collect the hard negatives

Received zips go into the app: page **Hard negatives** → *For the person who trains the model*. Each zip
becomes a folder `data/labeling/hardneg_<zip name>/`, and its videos are marked `done` in `videos.csv`.
You can also do it in a terminal:

```bash
python -m beemon import-hardneg contributions/received/hardneg_Anna_20261120.zip
python -m beemon import-hardneg dataset_export.zip --review-csv review_report.csv   # zip of another review tool
```

**Test videos** must never be used for training. Choose 4–6 videos, put their names into
`configs/split.yaml` and set their status to `test` in `videos.csv`. Then nobody collects hard
negatives on them.

## 3. Build the dataset and fine-tune

```bash
python -m beemon build-dataset data/labeling/hardneg_* --out data/datasets/hn_v1
```

Fine-tune on **BuzzSet + the new pictures together**. On the new pictures alone (mostly "no insect")
the model would forget what insects look like. A `data.yaml` can list several folders:

```yaml
# data/datasets/hn_v1_plus_buzzset.yaml
train: [/path/to/buzzset/yolo/images/train, /full/path/to/data/datasets/hn_v1/images/train]
val:   [/path/to/buzzset/yolo/images/valid, /full/path/to/data/datasets/hn_v1/images/val]
names: {0: bee, 1: bumblebee, 2: hoverfly, 3: other_insect, 4: moth}
```

```bash
python -m beemon train --preset finetune --weights models/yolo11m_buzzset.pt \
       --data data/datasets/hn_v1_plus_buzzset.yaml --name hn_v1
```

## 4. Compare and share

```bash
python -m beemon val --weights runs/hn_v1/weights/best.pt --data configs/buzzset_data.yaml --imgsz 1280
```

Also run both models on the same test video in the app (*Track a video*). Fewer rejected tracks and
fewer boxes on flowers mean fewer hard negatives were missed.

Before you share a model, remove the folder paths of the training computer from the file:

```bash
python -m beemon clean-model runs/hn_v1/weights/best.pt --out models/yolo11m_hn_v1.pt
```

Upload it to Google Drive and replace the link in the README.

## More

`python -m beemon --help` lists everything else: automatic labelling of all videos in rounds
(`beemon workflow`, settings in `configs/workflow.yaml`), tracking evaluation (`eval-track`,
`eval-counts`), synthetic data, tiles, and a crop classifier. The full labeller for complete box labels
is `streamlit run beemon/labeler/app.py`.
