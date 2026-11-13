import zipfile

import cv2
import numpy as np
import yaml

from beemon.data.common import read_labels, write_cvat_yolo_package
from beemon.data.dataset import build_dataset, tile_dataset
from beemon.data.extract import extract_frames, make_clip
from beemon.data.synth import make_synthetic


def _batch_from_video(video, tmp_path, name="batch01"):
    """Extract frames, then fake a CVAT YOLO 1.1 export whose class order differs from classes.yaml."""
    batch = tmp_path / name
    extract_frames(video, batch, every_s=0.3, max_per_video=None, split_yaml=None, log=lambda *_: None)
    items = []
    for i, image in enumerate(sorted((batch / "images").glob("*.jpg"))):
        # CVAT label order: [hoverfly, bee]; boxes: one "bee" (id 1) on even frames, none on odd frames
        labels = np.array([[1, 0.5, 0.5, 0.05, 0.05]]) if i % 2 == 0 else np.zeros((0, 5))
        items.append((image.name, labels))
    package = write_cvat_yolo_package(tmp_path / f"{name}_export.zip", ["hoverfly", "bee"], items)
    with zipfile.ZipFile(package) as archive:
        archive.extractall(batch / "cvat_export")
    return batch


def test_build_rgb_and_mie_datasets(synthetic_video, tmp_path):
    video, _, _ = synthetic_video
    batch = _batch_from_video(video, tmp_path)
    classes = tmp_path / "classes.yaml"
    classes.write_text(yaml.safe_dump({"names": ["bee", "bumblebee", "hoverfly"]}))
    split = tmp_path / "split.yaml"
    split.write_text(yaml.safe_dump({"test_videos": [], "val_videos": []}))

    out = build_dataset([batch], tmp_path / "ds_rgb", classes, split, mode="rgb", log=lambda *_: None)
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "bee", 1: "bumblebee", 2: "hoverfly"}
    labels = [read_labels(p) for p in (out / "labels").rglob("*.txt")]
    assert labels and all(len(lab) == 0 or set(lab[:, 0]) == {0} for lab in labels)  # "bee" remapped 1 -> 0
    assert any(len(lab) == 0 for lab in labels)  # empty frames kept as background examples

    out_mie = build_dataset([batch], tmp_path / "ds_mie", classes, split, mode="mie", mie_offset=2,
                            log=lambda *_: None)
    info = yaml.safe_load((out_mie / "dataset_info.yaml").read_text())
    assert info["mode"] == "mie" and info["mie_offset"] == 2
    image = cv2.imread(str(next((out_mie / "images").rglob("*.jpg"))))
    assert image is not None and image.shape[:2] == (360, 640)

    tiles = tile_dataset(out / "data.yaml", tmp_path / "ds_tiles", size=320, overlap=0.2, empty_keep=1.0,
                         log=lambda *_: None)
    assert any((tiles / "images").rglob("*.jpg"))

    synth = make_synthetic(out / "data.yaml", tmp_path / "ds_synth", n_images=5, per_image=(1, 2),
                           log=lambda *_: None)
    synth_labels = [read_labels(p) for p in (synth / "labels" / "train").glob("*.txt")]
    assert len(synth_labels) == 5 and all(len(lab) >= 1 for lab in synth_labels)
    merged = yaml.safe_load((synth / "data.yaml").read_text())
    assert len(merged["train"]) == 2


def test_clip_export(synthetic_video, tmp_path):
    video, _, _ = synthetic_video
    out = make_clip(video, tmp_path / "clip", start_s=0.5, duration_s=1.0, stride=2, log=lambda *_: None)
    assert len(list((out / "frames").glob("*.jpg"))) == 15
    assert (out / "clip_frames.zip").is_file() and (out / "clip.json").is_file()
