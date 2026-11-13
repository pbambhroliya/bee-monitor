import zipfile

import numpy as np
import pandas as pd
import yaml

from beemon.config import load_config
from beemon.data.autolabel import autolabel_results, classify_frames, label_quality, DEFAULTS
from beemon.data.common import read_labels, write_cvat_yolo_package
from beemon.data.dataset import build_dataset
from beemon.pipeline import process_video


def _process(synthetic_video, fake_detector_factory, tmp_path):
    path, paths, _ = synthetic_video
    cfg = load_config(None, ["video.frame_stride=2", "video.idle_stride=1", "filters.static_min_s=1.5",
                             "filters.hotspot_min_video_s=1", "output.save_video=false"])
    return process_video(path, cfg, tmp_path / "results", detector=fake_detector_factory(paths, miss_rate=0.25),
                         classifier=None, log=lambda *_: None)


def test_frames_are_categorised(synthetic_video, fake_detector_factory, tmp_path):
    result = _process(synthetic_video, fake_detector_factory, tmp_path)
    frames, rows, good = classify_frames(result, {**DEFAULTS, "min_track_conf": 0.3})
    kinds = set(frames["kind"])
    assert len(good) == 2                      # the two moving bees
    assert "hard_positive" in kinds            # frames where the fake detector missed a tracked bee
    assert not frames.empty


def test_autolabel_review_override_and_quality(synthetic_video, fake_detector_factory, tmp_path):
    result = _process(synthetic_video, fake_detector_factory, tmp_path)
    names = ["bee", "bumblebee"]
    batch = tmp_path / "r1_auto"
    info = autolabel_results([result], batch, names, None,
                             {"min_track_conf": 0.3, "frames_per_video": 20, "min_gap_s": 0.1,
                              "review_random": 5, "review_uncertain": 5}, log=lambda *_: None)
    assert info["auto_labelled"] > 0
    labels = {p.stem: read_labels(p) for p in (batch / "labels_auto").glob("*.txt")}
    catalogue = pd.read_csv(batch / "autolabel.csv")
    positives = catalogue[catalogue["kind"].isin(["positive", "hard_positive"])]
    assert len(positives) and all(len(labels[s.replace(".jpg", "")]) >= 1 for s in positives["image"])
    # the static "bud" is never a positive label: every positive box lies on a bee path
    for image in positives["image"]:
        assert all(abs(cy - 60 / 360) > 0.02 or abs(cx - 320 / 640) > 0.02
                   for _, cx, cy, _, _ in labels[image.replace(".jpg", "")])

    # simulate the CVAT review of the random sample: the reviewer deletes the boxes of the first picture
    # and keeps the others (CVAT lists every task picture in train.txt)
    sample = (batch / "review" / "random_sample.txt").read_text().split()
    kept = [read_labels(batch / "labels_auto" / n.replace(".jpg", ".txt")) for n in sample[1:]]
    package = write_cvat_yolo_package(tmp_path / "export.zip", names,
                                      [(sample[0], np.zeros((0, 5)))] + list(zip(sample[1:], kept)))
    with zipfile.ZipFile(package) as archive:
        archive.extractall(batch / "cvat_export")
    quality = label_quality(batch)
    assert quality["frames_checked"] == len(sample)

    classes = tmp_path / "classes.yaml"
    classes.write_text(yaml.safe_dump({"names": names}))
    out = build_dataset([batch], tmp_path / "ds", classes, None, log=lambda *_: None)
    built = {p.stem: read_labels(p) for p in (out / "labels").rglob("*.txt")}
    assert len(built[sample[0].replace(".jpg", "")]) == 0   # human correction replaced the automatic label
    # with human_only the automatic labels of the batch are ignored
    out2 = build_dataset([batch], tmp_path / "ds2", classes, None, human_only=[batch], log=lambda *_: None)
    assert len(list((out2 / "labels").rglob("*.txt"))) == len(sample)


def test_track_drifting_onto_a_flower_is_not_labelled():
    from beemon.data.autolabel import _trusted_rows

    # a bee (moving, motion ratio 3) whose track then sits on a flower centre (motion ratio 1)
    rows = pd.DataFrame({
        "frame": range(10), "track_id": 1, "detected": [True] * 4 + [False] + [True] * 5,
        "conf": [0.6] * 10, "motion_ratio": [3.0, 3.0, 3.0, 3.0, 0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })
    trusted = _trusted_rows(rows, DEFAULTS)
    assert trusted[:4].all()            # the bee
    assert not trusted[4]               # gap between a bee box and a flower box: not trusted
    assert not trusted[5:].any()        # the flower centre


def test_unfinished_review_is_not_used_as_background(tmp_path, synthetic_video, fake_detector_factory):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "beemon" / "labeler"))
    from labeling import load_batch

    result = _process(synthetic_video, fake_detector_factory, tmp_path)
    batch = tmp_path / "r_auto"
    autolabel_results([result], batch, ["bee"], None, {"min_track_conf": 0.3, "frames_per_video": 20,
                                                       "min_gap_s": 0.1, "review_random": 6, "review_uncertain": 6},
                      log=lambda *_: None)
    review = load_batch(batch)
    review.save(review.images[0], [], "tester")     # the reviewer finished ONE picture in the labeller
    classes = tmp_path / "classes.yaml"
    classes.write_text(yaml.safe_dump({"names": ["bee"]}))
    out = build_dataset([batch], tmp_path / "ds", classes, None, human_only=[batch], log=lambda *_: None)
    assert len(list((out / "labels").rglob("*.txt"))) == 1   # the unfinished ones are NOT turned into background
