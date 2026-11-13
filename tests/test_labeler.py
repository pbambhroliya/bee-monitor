import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "beemon" / "labeler"))

from labeling import load_batch  # noqa: E402

from beemon.data.common import read_labels, write_cvat_yolo_package  # noqa: E402
from beemon.data.dataset import build_dataset  # noqa: E402
from beemon.data.share import import_labels, make_package  # noqa: E402


def _batch(tmp_path, review=False):
    batch = tmp_path / "gold_test"
    (batch / "images").mkdir(parents=True)
    names = []
    for i in range(4):
        name = f"VID01__f{i * 10:07d}.jpg"
        cv2.imwrite(str(batch / "images" / name), np.full((360, 640, 3), 120, np.uint8))
        names.append(name)
    write_cvat_yolo_package(batch / "cvat_prelabels.zip", ["bee", "bumblebee"],
                            [(n, np.array([[0, 0.5, 0.5, 0.05, 0.05]])) for n in names])
    if review:
        (batch / "review").mkdir()
        (batch / "review" / "review_list.txt").write_text("\n".join(names[:2]) + "\n")
    return batch


def test_label_save_export_import_and_build(tmp_path):
    batch = _batch(tmp_path)
    b = load_batch(batch)
    assert b.classes == ["bee", "bumblebee"] and len(b.images) == 4
    boxes, saved = b.boxes_for(b.images[0])
    assert len(boxes) == 1 and not saved and abs(boxes[0]["x1"] - (0.475 * 640)) < 1e-6
    b.save(b.images[0], boxes + [{"x1": 10, "y1": 10, "x2": 40, "y2": 30, "cls": 1}], "tester")
    b.save(b.images[1], [], "tester")                        # "no insect here"
    assert b.done() == {b.images[0].stem, b.images[1].stem}
    assert len(read_labels(batch / "labels_human" / f"{b.images[0].stem}.txt")) == 2

    # a teammate works on a package copy and sends the labels back
    package = make_package(batch, tmp_path / "share" / "gold_test.zip", log=lambda *_: None)
    mate = tmp_path / "mate"
    with zipfile.ZipFile(package) as archive:
        archive.extractall(mate)
    mb = load_batch(mate / "bee-labeller" / "data" / "gold_test")
    assert mb.done() == set()                                # the package has no labels of ours
    mb.save(mb.images[2], [{"x1": 100, "y1": 100, "x2": 130, "y2": 125, "cls": 0}], "mate")
    returned = tmp_path / "labels_mate.zip"
    returned.write_bytes(mb.export_zip("mate"))
    assert import_labels([returned], batch, log=lambda *_: None) == 1
    assert len(load_batch(batch).done()) == 3

    classes = tmp_path / "classes.yaml"
    classes.write_text(yaml.safe_dump({"names": ["bee", "bumblebee"]}))
    out = build_dataset([batch], tmp_path / "ds", classes, None, log=lambda *_: None)
    labels = {p.stem: read_labels(p) for p in (out / "labels").rglob("*.txt")}
    assert len(labels) == 3                                  # only the 3 labelled pictures, unlabelled skipped
    assert len(labels[b.images[1].stem]) == 0                # saved empty = background


def test_review_mode_shows_only_listed_pictures(tmp_path):
    batch = _batch(tmp_path, review=True)
    b = load_batch(batch)
    assert b.review_mode and len(b.images) == 2
