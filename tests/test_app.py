"""The app (3 pages) and the hard-negative round trip, in a temporary project folder."""

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

import beemon.catalog as catalog
import beemon.config as config
import beemon.hardneg as hardneg
from beemon.config import load_config
from beemon.pipeline import process_video

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Point every module at an empty project folder with a small videos.csv."""
    (tmp_path / "videos" / "20240724_WM2_plot18").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    with (tmp_path / "videos.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(catalog.FIELDS)
        writer.writerow(["20240724_WM2_plot18_SYNTH01", "2024-07-24", "20240724/20240724_WM2_plot18",
                         "SYNTH01.avi", "0.01", "todo", "", ""])
        writer.writerow(["20240801_P_RIMG0005", "2024-08-01", "20240801/P", "RIMG0005.MOV", "", "todo", "", ""])
    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(catalog, "CATALOG", tmp_path / "videos.csv")
    monkeypatch.setattr(catalog, "VIDEOS_DIR", tmp_path / "videos")
    monkeypatch.setattr(catalog, "MEMORY", tmp_path / "results" / "_app" / "remembered.json")
    monkeypatch.setattr(hardneg, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(hardneg, "CONTRIB_DIR", tmp_path / "contributions")
    monkeypatch.setattr(hardneg, "WORK_DIR", tmp_path / "contributions" / "work")
    monkeypatch.setattr(hardneg, "RECEIVED_DIR", tmp_path / "contributions" / "received")
    return tmp_path


class _Boxes:
    def __init__(self, xyxy):
        self.xyxy = np.array(xyxy, float).reshape(-1, 4)
        self.cls = np.zeros(len(self.xyxy))
        self.conf = np.full(len(self.xyxy), 0.6)

    def __len__(self):
        return len(self.xyxy)


class FakeYolo:
    """Two boxes on every picture: a 'bee' and a 'flower centre'."""
    names = {0: "bee", 1: "bumblebee", 2: "hoverfly", 3: "other_insect", 4: "moth"}

    def predict(self, frame, **_):
        return [type("R", (), {"boxes": _Boxes([[10, 10, 40, 30], [100, 100, 130, 125]])})()]


def test_video_names_are_unique(project, synthetic_video):
    rows = catalog.load_catalog()
    target = project / "videos" / "20240724_WM2_plot18" / "SYNTH01.avi"
    target.write_bytes(synthetic_video[0].read_bytes())
    assert catalog.video_name(target, rows) == "20240724_WM2_plot18_SYNTH01"
    assert catalog.video_name(Path("/elsewhere/SYNTH01.avi"), rows) == "SYNTH01"
    assert catalog.find_local(rows[0]) == target
    assert catalog.find_local(rows[1]) is None
    assert catalog.set_status(["20240801_P_RIMG0005"], "done") == 1
    assert catalog.load_catalog()[1]["status"] == "done"


def test_hard_negative_round_trip(project, synthetic_video):
    video_path = synthetic_video[0]
    video = "20240724_WM2_plot18_SYNTH01"
    state = hardneg.find_pictures(video, video_path, FakeYolo(), "fake.pt", every_s=0.5, want=3, imgsz=640)
    assert len(state["frames"]) == 3
    first, second = state["frames"][:2]
    # picture 1: keep the bee, delete the flower centre, add a missed insect
    boxes = hardneg.boxes_for_canvas(first)
    hardneg.save_review(video, state, first["frame_id"], [boxes[0], {"x1": 200, "y1": 50, "x2": 230, "y2": 70,
                                                                       "cls": 0}])
    # picture 2: nothing is an insect ("No insect here")
    hardneg.save_review(video, state, second["frame_id"], [])
    state = hardneg.load_state(video)
    assert [d["status"] for d in state["frames"][0]["detections"]] == ["true_positive", "false_positive",
                                                                       "true_positive"]
    assert hardneg.counts(state) == {"pictures": 3, "reviewed": 2, "wrong_boxes": 3, "insects": 2}
    # every model box has its "does it move?" picture, and a letter on the canvas
    assert all(hardneg.crop_path(video, d["detection_id"]).is_file() for f in state["frames"]
               for d in f["detections"] if not d["manual"])
    assert [b.get("label") for b in hardneg.boxes_for_canvas(state["frames"][2])] == ["A", "B"]
    # searching again continues: no picture twice, the old decisions stay
    state = hardneg.find_pictures(video, video_path, FakeYolo(), "fake.pt", every_s=0.5, want=5, imgsz=640)
    assert len({f["frame_id"] for f in state["frames"]}) == len(state["frames"]) == 5
    assert hardneg.counts(state)["reviewed"] == 2
    # re-saving keeps the manual box once
    hardneg.save_review(video, state, first["frame_id"], hardneg.boxes_for_canvas(state["frames"][0]))
    assert sum(d["manual"] for d in hardneg.load_state(video)["frames"][0]["detections"]) == 1

    data, summary = hardneg.export_zip([video], "Tester")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = set(archive.namelist())
        labels = {n: archive.read(n).decode() for n in names if n.startswith("labels/")}
        report = list(csv.DictReader(io.StringIO(archive.read("review_report.csv").decode())))
        data_yaml = yaml.safe_load(archive.read("data.yaml"))
    assert summary["pictures"] == 2 and summary["wrong_boxes_deleted"] == 3
    assert {f"images/train/{first['frame_id']}.jpg", f"images/train/{second['frame_id']}.jpg"} <= names
    assert sorted(len(t.splitlines()) for t in labels.values()) == [0, 2]   # hard negative = empty label file
    assert data_yaml["names"][0] == "bee" and "path" not in data_yaml
    assert {r["final_annotation_status"] for r in report} >= {"included", "excluded_rejected", "excluded_unreviewed"}

    zip_path = project / "hardneg_Tester.zip"
    zip_path.write_bytes(data)
    batch, videos = hardneg.import_contribution(zip_path, log=lambda *_: None)
    assert videos == [video]
    assert len(list((batch / "labels_human").glob("*.txt"))) == 2
    assert len(list((batch / "images").glob(f"{video}__f*.jpg"))) == 2


def test_import_zip_of_the_old_app(project):
    """Dataset zip of another review tool: hex video codes in the names + a separate review_report.csv."""
    import cv2

    buffer = io.BytesIO()
    image = cv2.imencode(".jpg", np.zeros((40, 60, 3), np.uint8))[1].tobytes()
    with zipfile.ZipFile(buffer, "w") as archive:
        for split, code, frame in [("train", "a1b2c3d4e5f60718", 30), ("val", "0f1e2d3c4b5a6978", 60),
                                   ("test", "99aa88bb77cc66dd", 90)]:
            archive.writestr(f"images/{split}/{code}_{frame:09d}.jpg", image)
            archive.writestr(f"labels/{split}/{code}_{frame:09d}.txt", "0 0.5 0.5 0.2 0.2\n" if split == "train" else "")
        archive.writestr("data.yaml", yaml.safe_dump({"path": "/somewhere/else", "train": "images/train",
                                                      "val": "images/val", "test": "images/test", "nc": 5,
                                                      "names": FakeYolo.names and list(FakeYolo.names.values())}))
    zip_path = project / "dataset_export.zip"
    zip_path.write_bytes(buffer.getvalue())
    report = project / "review_report.csv"
    with report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=hardneg.CSV_FIELDS)
        writer.writeheader()
        for name, frame in [("RIMG0001.MOV", 30), ("RIMG0002.MOV", 60), ("RIMG0001.MOV", 90)]:
            writer.writerow({"video_filename": name, "frame_number": frame})
    batch, videos = hardneg.import_contribution(zip_path, report, log=lambda *_: None)
    # two different RIMG0001 videos (other plots) stay apart
    assert len(videos) == 3 and "RIMG0002" in videos and sum(v.startswith("RIMG0001") for v in videos) == 2
    assert len(list((batch / "labels_human").glob("*.txt"))) == 3


def test_app_pages_render(project, synthetic_video, fake_detector_factory):
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    import app_pages._shared as shared

    shared.MODELS_DIR = project / "models"
    path, paths, _ = synthetic_video
    cfg = load_config(None, ["video.frame_stride=2", "video.idle_stride=1", "output.save_video=false"])
    process_video(path, cfg, project / "results", detector=fake_detector_factory(paths), classifier=None,
                  log=lambda *_: None, name="20240724_WM2_plot18_SYNTH01")
    video = "20240724_WM2_plot18_SYNTH01"
    hardneg.find_pictures(video, path, FakeYolo(), "fake.pt", every_s=0.5, want=2, imgsz=640)
    catalog.remember("me", "last_video", video)

    at = streamlit_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.metric[0].value == "1"   # one processed video
    for page in ["app_pages/track.py", "app_pages/hard_negatives.py"]:
        at.switch_page(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])
    assert any("Picture **1** of 2" in m.value for m in at.markdown)
    assert any("Does it move?" in m.value for m in at.markdown)
    json.dumps(catalog.load_memory())


def test_big_frame_skip_reads_exact_frames_and_renders_from_cache(synthetic_video, fake_detector_factory, tmp_path):
    from beemon.pipeline import adapt_to_stride
    from beemon.video import iter_frames
    from conftest import read_index

    path, paths, _ = synthetic_video
    packs = list(iter_frames(path, [3, 50, 88], offset=2))   # gaps > SEEK_GAP are jumped over
    assert [p.index for p in packs] == [3, 50, 88]
    assert [read_index(p.frame) for p in packs] == [3, 50, 88]
    cfg = load_config(None, ["video.frame_stride=5", "output.save_video=true"])
    adapted = adapt_to_stride(cfg, 30.0)
    assert adapted["tracker"]["min_hits"] == 2 and adapted["filters"]["min_detections"] == 2
    out = process_video(path, cfg, tmp_path / "results", detector=fake_detector_factory(paths), classifier=None,
                        log=lambda *_: None)
    assert (out / "annotated.mp4").is_file() and not (out / "_frames").exists()
