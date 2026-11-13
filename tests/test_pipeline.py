import json

import pandas as pd

from beemon.config import load_config
from beemon.pipeline import postprocess, process_video
from beemon.video import iter_frames, probe
from conftest import read_index


def test_iter_frames_gives_neighbours(synthetic_video):
    path, _, _ = synthetic_video
    info = probe(path)
    assert info.frame_count == 90 and abs(info.fps - 30) < 0.01
    packs = list(iter_frames(path, range(0, 90, 10), offset=4))
    assert [p.index for p in packs] == list(range(0, 90, 10))
    for p in packs[1:-1]:
        assert read_index(p.frame) == p.index
        assert read_index(p.prev) == p.index - 4
        assert read_index(p.next) == p.index + 4
    assert read_index(packs[0].prev) == 0  # clamped at the start


def test_full_pipeline_with_fake_detector(synthetic_video, fake_detector_factory, tmp_path):
    path, paths, _ = synthetic_video
    cfg = load_config(None, [
        "video.frame_stride=2", "video.idle_stride=1", "filters.static_min_s=1.5", "filters.hotspot_min_video_s=1",
        "filters.min_duration_s=0.5", "output.save_video=true", "output.video_scale=0.5",
    ])
    detector = fake_detector_factory(paths, miss_rate=0.25)
    out = process_video(path, cfg, tmp_path / "results", detector=detector, classifier=None, log=lambda *_: None)
    summary = json.loads((out / "summary.json").read_text())
    tracks = pd.read_csv(out / "tracks_summary.csv")
    assert summary["visits"] == 2, tracks[["track_id", "valid", "reason", "duration_s", "max_radius_px"]]
    rejected = tracks[~tracks["valid"]]
    assert set(rejected["reason"]) <= {"static object", "background hotspot", "too short"}
    assert (rejected["reason"] != "too short").any()  # the bud was recognised as static/hotspot
    for name in ["tracks.csv", "detections.csv", "activity.csv", "heatmap.png", "tracks_mot.txt", "annotated.mp4",
                 "frame_log.csv", "background.jpg"]:
        assert (out / name).is_file(), name
    assert any((out / "thumbs").iterdir())

    # refilter with a very long static threshold: the bud now counts (shows the rule is what rejects it)
    cfg2 = load_config(None, ["filters.static_min_s=999", "filters.hotspot_fraction=1.1"])
    assert postprocess(out, cfg2, log=lambda *_: None)["visits"] == 3


def test_tracking_evaluation_perfect_and_switch(tmp_path):
    from beemon.evaluate import mot_metrics

    gt = pd.DataFrame({"frame": [1, 2, 3, 1, 2, 3], "id": [1, 1, 1, 2, 2, 2], "cx": [0, 1, 2, 100, 101, 102],
                       "cy": [0] * 6})
    perfect = gt.copy()
    perfect["id"] += 10
    m = mot_metrics(gt, perfect)
    assert m["MOTA"] == 1.0 and m["IDF1"] == 1.0 and m["ID_switches"] == 0
    swapped = perfect.copy()
    swapped.loc[(swapped["frame"] == 3) & (swapped["id"] == 11), "id"] = 99
    m2 = mot_metrics(gt, swapped)
    assert m2["ID_switches"] == 1 and m2["IDF1"] < 1.0
