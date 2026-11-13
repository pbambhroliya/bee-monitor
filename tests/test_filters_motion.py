import numpy as np
import pandas as pd

from beemon.filters import apply_filters, hotspot_map, summarize_tracks
from beemon.motion import box_motion, mie_image
from beemon.tracker import ROW_FIELDS

VIDEO = {"width": 1920, "height": 1080, "duration_s": 120.0, "processed_s": 120.0}
CFG = {"min_duration_s": 0.4, "min_detections": 4, "static_min_s": 20.0, "static_radius_factor": 1.0,
       "static_motion_max": 1.3, "hotspot_fraction": 0.4, "hotspot_track_fraction": 0.5, "hotspot_min_video_s": 20}


def rows_for(track_id, xy, t0=0.0, dt=0.1, motion_ratio=1.0):
    return [(i, t0 + i * dt, track_id, x, y, 30.0, 24.0, 0.8, 0, True, 10.0, motion_ratio) for i, (x, y) in enumerate(xy)]


def frame(rows):
    return pd.DataFrame(rows, columns=list(ROW_FIELDS))


def test_static_object_vs_long_stay_vs_moving_bee():
    n = 300  # 30 s at 10 fps
    still = [(500.0, 500.0)] * n
    moving = [(100.0 + 3 * i, 300.0) for i in range(n)]
    rows = frame(rows_for(1, still, motion_ratio=1.0) + rows_for(2, still, motion_ratio=3.0) + rows_for(3, moving))
    summary = apply_filters(summarize_tracks(rows, {0: "bee"}), rows, None, CFG, VIDEO).set_index("track_id")
    assert summary.loc[1, "reason"] == "static object" and not summary.loc[1, "valid"]
    assert summary.loc[2, "valid"] and summary.loc[2, "long_stay"]  # sits still but moves itself: keep, flag
    assert summary.loc[3, "valid"] and not summary.loc[3, "long_stay"]


def test_short_and_unconfirmed_tracks_are_rejected():
    rows = frame(rows_for(1, [(10.0, 10.0)] * 3) + rows_for(2, [(50.0 + i, 50.0) for i in range(20)]))
    summary = summarize_tracks(rows, {0: "bee"}, confirmed_ids={2})
    result = apply_filters(summary, rows, None, CFG, VIDEO).set_index("track_id")
    assert result.loc[1, "reason"] == "too short"
    assert result.loc[2, "valid"]


def test_hotspot_rejects_tracks_in_always_busy_cells():
    dets = pd.DataFrame({"frame": np.arange(100), "x1": 990.0, "y1": 530.0, "x2": 1010.0, "y2": 550.0})
    hot = hotspot_map(dets, n_frames=100, width=1920, height=1080, grid=(32, 18))
    assert hot.max() == 1.0
    rows = frame(rows_for(1, [(1000.0 + (i % 3), 540.0) for i in range(50)]))
    result = apply_filters(summarize_tracks(rows, {0: "bee"}), rows, hot, CFG, VIDEO)
    assert result.iloc[0]["reason"] == "background hotspot"


def test_mie_channels_and_box_motion():
    image = np.zeros((100, 100, 3), np.uint8)
    image[..., 0], image[..., 1], image[..., 2] = 100, 150, 200
    fd = np.zeros((100, 100), np.uint8)
    fd[40:60, 40:60] = 200  # something moving in the middle
    mie = mie_image(image, fd)
    assert mie[0, 0, 0] == 150 and mie[0, 0, 1] == 150 and mie[50, 50, 2] == 200 and mie[0, 0, 2] == 0
    inside, ratio = box_motion(fd, np.array([[40, 40, 60, 60], [5, 5, 15, 15]]))
    assert inside[0] > 150 and ratio[0] > 5
    assert inside[1] == 0 and abs(ratio[1] - 1.0) < 1e-6
