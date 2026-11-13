"""The list of project videos (videos.csv) and where they are on this computer.

Many videos have the same file name (RIMG0001.MOV exists in several plot folders), so every
video gets a unique name in videos.csv, e.g. ``20240724_WM2_plot18_RIMG0001``:
the Sciebo folder + the file name. Everything (results, pictures, contributions) uses that name.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from beemon.config import PROJECT_DIR
from beemon.video import VIDEO_EXTENSIONS

CATALOG = PROJECT_DIR / "videos.csv"
VIDEOS_DIR = PROJECT_DIR / "videos"
MEMORY = PROJECT_DIR / "results" / "_app" / "remembered.json"
FIELDS = ["video", "date", "sciebo_folder", "file", "size_gb", "status", "taken_by", "notes"]
STATUS_TEXT = {"todo": "free", "taken": "someone is on it", "done": "done", "skip": "skip", "test": "test video"}


def load_catalog(path: Path | None = None) -> list[dict]:
    path = path or CATALOG
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def save_catalog(rows: list[dict], path: Path | None = None) -> None:
    path = path or CATALOG
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def set_status(video_ids, status: str, path: Path | None = None) -> int:
    rows = load_catalog(path)
    changed = 0
    for row in rows:
        if row["video"] in set(video_ids) and row["status"] != status:
            row["status"] = status
            changed += 1
    if changed:
        save_catalog(rows, path)
    return changed


def _folder_name(row: dict) -> str:
    return row["sciebo_folder"].rstrip("/").split("/")[-1]


def video_name(path: str | Path, rows: list[dict] | None = None) -> str:
    """Unique name of a video file: its videos.csv name when the file sits in a folder named like
    the Sciebo folder (or is already named like that), else the file name without extension."""
    path = Path(path)
    for row in load_catalog() if rows is None else rows:
        if path.stem == row["video"]:
            return row["video"]
        if path.name.lower() == row["file"].lower() and path.parent.name == _folder_name(row):
            return row["video"]
    return path.stem


def find_local(row: dict, videos_dir: Path | None = None) -> Path | None:
    """Where this video is on this computer: remembered path, or videos/<sciebo folder>/<file>,
    or videos/<unique name>.MOV."""
    remembered = load_memory().get("videos", {}).get(row["video"])
    if remembered and Path(remembered).is_file():
        return Path(remembered)
    videos_dir = videos_dir or VIDEOS_DIR
    if not videos_dir.is_dir():
        return None
    for candidate in videos_dir.rglob("*"):
        if candidate.suffix.lower() not in VIDEO_EXTENSIONS or not candidate.is_file():
            continue
        if candidate.stem == row["video"]:
            return candidate
        if candidate.name.lower() == row["file"].lower() and candidate.parent.name == _folder_name(row):
            return candidate
    return None


# ---------------------------------------------------------------- remembered paths (this computer only)
def load_memory() -> dict:
    try:
        return json.loads(MEMORY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def remember(kind: str, key: str, value: str) -> None:
    """kind 'models': key = path; kind 'videos': key = video name, value = path."""
    memory = load_memory()
    memory.setdefault(kind, {})[key] = value
    MEMORY.parent.mkdir(parents=True, exist_ok=True)
    MEMORY.write_text(json.dumps(memory, indent=1), encoding="utf-8")


def forget(kind: str, key: str) -> None:
    memory = load_memory()
    if memory.get(kind, {}).pop(key, None) is not None:
        MEMORY.write_text(json.dumps(memory, indent=1), encoding="utf-8")
