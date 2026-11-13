"""Pieces used by several pages: choosing a model and a video (used where they are — never copied)."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import streamlit as st

from beemon.catalog import (CATALOG, STATUS_TEXT, VIDEOS_DIR, find_local, load_catalog, load_memory, remember,
                            video_name)
from beemon.config import PROJECT_DIR
from beemon.video import VIDEO_EXTENSIONS

MODELS_DIR = PROJECT_DIR / "models"
OTHER = "Another file on this computer…"


# ------------------------------------------------------------------ native "open file" window
def _dialog_command(title: str, kind: str) -> list[str] | None:
    patterns = ["*.pt"] if kind == "model" else [f"*{e}" for e in sorted(VIDEO_EXTENSIONS)] + \
        [f"*{e.upper()}" for e in sorted(VIDEO_EXTENSIONS)]
    if sys.platform.startswith("linux"):
        if shutil.which("zenity"):
            return ["zenity", "--file-selection", f"--title={title}",
                    f"--file-filter={kind} files | {' '.join(patterns)}", "--file-filter=all files | *"]
        if shutil.which("kdialog"):
            return ["kdialog", "--getopenfilename", str(Path.home()), " ".join(patterns)]
        return None
    if sys.platform == "darwin":
        return ["osascript", "-e", f'POSIX path of (choose file with prompt "{title}")']
    script = ("import tkinter as tk\nfrom tkinter import filedialog\nr = tk.Tk(); r.withdraw()\n"
              "r.attributes('-topmost', True)\n"
              f"print(filedialog.askopenfilename(title={title!r}, filetypes=[({kind!r}, {' '.join(patterns)!r}), "
              "('all files', '*')]))")
    return [sys.executable, "-c", script]


def can_browse() -> bool:
    return _dialog_command("x", "video") is not None


def _browse(key: str, title: str, kind: str) -> None:
    command = _dialog_command(title, kind)
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=900).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    if out:
        st.session_state[key] = out


def file_field(label: str, key: str, kind: str, placeholder: str) -> Path | None:
    """Text field for a file path + a 'Browse…' button that opens the normal file window."""
    row = st.container(horizontal=True, vertical_alignment="bottom")
    text = row.text_input(label, key=key, placeholder=placeholder)
    if can_browse():
        row.button("Browse…", key=key + "_browse", icon=":material/folder_open:", on_click=_browse,
                   args=(key, label, kind))
    if not text.strip():
        return None
    path = Path(text.strip().strip('"').strip("'")).expanduser()
    if not path.is_file():
        st.error(f"There is no file at `{path}`.")
        return None
    return path


# ------------------------------------------------------------------ model
@st.cache_data(show_spinner=False, max_entries=32)
def model_info(path: str, mtime: float) -> dict:
    """Classes, base architecture and training size of a YOLO .pt."""
    import torch

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        return {"error": str(error)}
    model = ckpt.get("model") or ckpt.get("ema")
    names = getattr(model, "names", {}) or {}
    args = ckpt.get("train_args") or {}
    base = Path(str(args.get("model", ""))).stem
    scale = next((c for c in "nslmx" if base.endswith(c)), None)
    return {"names": list(names.values()) if isinstance(names, dict) else list(names), "base": base,
            "scale": scale, "imgsz": args.get("imgsz"), "task": getattr(model, "task", "detect"),
            "mb": round(Path(path).stat().st_size / 1e6, 1)}


def model_picker(key: str) -> tuple[Path | None, dict]:
    """Models in models/ + models used before (remembered by their path) + any other file."""
    found = sorted(MODELS_DIR.glob("*.pt"))
    remembered = [Path(p) for p in load_memory().get("models", {}) if Path(p).is_file()]
    options = found + [p for p in remembered if p not in found]
    if not options:
        st.info("No model yet. Download it (link in the README) and put it into the `models` folder, "
                "or choose the file below.", icon=":material/info:")
    labels = {str(p): (p.name if p.parent == MODELS_DIR else f"{p.name}  ·  {p.parent}") for p in options}
    choice = st.selectbox("Model", [str(p) for p in options] + [OTHER], key=key,
                          format_func=lambda s: labels.get(s, s))
    path = Path(choice) if choice != OTHER else file_field("Model file (.pt)", key + "_path", "model",
                                                            "e.g. /home/me/Downloads/best.pt")
    if path is None:
        return None, {}
    info = model_info(str(path), path.stat().st_mtime)
    if info.get("error"):
        st.error(f"This file is not a model that can be read: {info['error']}")
        return None, {}
    if info.get("task") not in (None, "detect"):
        st.error(f"This is a '{info['task']}' model, not a detector.")
        return None, {}
    if choice == OTHER:
        remember("models", str(path), "")
    st.caption(f"{info['mb']} MB · {info['base'] or '?'} · trained at {info['imgsz'] or '?'} px · "
               f"insect types: {', '.join(info['names'])}")
    return path, info


# ------------------------------------------------------------------ video
def _label(row: dict, here: Path | None) -> str:
    status = STATUS_TEXT.get(row["status"], row["status"] or "free")
    return (f"{row['video']}  ·  {status}" + ("" if here else "  ·  not on this computer"))


def video_picker(key: str, only_listed: bool = False) -> tuple[Path | None, str | None, dict | None]:
    """Returns (file, unique video name, row of videos.csv or None)."""
    rows = load_catalog()
    local = {row["video"]: find_local(row) for row in rows}
    names = [row["video"] for row in rows]
    options = sorted(names, key=lambda n: local[n] is None) + ([] if only_listed else [OTHER])
    by_name = {row["video"]: row for row in rows}
    choice = st.selectbox("Video", options, key=key,
                          format_func=lambda n: n if n == OTHER else _label(by_name[n], local[n]),
                          help=f"The list is `{CATALOG.name}`. Videos you put into the `videos` folder are found "
                               "automatically.")
    if choice == OTHER:
        path = file_field("Video file", key + "_path", "video", "e.g. /home/me/Downloads/RIMG0006.MOV")
        return (path, video_name(path, rows), None) if path else (None, None, None)
    row = by_name[choice]
    path = local[choice]
    if path is None:
        folder = row["sciebo_folder"].rstrip("/").split("/")[-1]
        st.warning(f"Download **{row['file']}** from the Sciebo folder `video_records_2024/{row['sciebo_folder']}`. "
                   f"Put it into `videos/{folder}/` (found automatically) or choose the file:",
                   icon=":material/download:")
        path = file_field("Video file", key + "_path_" + choice, "video", f"…/{row['file']}")
        if path is None:
            return None, choice, row
        if path.name.lower() != row["file"].lower():
            st.warning(f"The file is called `{path.name}`, the list says `{row['file']}` — is it the right video?")
        remember("videos", choice, str(path))
    st.caption(f"`{path}`")
    return path, choice, row


def videos_dir_hint() -> str:
    return str(VIDEOS_DIR.relative_to(PROJECT_DIR))


# ------------------------------------------------------------------ background processes
def running(pid: int) -> bool:
    """Is the job still alive? Ended children are collected so they do not linger as zombies."""
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except (ChildProcessError, OSError):
        pass
    if sys.platform.startswith("win"):
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        code = ctypes.c_ulong()
        alive = bool(handle) and ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) \
            and code.value == 259
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
        return bool(alive)
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        try:
            return "\tZ" not in next(ln for ln in status.read_text().splitlines() if ln.startswith("State:"))
        except (OSError, StopIteration):
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop(pid: int) -> None:
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
