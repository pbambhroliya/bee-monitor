"""The training workflow, step by step (configured in configs/workflow.yaml, progress in workflow/state.json).

    base-train    train one/all candidate YOLO models on BuzzSet                  (cluster, GPU)
    base-select   evaluate the candidates, copy the best to models/base.pt
    process       the current model processes the videos of a round (tracking)   (cluster, GPU)
    label         turn the processed videos into an auto-labelled batch + review package
    quality       after the CVAT review: how good were the automatic labels?
    train         build the round dataset (gold + auto + reviewed) and fine-tune  (cluster, GPU)
    status/report where are we, and how did every model score on the gold test set

On a GPU cluster, chain these steps as batch jobs (one job per step / per video).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from beemon.config import PROJECT_DIR, load_config, read_yaml

WORKFLOW_FILE = PROJECT_DIR / "configs" / "workflow.yaml"
STATE_DIR = PROJECT_DIR / "workflow"
STATE_FILE = STATE_DIR / "state.json"


# ------------------------------------------------------------------------------------- config & state
def p(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def load_workflow(path: str | Path | None = None) -> dict:
    return read_yaml(path or WORKFLOW_FILE)


def load_state() -> dict:
    if STATE_FILE.is_file():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"base": {}, "models": {}, "rounds": {}}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _class_names(wf: dict) -> list[str]:
    from beemon.data.common import load_class_names

    return load_class_names(p(wf["classes"]))


def round_config(wf: dict, name: str) -> tuple[int, dict]:
    for index, rnd in enumerate(wf.get("rounds") or []):
        if rnd["name"] == name:
            return index, rnd
    raise ValueError(f"Round {name!r} is not in configs/workflow.yaml (rounds: {[r['name'] for r in wf['rounds']]})")


def auto_batch(name: str) -> Path:
    return PROJECT_DIR / "data" / "labeling" / f"{name}_auto"


def round_results(name: str) -> Path:
    return PROJECT_DIR / "results" / "rounds" / name


# ------------------------------------------------------------------------------------- stage 1: base
def candidate_weights(wf: dict, candidate: str) -> str:
    local = p(wf["base"].get("weights_dir", "models/pretrained")) / candidate
    return str(local) if local.is_file() else candidate


def base_train(wf: dict, index: int | None = None, candidate: str | None = None, device=None, workers: int = 8,
               log: Callable[[str], None] = print) -> list[Path]:
    from beemon.train import train

    b = wf["base"]
    candidates = b["candidates"]
    if candidate:
        selected = [candidate]
    elif index is not None:
        selected = [candidates[int(index)]]
    else:
        selected = candidates
    outputs = []
    for cand in selected:
        name = Path(cand).stem
        log(f"=== base candidate {name} on {b['data']}")
        best = train("pretrain", str(p(b["data"])), candidate_weights(wf, cand), name=name,
                     project=str(PROJECT_DIR / "runs" / "base"), device=device, workers=workers, test_after=False,
                     imgsz=b.get("imgsz", 960), epochs=b.get("epochs", 150), patience=b.get("patience", 30),
                     batch=b.get("batch", 0.7))
        outputs.append(best)
    return outputs


def build_gold_test(wf: dict, log: Callable[[str], None] = print) -> Path | None:
    """Dataset of the hand-labelled TEST frames (used to score every model)."""
    from beemon.data.dataset import build_dataset

    batches = [p(b) for b in (wf.get("gold") or {}).get("test_batches") or [] if (p(b) / "images").is_dir()]
    if not batches:
        return None
    out = PROJECT_DIR / "data" / "datasets" / "gold_test"
    build_dataset(batches, out, p(wf["classes"]), p(wf["split"]), overwrite=True, missing_as_empty=False,
                  human_only=batches, log=lambda *_: None)
    info = read_yaml(out / "dataset_info.yaml")
    if not info["images"].get("test"):
        log("WARNING: gold test batches contain no frames of TEST videos (configs/split.yaml) — not used")
        return None
    return out / "data.yaml"


def _model_info(weights: Path) -> dict:
    from ultralytics import YOLO

    try:
        layers, params, _, gflops = YOLO(str(weights)).info(verbose=False)
        return {"params_M": round(params / 1e6, 1), "GFLOPs": round(float(gflops), 1)}
    except Exception:
        return {}


def base_select(wf: dict, device=None, log: Callable[[str], None] = print) -> Path:
    from beemon.train import evaluate

    b = wf["base"]
    metric = b.get("select_metric", "mAP50-95")
    gold = build_gold_test(wf, log)
    rows = []
    for cand in b["candidates"]:
        name = Path(cand).stem
        runs = sorted((PROJECT_DIR / "runs" / "base").glob(f"{name}*/weights/best.pt"),
                      key=lambda f: f.stat().st_mtime)
        if not runs:
            log(f"{name}: not trained (no runs/base/{name}/weights/best.pt) — skipped")
            continue
        best = runs[-1]
        row = {"candidate": name, "weights": str(best.relative_to(PROJECT_DIR)), **_model_info(best)}
        for k, v in evaluate(best, p(b["data"]), "test", b.get("imgsz", 960), device).items():
            row[f"buzzset_{k}"] = v
        if gold:
            for k, v in evaluate(best, gold, "test", wf["finetune"].get("imgsz", 1280), device).items():
                row[f"gold_{k}"] = v
        rows.append(row)
    if not rows:
        raise SystemExit("No trained base candidates found. Run `beemon workflow base-train` first.")
    table = pd.DataFrame(rows)
    key = f"gold_{metric}" if gold and f"gold_{metric}" in table else f"buzzset_{metric}"
    table = table.sort_values(key, ascending=False).reset_index(drop=True)
    within = float(b.get("prefer_smaller_within", 0.0) or 0.0)
    chosen = table.iloc[0]
    if within > 0 and "params_M" in table:
        close = table[table[key] >= chosen[key] - within]
        chosen = close.sort_values("params_M").iloc[0]
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(STATE_DIR / "base_selection.csv", index=False)
    target = PROJECT_DIR / "models" / "base.pt"
    shutil.copy2(PROJECT_DIR / chosen["weights"], target)
    state = load_state()
    state["base"] = {"chosen": chosen["candidate"], "metric": key, "score": float(chosen[key]),
                     "table": str((STATE_DIR / "base_selection.csv").relative_to(PROJECT_DIR)),
                     "time": time.strftime("%Y-%m-%d %H:%M")}
    state["models"]["base"] = {"path": "models/base.pt", "input": "rgb", "test": _test_metrics(chosen, gold)}
    save_state(state)
    log(table.to_string(index=False))
    log(f"\nBase model: {chosen['candidate']} ({key} = {chosen[key]:.3f}) -> models/base.pt")
    return target


def _test_metrics(row, gold) -> dict:
    prefix = "gold_" if gold else "buzzset_"
    return {k[len(prefix):]: float(v) for k, v in row.items() if str(k).startswith(prefix)}


# ------------------------------------------------------------------------------------- rounds
def teacher_for(wf: dict, state: dict, name: str) -> tuple[str, dict]:
    """The model that labels round ``name``: the previous round's model, or the base model."""
    index, _ = round_config(wf, name)
    for previous in reversed(wf["rounds"][:index]):
        if previous["name"] in state["models"]:
            return previous["name"], state["models"][previous["name"]]
    if "base" not in state["models"]:
        raise SystemExit("No base model yet. Train a base model (docs/TRAINING.md) or copy a model to models/base.pt "
                         "and run `beemon workflow set-base models/base.pt`.")
    return "base", state["models"]["base"]


def round_videos(wf: dict, name: str) -> list[Path]:
    from beemon.data.common import load_split
    from beemon.video import find_videos

    _, rnd = round_config(wf, name)
    test = set(load_split(p(wf["split"]))["test_videos"])
    folder = p(wf["videos_dir"])
    videos = [v for v in find_videos(folder) if v.stem not in test] if folder.is_dir() else []
    choice = rnd.get("videos", "all")
    if isinstance(choice, list):
        wanted = {str(v) for v in choice}
        videos = [v for v in videos if v.stem in wanted]
    elif str(choice) != "all":
        n = int(choice)
        if len(videos) > n:  # evenly spread over the (sorted) list -> covers different days
            idx = np.linspace(0, len(videos) - 1, n).round().astype(int)
            videos = [videos[i] for i in sorted(set(idx))]
    return videos


def round_process(wf: dict, name: str, index: int | None = None, device=None, force: bool = False,
                  log: Callable[[str], None] = print) -> None:
    from beemon.pipeline import build_detector, process_video

    state = load_state()
    teacher_name, teacher = teacher_for(wf, state, name)
    videos = round_videos(wf, name)
    if not videos:
        raise SystemExit(f"No videos for round {name} in {p(wf['videos_dir'])} (test videos are excluded).")
    if index is not None:
        if int(index) >= len(videos):
            log(f"index {index} >= {len(videos)} videos: nothing to do")
            return
        videos = [videos[int(index)]]
    overrides = [f"detector.weights={p(teacher['path'])}", f"detector.input={teacher.get('input', 'rgb')}"]
    if device is not None:
        overrides.append(f"detector.device={device}")
    if teacher.get("input") == "mie":
        overrides.append(f"motion.offset_frames={teacher.get('mie_offset', 4)}")
    cfg = load_config(p(wf["teacher"]["process_config"]), overrides)
    detector = build_detector(cfg)
    out_root = round_results(name)
    log(f"round {name}: {len(videos)} video(s) labelled by '{teacher_name}' ({teacher['path']})")
    for video in videos:
        if (out_root / video.stem / "summary.json").is_file() and not force:
            log(f"{video.stem}: already processed (use --force to redo)")
            continue
        process_video(video, cfg, out_root, detector=detector, classifier=None, log=log)


def round_label(wf: dict, name: str, log: Callable[[str], None] = print) -> dict:
    from beemon.data.autolabel import autolabel_results

    results = sorted(d.parent for d in round_results(name).glob("*/summary.json"))
    if not results:
        raise SystemExit(f"No processed videos in {round_results(name)}. Run `beemon workflow process` first.")
    out = auto_batch(name)
    if (out / "labels_auto").is_dir() and any((out / "labels_auto").glob("*.txt")):
        raise SystemExit(f"{out} already contains automatic labels. Delete the folder to label the round again.")
    teacher_cfg = {k: v for k, v in wf["teacher"].items() if k != "process_config"}
    info = autolabel_results(results, out, _class_names(wf), p(wf["split"]), teacher_cfg, log=log)
    state = load_state()
    state["rounds"].setdefault(name, {})["labelled"] = {**info, "videos": len(results),
                                                       "time": time.strftime("%Y-%m-%d %H:%M")}
    save_state(state)
    _, rnd = round_config(wf, name)
    if rnd.get("review"):
        log(f"\nReview needed before training: see {out / 'review' / 'HOW_TO_REVIEW.md'}")
    return info


def round_quality(wf: dict, name: str, log: Callable[[str], None] = print) -> dict:
    from beemon.data.autolabel import label_quality

    quality = label_quality(auto_batch(name))
    state = load_state()
    state["rounds"].setdefault(name, {})["label_quality"] = quality
    save_state(state)
    log(json.dumps(quality, indent=2))
    return quality


def round_train(wf: dict, name: str, device=None, workers: int = 8, skip_review: bool = False,
                log: Callable[[str], None] = print) -> Path:
    from beemon.data.common import find_label_files
    from beemon.data.dataset import build_dataset
    from beemon.train import evaluate, train

    index, rnd = round_config(wf, name)
    batch = auto_batch(name)
    if not (batch / "labels_auto").is_dir():
        raise SystemExit(f"Round {name} is not labelled yet (`beemon workflow label --round {name}`).")
    reviewed = any(not auto for _, _, auto in find_label_files(batch).values())
    if rnd.get("review") and not reviewed and not skip_review:
        raise SystemExit(f"Round {name} needs the CVAT review first ({batch / 'review' / 'HOW_TO_REVIEW.md'}), "
                         "or run with --skip-review.")
    gold = wf.get("gold") or {}
    previous_auto = [auto_batch(r["name"]) for r in wf["rounds"][:index] if auto_batch(r["name"]).is_dir()]
    batches = [p(b) for b in (gold.get("train_batches") or []) + (gold.get("test_batches") or [])
               if (p(b) / "images").is_dir()] + previous_auto + [batch]
    ft = wf["finetune"]
    mode = ft.get("mode", "rgb")
    data_dir = PROJECT_DIR / "data" / "datasets" / name
    gold_batches = [b for b in batches if b not in previous_auto and b != batch]
    build_dataset(batches, data_dir, p(wf["classes"]), p(wf["split"]), mode=mode, mie_offset=ft.get("mie_offset", 4),
                  videos_dir=p(wf["videos_dir"]), overwrite=True, human_only=previous_auto + gold_batches, log=log)

    state = load_state()
    teacher_name, teacher = teacher_for(wf, state, name)
    start = state["models"]["base"] if ft.get("from", "previous") == "base" else teacher
    log(f"fine-tuning {start['path']} on {data_dir}")
    best = train("finetune", str(data_dir / "data.yaml"), str(p(start["path"])), name=name,
                 project=str(PROJECT_DIR / "runs" / "rounds"), device=device, workers=workers, test_after=False,
                 imgsz=ft.get("imgsz", 1280), epochs=ft.get("epochs", 80), patience=ft.get("patience", 20),
                 lr0=ft.get("lr0", 0.002), batch=ft.get("batch", 0.7))
    metrics = evaluate(best, data_dir / "data.yaml", "test", ft.get("imgsz", 1280), device)
    split_used = "test (gold)"
    if not metrics:
        metrics = evaluate(best, data_dir / "data.yaml", "val", ft.get("imgsz", 1280), device)
        split_used = "val (partly automatic labels — optimistic)"
    target = PROJECT_DIR / "models" / f"{name}.pt"
    shutil.copy2(best, target)
    shutil.copy2(best, PROJECT_DIR / "models" / "detector.pt")
    state = load_state()
    state["models"][name] = {"path": f"models/{name}.pt", "input": mode, "mie_offset": ft.get("mie_offset", 4),
                             "test": metrics, "evaluated_on": split_used, "trained_from": start["path"],
                             "dataset": str(data_dir.relative_to(PROJECT_DIR))}
    state["rounds"].setdefault(name, {})["trained"] = time.strftime("%Y-%m-%d %H:%M")
    save_state(state)
    previous = {}
    if split_used.startswith("test") and teacher.get("input", "rgb") == mode:  # same test frames, fair comparison
        previous = evaluate(p(teacher["path"]), data_dir / "data.yaml", "test", ft.get("imgsz", 1280), device)
    log(f"\nRound {name}: {metrics} on {split_used}")
    if previous and metrics:
        delta = metrics.get("mAP50-95", 0) - previous.get("mAP50-95", 0)
        log(f"compared with '{teacher_name}': mAP50-95 {delta:+.3f}"
            + ("  -> keep going with the next round" if delta > 0.005 else "  -> little gain: stop or review more"))
    log(f"models/{name}.pt and models/detector.pt updated")
    return target


def set_base(path: str | Path, log: Callable[[str], None] = print) -> None:
    """Use an existing model as base (skips the base stage)."""
    source = Path(path)
    target = PROJECT_DIR / "models" / "base.pt"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    state = load_state()
    state["base"] = {"chosen": f"manual: {source.name}", "time": time.strftime("%Y-%m-%d %H:%M")}
    state["models"]["base"] = {"path": "models/base.pt", "input": "rgb", "test": {}}
    save_state(state)
    log(f"base model set to {source} -> models/base.pt")


# ------------------------------------------------------------------------------------- overview
def report(wf: dict) -> pd.DataFrame:
    state = load_state()
    rows = []
    for name in ["base"] + [r["name"] for r in wf.get("rounds") or []]:
        model = state["models"].get(name)
        rnd = state["rounds"].get(name, {})
        rows.append({
            "model": name,
            "file": model["path"] if model else "-",
            "input": model.get("input") if model else "-",
            **{f"test_{k}": v for k, v in ((model or {}).get("test") or {}).items()},
            "auto_frames": (rnd.get("labelled") or {}).get("auto_labelled", "-"),
            "label_precision": (rnd.get("label_quality") or {}).get("box_precision", "-"),
            "label_recall": (rnd.get("label_quality") or {}).get("box_recall", "-"),
        })
    return pd.DataFrame(rows)


def status(wf: dict) -> str:
    state = load_state()
    lines = []
    if "base" in state["models"]:
        lines.append(f"[x] base model: {state['base'].get('chosen')} -> models/base.pt")
    else:
        trained = sorted(f.parts[-3] for f in (PROJECT_DIR / "runs" / "base").glob("*/weights/best.pt"))
        lines.append(f"[ ] base model (trained candidates: {trained or 'none'})")
        lines.append("    next: beemon workflow base-train   (then: beemon workflow base-select)")
        return "\n".join(lines)
    for rnd in wf.get("rounds") or []:
        name = rnd["name"]
        done_videos = len(list(round_results(name).glob("*/summary.json")))
        total = len(round_videos(wf, name))
        labelled = (auto_batch(name) / "labels_auto").is_dir()
        trained = name in state["models"]
        if total == 0:
            lines.append(f"[ ] round {name}: no videos in {p(wf['videos_dir'])} (download them, see README)")
            break
        mark = "x" if trained else " "
        lines.append(f"[{mark}] round {name}: processed {done_videos}/{total} videos, "
                     f"labelled {'yes' if labelled else 'no'}, trained {'yes' if trained else 'no'}")
        if not trained:
            if done_videos < total:
                lines.append(f"    next: beemon workflow process --round {name}   (then: label, train)")
            elif not labelled:
                lines.append(f"    next: beemon workflow label --round {name}")
            elif rnd.get("review"):
                lines.append(f"    next: review in CVAT ({auto_batch(name) / 'review' / 'HOW_TO_REVIEW.md'}), "
                             f"then beemon workflow train --round {name}")
            else:
                lines.append(f"    next: beemon workflow train --round {name}")
            break
    else:
        lines.append("All rounds done: models/detector.pt is the final model -> use it in the app (Track a video).")
    return "\n".join(lines)
