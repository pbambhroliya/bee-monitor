"""Command-line interface:  python -m beemon <command> --help   (or `beemon <command>` after pip install -e .)"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from beemon.config import PROJECT_DIR, load_config

DEFAULT_SPLIT = PROJECT_DIR / "configs" / "split.yaml"
DEFAULT_CLASSES = PROJECT_DIR / "configs" / "classes.yaml"
DEFAULT_VIDEOS_CSV = PROJECT_DIR / "videos.csv"


def _kv(items: list[str]) -> dict:
    import yaml

    result = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        result[key.strip()] = yaml.safe_load(raw)
    return result


# ------------------------------------------------------------------------------------------ commands
def cmd_process(a):
    from beemon.pipeline import build_classifier, build_detector, process_video
    from beemon.video import find_videos

    overrides = list(a.set or [])
    if a.weights:
        overrides.append(f"detector.weights={a.weights}")
    if a.tracker:
        overrides.append(f"tracker.type={a.tracker}")
    if a.no_video:
        overrides.append("output.save_video=false")
    if a.start_s is not None:
        overrides.append(f"video.start_s={a.start_s}")
    if a.end_s is not None:
        overrides.append(f"video.end_s={a.end_s}")
    cfg = load_config(a.config, overrides)
    videos = [v for source in a.videos for v in find_videos(source)]
    if not videos:
        raise SystemExit("No videos found")
    detector = build_detector(cfg)  # load the model once for all videos
    classifier = build_classifier(cfg)
    failed = []
    for video in videos:
        try:
            process_video(video, cfg, a.out, detector=detector, classifier=classifier,
                          name=(a.name if len(videos) == 1 else None))
        except Exception as error:  # keep going with the other videos
            print(f"ERROR processing {video}: {error}", file=sys.stderr)
            failed.append(str(video))
    if failed:
        raise SystemExit(f"{len(failed)} video(s) failed: {failed}")


def cmd_refilter(a):
    from beemon.pipeline import postprocess

    for result_dir in a.results:
        cfg = load_config(a.config, a.set)
        result = postprocess(result_dir, cfg, render=a.render)
        print(f"{result_dir}: {result['visits']} visits")


def cmd_render(a):
    from beemon.pipeline import postprocess

    cfg = load_config(a.config, a.set)
    postprocess(a.result, cfg, render=True)


def cmd_stats(a):
    from beemon.stats import collect_sessions

    sessions = collect_sessions(a.results, a.videos_csv)
    out = Path(a.results) / "sessions.csv"
    sessions.to_csv(out, index=False)
    print(sessions.to_string(index=False) if len(sessions) else "no results yet")
    print(f"\nwritten {out}")


def cmd_extract(a):
    from beemon.data.extract import extract_frames

    extract_frames(a.source, a.out, a.every_s, a.max_per_video, a.start_s, a.end_s, a.split, a.include_test,
                   a.prelabel, a.prelabel_conf, a.prelabel_imgsz, a.device, a.only_with_detections, a.empty_fraction,
                   only_test=a.only_test)


def cmd_clip(a):
    from beemon.data.extract import make_clip

    make_clip(a.video, a.out, a.start_s, a.duration_s, a.stride)


def cmd_motion_preview(a):
    from beemon.data.extract import motion_preview

    path = motion_preview(a.video, a.out, a.time_s, a.offsets, a.blur, local_kernel=a.local_kernel)
    print(f"written {path}")


def cmd_build(a):
    from beemon.data.dataset import build_dataset

    out = build_dataset(a.batches, a.out, a.classes, a.split, mode=a.mode, mie_offset=a.mie_offset, mie_blur=a.mie_blur,
                  videos_dir=a.videos_dir, single_class=a.single_class, mie_local_kernel=a.mie_local_kernel,
                  missing_as_empty=a.missing_as_empty, overwrite=a.overwrite)
    if a.zip:
        from beemon.data.bridge import zip_dataset

        zip_dataset(out, Path(str(out) + ".zip"))


def cmd_import_hardneg(a):
    from beemon.data.bridge import import_dataset_zip, import_session

    source = Path(a.source)
    out = a.out or str(PROJECT_DIR / "data" / "labeling" / f"hn_{source.stem[:12]}")
    if source.is_dir():
        import_session(source, out)
    else:
        import_dataset_zip(source, out, a.review_csv)
    print(f"Use it: add {out} to gold.train_batches in configs/workflow.yaml, "
          "or build a dataset with it: beemon build-dataset <batch(es)> --out data/datasets/<name>. Unfinished frames: finish them with  streamlit run beemon/labeler/app.py")


def cmd_tile(a):
    from beemon.data.dataset import tile_dataset

    tile_dataset(a.data, a.out, a.size, a.overlap, a.empty_keep)


def cmd_synth(a):
    from beemon.data.synth import make_synthetic

    make_synthetic(a.data, a.out, a.n_images, (a.min_per_image, a.max_per_image), a.backgrounds,
                   balance_classes=not a.no_balance, seed=a.seed)


def cmd_mine(a):
    from beemon.data.mine import mine_hard_examples

    mine_hard_examples(a.result, a.out, a.video, a.split, a.max_positive, a.max_negative, a.per_track,
                       include_test=a.include_test)


def cmd_crops(a):
    from beemon.data.crops import make_crops

    make_crops(a.data, a.out, a.size, a.pad, a.fp_weights, a.fp_conf, a.fp_imgsz)


def cmd_train(a):
    from beemon.train import train

    overrides = {k: getattr(a, k) for k in ("imgsz", "epochs", "patience", "batch", "lr0")}
    train(a.preset, a.data, a.weights, a.name, a.project, a.single_class, a.device, a.workers, a.cache, a.resume,
          not a.no_test, _kv(a.set), **overrides)


def cmd_hardneg_find(a):
    from ultralytics import YOLO

    from beemon.hardneg import find_pictures

    state = find_pictures(a.video_name, a.video, YOLO(a.weights), a.weights, every_s=a.every_s, want=a.want,
                          imgsz=a.imgsz, conf=a.conf, progress=lambda f, text: print(text, flush=True))
    print(f"{len(state['frames'])} pictures to check in contributions/work/{a.video_name}")


def cmd_clean_model(a):
    from beemon.train import clean_weights

    out = clean_weights(a.weights, a.out)
    print(f"ready to share: {out}  (no optimizer, no folder paths of the training computer)")


def cmd_val(a):
    from beemon.train import evaluate

    evaluate(a.weights, a.data, a.split, a.imgsz, a.device)


def cmd_eval_track(a):
    from beemon.evaluate import evaluate_tracking

    metrics = evaluate_tracking(a.gt, a.result, a.clip, a.max_dist, a.all_tracks)
    print(json.dumps(metrics, indent=2))
    if a.save:
        Path(a.save).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def cmd_eval_counts(a):
    from beemon.evaluate import evaluate_counts

    table = evaluate_counts(a.results, a.manual)
    print(table.to_string(index=False))
    if len(table.dropna()):
        print(f"\nmean absolute error: {table['error'].abs().mean():.2f} visits per video")


def cmd_workflow(a):
    from beemon import workflow as wfm

    wf = wfm.load_workflow(a.workflow)
    action = a.action
    if action == "status":
        print(wfm.status(wf))
    elif action == "report":
        table = wfm.report(wf)
        print(table.to_string(index=False))
    elif action == "count":
        if a.what == "candidates":
            print(len(wf["base"]["candidates"]))
        else:
            print(len(wfm.round_videos(wf, _need_round(a))))
    elif action == "base-train":
        wfm.base_train(wf, a.index, a.candidate, a.device, a.workers)
    elif action == "base-select":
        wfm.base_select(wf, a.device)
    elif action == "set-base":
        if not a.weights:
            raise SystemExit("set-base needs --weights path/to/model.pt")
        wfm.set_base(a.weights)
    elif action == "videos":
        for video in wfm.round_videos(wf, _need_round(a)):
            print(video)
    elif action == "process":
        wfm.round_process(wf, _need_round(a), a.index, a.device, a.force)
    elif action == "label":
        wfm.round_label(wf, _need_round(a))
    elif action == "quality":
        wfm.round_quality(wf, _need_round(a))
    elif action == "train":
        wfm.round_train(wf, _need_round(a), a.device, a.workers, a.skip_review)


def _need_round(a) -> str:
    if not a.round:
        raise SystemExit(f"`workflow {a.action}` needs --round NAME (see configs/workflow.yaml)")
    return a.round


def cmd_labeling_package(a):
    from beemon.data.share import make_package

    out = a.out or str(PROJECT_DIR / "share" / f"{Path(a.batch).name}.zip")
    make_package(a.batch, out)


def cmd_labeling_import(a):
    from beemon.data.share import import_labels

    import_labels(a.zips, a.batch)


def cmd_download(a):
    from beemon.download import download_share

    download_share(a.url, a.out, password=a.password, pattern=a.pattern, dry_run=a.list)


# ------------------------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="beemon", description="Bee monitoring pipeline. Step-by-step guide: README.md")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("process", help="detect + track + filter + statistics for video(s)")
    s.add_argument("videos", nargs="+", help="video files or folders")
    s.add_argument("--weights", help="detector .pt (overrides detector.weights)")
    s.add_argument("--config", help="config YAML (default: configs/pipeline.yaml)")
    s.add_argument("--out", default=str(PROJECT_DIR / "results"))
    s.add_argument("--name", help="result folder name (single video only; default: video name)")
    s.add_argument("--tracker", choices=["beetrack", "bytetrack"])
    s.add_argument("--start-s", type=float)
    s.add_argument("--end-s", type=float)
    s.add_argument("--no-video", action="store_true", help="skip the annotated video")
    s.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE", help="override any config value")
    s.set_defaults(func=cmd_process)

    s = sub.add_parser("refilter", help="re-apply filters/statistics to processed results (no detection)")
    s.add_argument("results", nargs="+", help="result folders")
    s.add_argument("--config")
    s.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE")
    s.add_argument("--render", action="store_true", help="also re-render the annotated video")
    s.set_defaults(func=cmd_refilter)

    s = sub.add_parser("render", help="(re)create the annotated video of a result folder")
    s.add_argument("result")
    s.add_argument("--config")
    s.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE")
    s.set_defaults(func=cmd_render)

    s = sub.add_parser("stats", help="table of all processed videos (results/sessions.csv)")
    s.add_argument("--results", default=str(PROJECT_DIR / "results"))
    s.add_argument("--videos-csv", default=str(DEFAULT_VIDEOS_CSV))
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("extract", help="sample frames for labelling (optionally with model pre-labels)")
    s.add_argument("source", help="video file or folder")
    s.add_argument("--out", required=True, help="labelling batch folder, e.g. data/labeling/batch01")
    s.add_argument("--every-s", type=float, default=3.0)
    s.add_argument("--max-per-video", type=int, default=80)
    s.add_argument("--start-s", type=float, default=0.0)
    s.add_argument("--end-s", type=float)
    s.add_argument("--split", default=str(DEFAULT_SPLIT), help="split file: test videos are skipped")
    s.add_argument("--include-test", action="store_true", help="also extract from test videos")
    s.add_argument("--only-test", action="store_true", help="only the test videos (for the gold test set)")
    s.add_argument("--prelabel", help="detector .pt used to pre-label the frames for CVAT")
    s.add_argument("--prelabel-conf", type=float, default=0.25)
    s.add_argument("--prelabel-imgsz", type=int, default=1280)
    s.add_argument("--device")
    s.add_argument("--only-with-detections", action="store_true",
                   help="keep frames where the pre-label model sees something (+ some empty ones)")
    s.add_argument("--empty-fraction", type=float, default=0.2)
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("clip", help="export a short clip as frames for TRACK annotation in CVAT")
    s.add_argument("video")
    s.add_argument("--out", required=True)
    s.add_argument("--start-s", type=float, required=True)
    s.add_argument("--duration-s", type=float, default=20.0)
    s.add_argument("--stride", type=int, default=2, help="use the same frame_stride as for processing")
    s.set_defaults(func=cmd_clip)

    s = sub.add_parser("motion-preview", help="picture comparing motion offsets (to choose motion.offset_frames)")
    s.add_argument("video")
    s.add_argument("--out", default=str(PROJECT_DIR / "results" / "motion_preview.jpg"))
    s.add_argument("--time-s", type=float, default=10.0)
    s.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 4, 8])
    s.add_argument("--blur", type=int, default=5)
    s.add_argument("--local-kernel", type=int, default=0, help="also show local motion (e.g. 101)")
    s.set_defaults(func=cmd_motion_preview)

    s = sub.add_parser("build-dataset", help="labelled batches -> YOLO dataset split by video (rgb or mie)")
    s.add_argument("batches", nargs="+", help="labelling batch folders (with images/, manifest.csv, CVAT export)")
    s.add_argument("--out", required=True)
    s.add_argument("--mode", choices=["rgb", "mie"], default="rgb")
    s.add_argument("--mie-offset", type=int, default=4)
    s.add_argument("--mie-blur", type=int, default=5)
    s.add_argument("--mie-local-kernel", type=int, default=0, help="experimental local motion (see config)")
    s.add_argument("--videos-dir", default=str(PROJECT_DIR / "videos"), help="source videos (mie mode)")
    s.add_argument("--classes", default=str(DEFAULT_CLASSES))
    s.add_argument("--split", default=str(DEFAULT_SPLIT))
    s.add_argument("--single-class", action="store_true", help="merge all labels into one class 'insect'")
    s.add_argument("--missing-as-empty", action="store_true")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--zip", action="store_true", help="also write <out>.zip, portable (for the old training scripts)")
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("import-hardneg", help="import hard-negative work: a YOLO dataset ZIP or a review session folder")
    s.add_argument("source", help="dataset .zip (images/, labels/, data.yaml) or a review session folder with review.json")
    s.add_argument("--out", help="new labelling batch (default: data/labeling/hn_<name>)")
    s.add_argument("--review-csv", help="(zip only) review_report.csv of the same session: recovers video names")
    s.set_defaults(func=cmd_import_hardneg)

    s = sub.add_parser("tile", help="cut a dataset into overlapping tiles (SAHI-style training)")
    s.add_argument("data", help="data.yaml")
    s.add_argument("--out", required=True)
    s.add_argument("--size", type=int, default=960)
    s.add_argument("--overlap", type=float, default=0.2)
    s.add_argument("--empty-keep", type=float, default=0.25, help="fraction of insect-free tiles to keep")
    s.set_defaults(func=cmd_tile)

    s = sub.add_parser("synth", help="copy-paste synthetic images (insects pasted on empty frames)")
    s.add_argument("data", help="data.yaml of the dataset to extend")
    s.add_argument("--out", required=True)
    s.add_argument("--n-images", type=int, default=500)
    s.add_argument("--min-per-image", type=int, default=1)
    s.add_argument("--max-per-image", type=int, default=4)
    s.add_argument("--backgrounds", help="extra folder of insect-free frames")
    s.add_argument("--no-balance", action="store_true", help="sample classes by frequency instead of uniformly")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("mine", help="hard positives/negatives from a processed video -> CVAT review batch")
    s.add_argument("result", help="result folder of a processed video")
    s.add_argument("--out", required=True, help="new labelling batch folder, e.g. data/labeling/mined01")
    s.add_argument("--video", help="source video (default: path stored in the result)")
    s.add_argument("--split", default=str(DEFAULT_SPLIT))
    s.add_argument("--max-positive", type=int, default=60)
    s.add_argument("--max-negative", type=int, default=60)
    s.add_argument("--per-track", type=int, default=2)
    s.add_argument("--include-test", action="store_true")
    s.set_defaults(func=cmd_mine)

    s = sub.add_parser("crops", help="crop dataset for the optional species classifier")
    s.add_argument("data", help="data.yaml of an RGB dataset")
    s.add_argument("--out", required=True)
    s.add_argument("--size", type=int, default=128)
    s.add_argument("--pad", type=float, default=0.25)
    s.add_argument("--fp-weights", help="RGB detector .pt: its false positives become class not_insect")
    s.add_argument("--fp-conf", type=float, default=0.3)
    s.add_argument("--fp-imgsz", type=int, default=1280)
    s.set_defaults(func=cmd_crops)

    s = sub.add_parser("train", help="train a model (presets: pretrain, finetune, classify)")
    s.add_argument("--preset", required=True, choices=["pretrain", "finetune", "classify"])
    s.add_argument("--data", required=True, help="data.yaml (detection) or crop folder (classify)")
    s.add_argument("--weights", help="starting weights (default: yolo11s.pt / yolo11s-cls.pt)")
    s.add_argument("--name", help="run name (default: <weights>_<preset>)")
    s.add_argument("--project", default=str(PROJECT_DIR / "runs"))
    s.add_argument("--imgsz", type=int)
    s.add_argument("--epochs", type=int)
    s.add_argument("--patience", type=int)
    s.add_argument("--batch", type=float, help="batch size, or 0-1 = fraction of GPU memory")
    s.add_argument("--lr0", type=float)
    s.add_argument("--single-class", action="store_true", help="treat all classes as one ('insect')")
    s.add_argument("--device")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--cache", choices=["disk", "ram", "none"], default="disk")
    s.add_argument("--resume", action="store_true", help="resume an interrupted run (--weights .../last.pt)")
    s.add_argument("--no-test", action="store_true", help="skip test-split evaluation after training")
    s.add_argument("--set", action="append", metavar="KEY=VALUE", help="any other Ultralytics train argument")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("hardneg-find", help="let the model look at a video: pictures with boxes to check (app page)")
    s.add_argument("video_name", help="unique video name (videos.csv)")
    s.add_argument("--video", required=True, help="video file")
    s.add_argument("--weights", required=True)
    s.add_argument("--want", type=int, default=30, help="stop when this many pictures with boxes exist")
    s.add_argument("--every-s", type=float, default=10.0)
    s.add_argument("--imgsz", type=int, default=1280)
    s.add_argument("--conf", type=float, default=0.25)
    s.set_defaults(func=cmd_hardneg_find)

    s = sub.add_parser("clean-model", help="make a trained .pt safe to share (removes paths of the training computer)")
    s.add_argument("weights")
    s.add_argument("--out", help="new file (default: overwrite)")
    s.set_defaults(func=cmd_clean_model)

    s = sub.add_parser("val", help="detection metrics of a model on a dataset split")
    s.add_argument("--weights", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--split", default="test")
    s.add_argument("--imgsz", type=int, default=1280)
    s.add_argument("--device")
    s.set_defaults(func=cmd_val)

    s = sub.add_parser("eval-track", help="tracking metrics (MOTA, IDF1) against a CVAT MOT 1.1 export")
    s.add_argument("--gt", required=True, help="gt.txt (or the unzipped CVAT MOT export folder)")
    s.add_argument("--result", required=True, help="result folder of the processed video")
    s.add_argument("--clip", help="clip.json from `beemon clip` (maps clip frames to video frames)")
    s.add_argument("--max-dist", type=float, default=30.0, help="max centre distance (px) for a match")
    s.add_argument("--all-tracks", action="store_true", help="evaluate all tracks, not only valid ones")
    s.add_argument("--save", help="write the metrics to this JSON file")
    s.set_defaults(func=cmd_eval_track)

    s = sub.add_parser("workflow", help="training workflow: base model -> rounds of auto-labelling -> final model")
    s.add_argument("action", choices=["status", "report", "count", "base-train", "base-select", "set-base", "videos",
                                      "process", "label", "quality", "train"])
    s.add_argument("--round", help="round name from configs/workflow.yaml (e.g. r1)")
    s.add_argument("--index", type=int, help="only this candidate / video (SLURM array task id)")
    s.add_argument("--candidate", help="base-train: one candidate, e.g. yolo11m.pt")
    s.add_argument("--weights", help="set-base: model to use as base")
    s.add_argument("--what", choices=["candidates", "videos"], default="videos", help="count: what to count")
    s.add_argument("--device")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--force", action="store_true", help="process: redo videos that are already processed")
    s.add_argument("--skip-review", action="store_true", help="train: do not wait for the CVAT review")
    s.add_argument("--workflow", help="workflow file (default: configs/workflow.yaml)")
    s.set_defaults(func=cmd_workflow)

    s = sub.add_parser("labeling-package", help="zip a picture set + the simple labeller app for a teammate")
    s.add_argument("batch", help="labelling batch folder, e.g. data/labeling/gold_test or data/labeling/r2_auto")
    s.add_argument("--out", help="zip file (default: share/<batch>.zip)")
    s.set_defaults(func=cmd_labeling_package)

    s = sub.add_parser("labeling-import", help="import the label zips teammates sent back")
    s.add_argument("zips", nargs="+", help="zip files from 'Download my labels'")
    s.add_argument("--batch", required=True, help="the labelling batch folder they belong to")
    s.set_defaults(func=cmd_labeling_import)

    s = sub.add_parser("download", help="download videos from a Sciebo / ownCloud / Nextcloud public share link")
    s.add_argument("url", help="share link, e.g. https://uni-bonn.sciebo.de/s/AbCdEf (may contain ?path=/sub)")
    s.add_argument("--out", default=str(PROJECT_DIR / "videos"))
    s.add_argument("--password", help="password of the share, if it has one")
    s.add_argument("--pattern", default="*", help="only files matching this pattern, e.g. '*2024*' or 'RIMG00*'")
    s.add_argument("--list", action="store_true", help="only list the files, do not download")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("eval-counts", help="compare visit counts with manual counts (CSV: video,visits)")
    s.add_argument("--manual", required=True)
    s.add_argument("--results", default=str(PROJECT_DIR / "results"))
    s.set_defaults(func=cmd_eval_counts)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
