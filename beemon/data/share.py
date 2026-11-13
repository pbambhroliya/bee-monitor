"""Hand a labelling batch to a teammate and take the labels back.

    beemon labeling-package data/labeling/gold_test --out share/gold_test.zip
        -> one zip: the small labeller app + start scripts + START_HERE.md + the pictures (+ suggestions)
    beemon labeling-import labels_gold_test_Aniket.zip ... --batch data/labeling/gold_test
        -> the teammates' labels go into <batch>/labels_human/ (used by build-dataset / the workflow)
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Callable

LABELER_DIR = Path(__file__).resolve().parents[1] / "labeler"
APP_FILES = ["app.py", "labeler_ui.py", "labeling.py", "bee_canvas.py"]


def make_package(batch: str | Path, out_zip: str | Path, log: Callable[[str], None] = print) -> Path:
    batch, out_zip = Path(batch), Path(out_zip)
    images = sorted((batch / "images").glob("*.jpg")) + sorted((batch / "images").glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No pictures in {batch / 'images'}")
    review_list = batch / "review" / "review_list.txt"
    if review_list.is_file():  # review batch: only the pictures that need a human
        wanted = {Path(n.strip()).stem for n in review_list.read_text().splitlines() if n.strip()}
        images = [p for p in images if p.stem in wanted]
    top = "bee-labeller"
    data = f"{top}/data/{batch.name}"
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as archive:  # JPEGs do not compress further
        for name in APP_FILES:
            archive.write(LABELER_DIR / name, f"{top}/labeler/{name}")
        archive.write(LABELER_DIR / "START_HERE.md", f"{top}/START_HERE.md")
        for script in ("run_labeller.sh", "run_labeller.bat"):
            info = zipfile.ZipInfo(f"{top}/{script}")
            info.external_attr = 0o755 << 16
            archive.writestr(info, (LABELER_DIR / script).read_bytes())
        archive.writestr(f"{top}/requirements.txt", "streamlit>=1.50\n")
        for image in images:
            archive.write(image, f"{data}/images/{image.name}")
        for extra in ("cvat_prelabels.zip", "review/review_list.txt", "review/cvat_review.zip"):
            if (batch / extra).is_file():
                archive.write(batch / extra, f"{data}/{extra}")
        stems = {p.stem for p in images}
        for folder in ("labels_auto", "labels_suggested"):
            auto = batch / folder
            if auto.is_dir():
                for label in auto.glob("*.txt"):
                    if label.stem in stems:
                        archive.write(label, f"{data}/{folder}/{label.name}")
                if (auto / "obj.names").is_file():
                    archive.write(auto / "obj.names", f"{data}/{folder}/obj.names")
    size_mb = out_zip.stat().st_size / 1e6
    log(f"package for teammates: {out_zip} ({len(images)} pictures, {size_mb:.0f} MB)\n"
        "They unzip it and follow START_HERE.md; they send back the zip from 'Download my labels'.")
    return out_zip


def import_labels(zips: list[str | Path], batch: str | Path, log: Callable[[str], None] = print) -> int:
    batch = Path(batch)
    target = batch / "labels_human"
    target.mkdir(parents=True, exist_ok=True)
    known = {p.stem for p in (batch / "images").glob("*")}
    total = 0
    for zip_path in map(Path, zips):
        new = updated = unknown = 0
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            info = archive.read("batch.txt").decode().splitlines() if "batch.txt" in names else []
            if info and info[0] != batch.name:
                log(f"WARNING: {zip_path.name} was made for picture set '{info[0]}', importing into '{batch.name}'")
            log_lines = []
            for name in names:
                path = Path(name)
                if path.parent.name != "labels_human":
                    continue
                if path.name == "obj.names":
                    (target / "obj.names").write_bytes(archive.read(name))
                elif path.name == "log.csv":
                    log_lines = archive.read(name).decode().splitlines()[1:]
                elif path.suffix == ".txt":
                    if path.stem not in known:
                        unknown += 1
                        continue
                    content = archive.read(name)
                    dest = target / path.name
                    if dest.is_file():
                        if dest.read_bytes() != content:
                            updated += 1
                    else:
                        new += 1
                    dest.write_bytes(content)
            if log_lines:
                log_file = target / "log.csv"
                with log_file.open("a") as handle:
                    if log_file.stat().st_size == 0:
                        handle.write("image,annotator,boxes,time\n")
                    handle.write("\n".join(log_lines) + "\n")
        who = info[1] if len(info) > 1 else "?"
        log(f"{zip_path.name} ({who}): {new} new, {updated} changed" + (f", {unknown} not in this set" if unknown else ""))
        total += new + updated
    done = len(list(target.glob("*.txt")))
    log(f"{batch.name}: {done} labelled pictures in total")
    return total
