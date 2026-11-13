"""Download videos from a public share link (Sciebo runs ownCloud; Nextcloud links work too).

    python -m beemon download https://uni-bonn.sciebo.de/s/AbCdEf --list
    python -m beemon download https://uni-bonn.sciebo.de/s/AbCdEf --pattern "*.MOV" --out data/videos

Uses the WebDAV interface every public share has. Files that are already complete are skipped,
interrupted downloads are resumed. Only video files are downloaded.
"""

from __future__ import annotations

import fnmatch
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote, unquote, urlparse

from beemon.video import VIDEO_EXTENSIONS

DAV = "{DAV:}"


def _parse_share(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url)
    match = re.search(r"/s/([A-Za-z0-9]+)", parsed.path)
    if not match:
        raise ValueError(f"Not a public share link (expected .../s/<token>): {url}")
    sub = parse_qs(parsed.query).get("path", ["/"])[0]
    return f"{parsed.scheme}://{parsed.netloc}", match.group(1), sub


def _propfind(session, url: str, auth) -> list[tuple[str, int, bool]]:
    response = session.request("PROPFIND", url, headers={"Depth": "1"}, auth=auth, timeout=120)
    if response.status_code in (401, 403):
        raise PermissionError("access denied: wrong password, or the share does not allow WebDAV access")
    if response.status_code != 207:
        raise ConnectionError(f"PROPFIND {url} -> HTTP {response.status_code}")
    entries = []
    for item in ET.fromstring(response.content).iter(f"{DAV}response"):
        href = item.findtext(f"{DAV}href") or ""
        is_dir = item.find(f".//{DAV}resourcetype/{DAV}collection") is not None
        size = int(item.findtext(f".//{DAV}getcontentlength") or 0)
        entries.append((href, size, is_dir))
    return entries


def list_share(url: str, password: str | None = None):
    """Return (session, auth, [(file_url, relative_path, size)])."""
    import requests

    base, token, sub = _parse_share(url)
    session = requests.Session()
    candidates = [(f"{base}/public.php/webdav", (token, password or "")),          # ownCloud / Sciebo
                  (f"{base}/public.php/dav/files/{token}", ("anonymous", password or "") if password else None)]
    last_error = None
    for root, auth in candidates:
        try:
            files, folders = [], [sub.strip("/")]
            root_path = urlparse(root).path.rstrip("/")
            while folders:
                folder = folders.pop()
                folder_url = f"{root}/{quote(folder)}/" if folder else f"{root}/"
                for href, size, is_dir in _propfind(session, folder_url, auth):
                    rel = unquote(href)
                    rel = rel[rel.find(root_path) + len(root_path):] if root_path in rel else rel
                    rel = rel.strip("/")
                    if rel == folder.strip("/"):
                        continue  # the folder itself
                    if is_dir:
                        folders.append(rel)
                    else:
                        files.append((f"{base}{href}" if href.startswith("/") else href, rel, size))
            return session, auth, files
        except (ConnectionError, PermissionError, ET.ParseError) as error:
            last_error = error
    raise ConnectionError(f"Could not read the share ({last_error}). Download the files in the browser instead and "
                          "put them into data/videos/.")


def download_share(url: str, out: str | Path, password: str | None = None, pattern: str = "*",
                   dry_run: bool = False, log: Callable[[str], None] = print) -> list[Path]:
    session, auth, files = list_share(url, password)
    videos = [(u, rel, size) for u, rel, size in files
              if Path(rel).suffix.lower() in VIDEO_EXTENSIONS and fnmatch.fnmatch(Path(rel).name, pattern)]
    total_gb = sum(size for _, _, size in videos) / 1e9
    log(f"{len(videos)} video(s) match '{pattern}' ({total_gb:.1f} GB)")
    out = Path(out)
    saved = []
    for file_url, rel, size in videos:
        target = out / Path(rel).name
        if dry_run:
            log(f"  {rel}  {size / 1e6:.0f} MB")
            continue
        if target.is_file() and (size == 0 or target.stat().st_size == size):
            log(f"  {target.name}: already complete")
            saved.append(target)
            continue
        part = target.with_name(target.name + ".part")
        out.mkdir(parents=True, exist_ok=True)
        start = part.stat().st_size if part.is_file() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        with session.get(file_url, auth=auth, headers=headers, stream=True, timeout=120) as response:
            if response.status_code not in (200, 206):
                log(f"  {rel}: HTTP {response.status_code}, skipped")
                continue
            mode = "ab" if start and response.status_code == 206 else "wb"
            done = start if mode == "ab" else 0
            with part.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
                    done += len(chunk)
            log(f"  {target.name}: {done / 1e6:.0f} MB")
        if size and part.stat().st_size != size:
            log(f"  {target.name}: incomplete ({part.stat().st_size} of {size} bytes) — run again to resume")
            continue
        part.replace(target)
        saved.append(target)
    return saved
