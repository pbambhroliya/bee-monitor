"""Motion cues from frame differencing.

Motion-Informed Enhancement (MIE) follows Bjerge et al., "Motion Informed Object Detection
of Small Insects in Time-lapse Camera Recordings" (Sensors 2023):

    I_gb  = GaussianBlur(gray(frame))
    FD_k  = |I_gb(k) - I_gb(k-n)| + |I_gb(k+n) - I_gb(k)|
    MIE   : red   = FD_k
            green = green
            blue  = 0.5 * blue + 0.5 * red

The result is an ordinary 3-channel image, so any YOLO model can be trained on it. In the
paper the frames were 30 s apart (time-lapse); for 30 fps video we use k-n, k, k+n with a
configurable n (``motion.offset_frames``).
"""

from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np

from beemon.video import FramePack


class MotionComputer:
    """Computes frame-difference maps, caching the blurred grey frames by index."""

    def __init__(self, blur: int = 5, offset: int = 4, cache_size: int = 24, local_kernel: int = 0):
        self.blur = int(blur) | 1  # kernel must be odd
        self.offset = int(offset)
        self.local_kernel = int(local_kernel)
        self._cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def _gray(self, key: tuple[int, int], frame: np.ndarray) -> np.ndarray:
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.blur > 1:
            gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)
        self._cache[key] = gray
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return gray

    def difference(self, pack: FramePack) -> np.ndarray:
        """Frame-difference map FD (uint8, H x W). Zeros when no neighbours are available."""
        if pack.prev is None or pack.next is None:
            return np.zeros(pack.frame.shape[:2], dtype=np.uint8)
        k = pack.index
        current = self._gray((k, 0), pack.frame)
        previous = self._gray((k - self.offset, 0), pack.prev)
        following = self._gray((k + self.offset, 0), pack.next)
        diff = cv2.add(cv2.absdiff(current, previous), cv2.absdiff(following, current))  # saturating add
        return local_motion(diff, self.local_kernel)


def local_motion(fd: np.ndarray, kernel: int) -> np.ndarray:
    """Optional: subtract the average motion of the surroundings (kernel x kernel px).

    Wind moves whole flower clusters, which gives large areas of motion. A bee moves
    differently from its surroundings. Subtracting the local mean keeps small, independent
    motion and suppresses broad swaying. ``kernel`` ~ 3-4x the bee size (e.g. 101 px for 1080p).
    0 = off (plain frame difference as in Bjerge et al.).
    """
    if not kernel or kernel <= 1:
        return fd
    return cv2.subtract(fd, cv2.blur(fd, (int(kernel), int(kernel))))


def mie_image(frame: np.ndarray, fd: np.ndarray) -> np.ndarray:
    """Build the motion-informed image (BGR order, as OpenCV uses)."""
    out = np.empty_like(frame)
    blue = frame[..., 0].astype(np.uint16)
    red = frame[..., 2].astype(np.uint16)
    out[..., 0] = ((blue + red) // 2).astype(np.uint8)
    out[..., 1] = frame[..., 1]
    out[..., 2] = fd
    return out


def frame_difference(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray, blur: int = 5,
                     local_kernel: int = 0) -> np.ndarray:
    """Stand-alone FD for three frames (used by dataset tools)."""
    blur = int(blur) | 1

    def g(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (blur, blur), 0) if blur > 1 else gray

    a, b, c = g(prev), g(cur), g(nxt)
    return local_motion(cv2.add(cv2.absdiff(b, a), cv2.absdiff(c, b)), local_kernel)


def box_motion(fd: np.ndarray, xyxy: np.ndarray, ring: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Mean motion inside each box and its ratio to the motion in a ring around the box.

    A flying or walking bee moves more than its surroundings (ratio > 1). A flower bud that a
    detector mistakes for a bee moves like its neighbours (ratio ~ 1) or not at all.
    Returns ``(inside_mean, ratio)``; both are 0-length arrays when there are no boxes.
    """
    n = len(xyxy)
    inside = np.zeros(n, dtype=np.float32)
    ratio = np.zeros(n, dtype=np.float32)
    if n == 0:
        return inside, ratio
    height, width = fd.shape[:2]
    integral = cv2.integral(fd, sdepth=cv2.CV_64F)

    def box_sum(x0, y0, x1, y1):
        return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]

    for i, (x1, y1, x2, y2) in enumerate(np.asarray(xyxy, dtype=np.float64)):
        ix0, iy0 = int(np.clip(np.floor(x1), 0, width - 1)), int(np.clip(np.floor(y1), 0, height - 1))
        ix1, iy1 = int(np.clip(np.ceil(x2), ix0 + 1, width)), int(np.clip(np.ceil(y2), iy0 + 1, height))
        pad_x = max(2, int(round((ix1 - ix0) * ring)))
        pad_y = max(2, int(round((iy1 - iy0) * ring)))
        ox0, oy0 = max(0, ix0 - pad_x), max(0, iy0 - pad_y)
        ox1, oy1 = min(width, ix1 + pad_x), min(height, iy1 + pad_y)
        inner_area = (ix1 - ix0) * (iy1 - iy0)
        outer_area = (ox1 - ox0) * (oy1 - oy0)
        inner_sum = box_sum(ix0, iy0, ix1, iy1)
        ring_area = max(1, outer_area - inner_area)
        ring_mean = (box_sum(ox0, oy0, ox1, oy1) - inner_sum) / ring_area
        inside[i] = inner_sum / max(1, inner_area)
        ratio[i] = (inside[i] + 1.0) / (ring_mean + 1.0)
    return inside, ratio
