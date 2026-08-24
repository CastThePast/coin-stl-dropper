from __future__ import annotations

import csv
import math
import os
import struct
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import fitz  # PyMuPDF, PDF support
except Exception:  # pragma: no cover
    fitz = None

try:
    from skimage.morphology import dilation, disk, skeletonize
except Exception:  # pragma: no cover
    dilation = disk = skeletonize = None


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pdf"}


@dataclass
class CoinSettings:
    disc_diameter_mm: float = 48.0
    disc_thickness_mm: float = 3.0
    inner_design_diameter_mm: float = 35.0
    min_line_width_mm: float = 1.2
    centre_engrave_depth_mm: float = 0.8
    outer_emboss_height_mm: float = 0.8
    edge_bevel_mm: float = 0.4
    guide_erase_halfwidth_mm: float = 0.60
    outer_feature_margin_mm: float = 0.8
    base_chamfer_mm: float = 0.45
    output_pixels: int = 1536
    mesh_radial_rings: int = 128
    mesh_angular_segments: int = 512
    mirror_for_stamp: bool = True
    remove_noise_below_mm2: float = 0.08
    make_diagnostic_png: bool = True

    def validate(self) -> None:
        if self.disc_diameter_mm <= 0:
            raise ValueError("Disc diameter must be positive.")
        if self.disc_thickness_mm <= 0:
            raise ValueError("Disc thickness must be positive.")
        if not 0 < self.inner_design_diameter_mm < self.disc_diameter_mm:
            raise ValueError("Inner design diameter must be between 0 and disc diameter.")
        if self.centre_engrave_depth_mm <= 0 or self.centre_engrave_depth_mm >= self.disc_thickness_mm:
            raise ValueError("Centre engraving depth must be positive and less than disc thickness.")
        if self.outer_emboss_height_mm < 0:
            raise ValueError("Outer emboss height cannot be negative.")
        if self.output_pixels < 512:
            raise ValueError("Output pixel resolution should be at least 512.")
        if self.mesh_radial_rings < 24 or self.mesh_angular_segments < 96:
            raise ValueError("Mesh resolution is too low.")


@dataclass
class GuideDetection:
    cx: float
    cy: float
    inner_radius: float
    outer_radius: float
    inner_ellipse: Tuple[Tuple[float, float], Tuple[float, float], float]
    outer_ellipse: Tuple[Tuple[float, float], Tuple[float, float], float]
    method: str
    score: float


@dataclass
class ProcessResult:
    input_file: str
    success: bool
    stl_file: str = ""
    diagnostic_file: str = ""
    error: str = ""
    detected_inner_outer_ratio: float = 0.0


# ------------------------------ image loading ------------------------------

def _read_raster(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path.name}")
    return img


def load_input_image(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".pdf":
        if fitz is None:
            raise RuntimeError("PDF support needs PyMuPDF. Install requirements.txt.")
        doc = fitz.open(str(path))
        if len(doc) < 1:
            raise ValueError("PDF contains no pages.")
        page = doc[0]
        # About 220 dpi: enough for circle / marker detection without huge memory use.
        pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr.copy()
    return _read_raster(path)


# -------------------------- page perspective correction -------------------

def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],   # top-left
        pts[np.argmin(d)],   # top-right
        pts[np.argmax(s)],   # bottom-right
        pts[np.argmax(d)],   # bottom-left
    ], dtype=np.float32)


def rectify_a4_page(img: np.ndarray) -> np.ndarray:
    """Try to find the paper boundary and perspective-warp it. Falls back safely."""
    h0, w0 = img.shape[:2]
    scale = min(1.0, 1600.0 / max(h0, w0))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = small.shape[0] * small.shape[1]
    quad = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        area = cv2.contourArea(c)
        if area < 0.28 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2) / scale
            break
    if quad is None:
        return img

    q = _order_quad(quad)
    tl, tr, br, bl = q
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    width = int(max(top, bottom))
    height = int(max(left, right))
    if width < 300 or height < 300:
        return img

    # Preserve orientation first; rotate to portrait afterwards.
    target_w = min(1700, width)
    target_h = int(target_w * height / max(width, 1))
    if target_h > 2400:
        target_h = 2400
        target_w = int(target_h * width / max(height, 1))
    dst = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    warped = cv2.warpPerspective(img, M, (target_w, target_h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    if warped.shape[1] > warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


# ------------------------------ guide detection ---------------------------

def _ellipse_equiv_radius(ellipse) -> float:
    (_, _), (w, h), _ = ellipse
    return 0.5 * math.sqrt(max(w, 1e-6) * max(h, 1e-6))


def _ellipse_axis_ratio(ellipse) -> float:
    (_, _), (w, h), _ = ellipse
    return min(w, h) / max(w, h)


def detect_guide_circles(img: np.ndarray, expected_ratio: float) -> GuideDetection:
    """Detect the two large concentric printed guide circles as ellipse pairs."""
    h0, w0 = img.shape[:2]
    scale = min(1.0, 1400.0 / max(h0, w0))
    work = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    # Flatten illumination so marker/circle ink stays dark even in phone photos.
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(15, min(work.shape[:2]) / 45))
    norm = cv2.divide(gray, bg, scale=245)
    norm = cv2.GaussianBlur(norm, (3, 3), 0)
    _, bw = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    short = min(work.shape[:2])
    candidates = []
    for c in contours:
        if len(c) < 40:
            continue
        if len(c) < 5:
            continue
        try:
            e = cv2.fitEllipse(c)
        except cv2.error:
            continue
        (cx, cy), (ew, eh), angle = e
        r = _ellipse_equiv_radius(e)
        ar = _ellipse_axis_ratio(e)
        if not (0.07 * short <= r <= 0.46 * short):
            continue
        if ar < 0.68:
            continue
        perim = cv2.arcLength(c, True)
        if perim < 0.55 * 2 * math.pi * r:
            continue
        # Do not reject thin outline circles by contour area: a photographed pencil/
        # printed ring can have a very small signed contour area even when it spans
        # almost a full ellipse.  Size, axis ratio, perimeter and concentric pairing
        # below are safer discriminators than filled area here.
        candidates.append(e)

    # Deduplicate nearly identical contour edges from the same printed circle.
    candidates.sort(key=_ellipse_equiv_radius)
    dedup = []
    for e in candidates:
        r = _ellipse_equiv_radius(e)
        cx, cy = e[0]
        same = False
        for d in dedup:
            rd = _ellipse_equiv_radius(d)
            dcx, dcy = d[0]
            if abs(r - rd) < max(3.0, 0.018 * r) and math.hypot(cx - dcx, cy - dcy) < max(4.0, 0.018 * r):
                same = True
                break
        if not same:
            dedup.append(e)

    page_cx = work.shape[1] / 2
    page_cy = work.shape[0] / 2
    best = None
    best_score = float("inf")
    for outer in dedup:
        ro = _ellipse_equiv_radius(outer)
        ocx, ocy = outer[0]
        for inner in dedup:
            ri = _ellipse_equiv_radius(inner)
            if ri >= ro:
                continue
            ratio = ri / ro
            if not (0.54 <= ratio <= 0.90):
                continue
            icx, icy = inner[0]
            cd = math.hypot(icx - ocx, icy - ocy) / ro
            if cd > 0.12:
                continue
            ar_diff = abs(_ellipse_axis_ratio(inner) - _ellipse_axis_ratio(outer))
            center_page = math.hypot(ocx - page_cx, ocy - page_cy) / max(short, 1)
            ratio_pen = abs(ratio - expected_ratio)
            # Strongly prefer the expected concentric pair, then a large pair near page centre.
            score = 10.0 * ratio_pen + 5.0 * cd + 1.5 * ar_diff + 0.35 * center_page - 0.25 * (ro / short)
            if score < best_score:
                best_score = score
                best = (inner, outer, ratio)

    if best is None:
        # Hough fallback for clean scans where contour pairing can be disrupted by overlapping drawings.
        hgray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            hgray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=120, param2=38,
            minRadius=int(0.07 * short), maxRadius=int(0.46 * short)
        )
        if circles is not None:
            cs = circles[0]
            for o in cs:
                for i in cs:
                    if i[2] >= o[2]:
                        continue
                    ratio = i[2] / o[2]
                    if not (0.54 <= ratio <= 0.90):
                        continue
                    cd = math.hypot(float(i[0] - o[0]), float(i[1] - o[1])) / float(o[2])
                    if cd > 0.08:
                        continue
                    score = 10.0 * abs(ratio - expected_ratio) + 5.0 * cd - 0.25 * (o[2] / short)
                    if score < best_score:
                        inner_e = ((float(i[0]), float(i[1])), (2 * float(i[2]), 2 * float(i[2])), 0.0)
                        outer_e = ((float(o[0]), float(o[1])), (2 * float(o[2]), 2 * float(o[2])), 0.0)
                        best = (inner_e, outer_e, ratio)
                        best_score = score
        if best is None:
            raise ValueError(
                "Could not reliably find the two concentric guide circles. "
                "Use a flat/straight-on photo or scan, with both circles fully visible."
            )
        method = "Hough fallback"
    else:
        method = "ellipse contours"

    inner, outer, ratio = best

    # Scale detection coordinates back to the current full-resolution image.
    inv = 1.0 / scale
    def scale_ellipse(e):
        (cx, cy), (ew, eh), angle = e
        return ((cx * inv, cy * inv), (ew * inv, eh * inv), angle)

    inner = scale_ellipse(inner)
    outer = scale_ellipse(outer)
    cx = (inner[0][0] + outer[0][0]) / 2.0
    cy = (inner[0][1] + outer[0][1]) / 2.0
    ri = _ellipse_equiv_radius(inner)
    ro = _ellipse_equiv_radius(outer)
    return GuideDetection(
        cx=cx, cy=cy,
        inner_radius=ri,
        outer_radius=ro,
        inner_ellipse=inner,
        outer_ellipse=outer,
        method=method,
        score=float(best_score),
    )


# ---------------------------- mask construction ----------------------------

def _ellipse_affine_to_circle(ellipse, N: int, target_radius_px: float) -> np.ndarray:
    """Affine transform mapping a fitted ellipse to a centred circle without rotating the drawing.

    Scale along the ellipse's own principal axes, but keep those axes pointing in
    the same directions on the page.  The previous implementation aligned an
    ellipse axis to screen-X, which could rotate a child's drawing by an arbitrary
    amount depending on the fitted ellipse angle.
    """
    (cx, cy), (ew, eh), angle_deg = ellipse
    th = math.radians(angle_deg)
    u = np.array([math.cos(th), math.sin(th)], dtype=np.float64)
    v = np.array([-math.sin(th), math.cos(th)], dtype=np.float64)
    a = max(0.5 * ew, 1e-6)
    b = max(0.5 * eh, 1e-6)
    # Symmetric anisotropic scale: no reflection and no arbitrary global rotation.
    A = (target_radius_px / a) * np.outer(u, u) + (target_radius_px / b) * np.outer(v, v)
    centre = np.array([cx, cy], dtype=np.float64)
    target = np.array([N / 2.0, N / 2.0], dtype=np.float64)
    t = target - A @ centre
    return np.hstack([A, t[:, None]]).astype(np.float32)


def _transform_point(M: np.ndarray, p: Tuple[float, float]) -> np.ndarray:
    x, y = p
    return M[:, :2] @ np.array([x, y], dtype=np.float64) + M[:, 2]


def _transformed_ellipse_radius(M: np.ndarray, ellipse, centre_out: np.ndarray) -> float:
    (cx, cy), (ew, eh), angle_deg = ellipse
    th = math.radians(angle_deg)
    p1 = (cx + 0.5 * ew * math.cos(th), cy + 0.5 * ew * math.sin(th))
    p2 = (cx - 0.5 * eh * math.sin(th), cy + 0.5 * eh * math.cos(th))
    q1 = _transform_point(M, p1)
    q2 = _transform_point(M, p2)
    return float((np.linalg.norm(q1 - centre_out) + np.linalg.norm(q2 - centre_out)) / 2.0)


def _remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    if min_area_px <= 1:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(1, n):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area_px:
            out[labels == idx] = 1
    return out.astype(bool)


def _radial_remap(mask: np.ndarray, src_inner_ratio: float, dst_inner_ratio: float, Rpx: float) -> np.ndarray:
    """Map source inner/outer guide radii exactly onto desired output radii, preserving angle."""
    N = mask.shape[0]
    C = (N - 1) / 2.0
    yy, xx = np.indices((N, N), dtype=np.float32)
    dx = xx - C
    dy = yy - C
    ro = np.sqrt(dx * dx + dy * dy)
    rn = ro / max(Rpx, 1e-6)

    rsn = np.empty_like(rn)
    inside = rn <= dst_inner_ratio
    rsn[inside] = rn[inside] * (src_inner_ratio / max(dst_inner_ratio, 1e-6))
    outer = ~inside
    rsn[outer] = src_inner_ratio + (rn[outer] - dst_inner_ratio) * (
        (1.0 - src_inner_ratio) / max(1.0 - dst_inner_ratio, 1e-6)
    )
    scale = np.ones_like(rn)
    nz = rn > 1e-6
    scale[nz] = rsn[nz] / rn[nz]
    mapx = C + dx * scale
    mapy = C + dy * scale
    remapped = cv2.remap(mask.astype(np.float32), mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    remapped[rn > 1.0] = 0
    return remapped >= 0.5


def _ensure_minimum_width(mask: np.ndarray, width_px: int) -> np.ndarray:
    if width_px <= 2 or skeletonize is None or dilation is None or disk is None:
        return mask
    try:
        skel = skeletonize(mask)
        rad = max(1, int(math.ceil(width_px / 2.0)))
        thick = dilation(skel, footprint=disk(rad))
        return np.logical_or(mask, thick)
    except Exception:
        # Conservative fallback: a small dilation rather than failing the batch.
        k = max(1, int(round(width_px / 4)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _transformed_ellipse_guide_mask(M: np.ndarray, ellipse, N: int, thickness_px: int) -> np.ndarray:
    """Rasterise a detected guide ellipse after the image affine transform.

    This is more reliable than erasing only an ideal concentric radial band when a
    phone photo or a hand-drawn/printed guide is slightly off-centre.
    """
    (cx, cy), (ew, eh), angle_deg = ellipse
    th = math.radians(angle_deg)
    ct, st = math.cos(th), math.sin(th)
    t = np.linspace(0.0, 2.0 * np.pi, 1440, endpoint=False, dtype=np.float64)
    ux = 0.5 * ew * np.cos(t)
    uy = 0.5 * eh * np.sin(t)
    x = cx + ct * ux - st * uy
    y = cy + st * ux + ct * uy
    pts = np.column_stack([x, y, np.ones_like(x)])
    q = pts @ M.T
    q = np.rint(q).astype(np.int32).reshape((-1, 1, 2))
    out = np.zeros((N, N), dtype=np.uint8)
    cv2.polylines(out, [q], True, 255, thickness=max(1, int(thickness_px)), lineType=cv2.LINE_AA)
    return out.astype(bool)


def build_design_masks(img: np.ndarray, detection: GuideDetection, settings: CoinSettings):
    settings.validate()
    N = settings.output_pixels
    C = (N - 1) / 2.0
    Rpx = 0.465 * N  # leaves a white margin around the whole disc in diagnostics
    px_per_mm = (2.0 * Rpx) / settings.disc_diameter_mm

    # Warp the photographed/scanned outer guide ellipse into a true circle.
    M = _ellipse_affine_to_circle(detection.outer_ellipse, N, Rpx)
    warped = cv2.warpAffine(img, M, (N, N), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Correct uneven phone-photo illumination before thresholding ink.
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(14, N / 38))
    norm = cv2.divide(gray, bg, scale=245)
    norm = cv2.GaussianBlur(norm, (3, 3), 0)
    _, ink = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = ink.astype(bool)

    # The supplied classroom template uses blue guide circles and children are asked
    # to draw in black.  Explicitly ignore blue/cyan guide ink (including anti-aliased
    # edges).  Geometry-based guide removal below remains as a fallback for monochrome
    # scans and older templates.
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    blue_guide = (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 145) & (hsv[:, :, 1] >= 45)
    if blue_guide.any():
        blue_guide = cv2.dilate(blue_guide.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
        ink[blue_guide] = False

    centre_out = np.array([N / 2.0, N / 2.0], dtype=np.float64)
    src_inner_px = _transformed_ellipse_radius(M, detection.inner_ellipse, centre_out)
    src_inner_ratio = float(np.clip(src_inner_px / Rpx, 0.50, 0.92))
    dst_inner_ratio = settings.inner_design_diameter_mm / settings.disc_diameter_mm

    yy, xx = np.indices((N, N), dtype=np.float32)
    rr = np.sqrt((xx - C) ** 2 + (yy - C) ** 2)
    guide_half = settings.guide_erase_halfwidth_mm * px_per_mm
    src_inner = src_inner_ratio * Rpx

    # The printed guides are ink too.  Remove both (a) generous ideal radial bands and
    # (b) the actually detected inner/outer guide curves after transformation.  The
    # second part matters for slightly skewed phone photos and imperfect circles.
    guide_band = (np.abs(rr - src_inner) <= guide_half) | (np.abs(rr - Rpx) <= guide_half)
    curve_thickness = max(3, int(round(2.0 * guide_half)))
    detected_guides = (
        _transformed_ellipse_guide_mask(M, detection.inner_ellipse, N, curve_thickness)
        | _transformed_ellipse_guide_mask(M, detection.outer_ellipse, N, curve_thickness)
    )
    ink[guide_band | detected_guides] = False

    centre_mask = ink & (rr < (src_inner - guide_half))
    outer_mask = ink & (rr > (src_inner + guide_half)) & (rr < (Rpx - settings.outer_feature_margin_mm * px_per_mm))

    # Radially map the detected template boundary to the requested physical inner diameter.
    centre_mask = _radial_remap(centre_mask, src_inner_ratio, dst_inner_ratio, Rpx)
    outer_mask = _radial_remap(outer_mask, src_inner_ratio, dst_inner_ratio, Rpx)

    # Force the two semantic zones after remapping so no accidental overlap can occur.
    rr_out = rr
    dst_inner_px = dst_inner_ratio * Rpx
    boundary_gap = max(1.0, settings.guide_erase_halfwidth_mm * px_per_mm * 0.55)
    centre_mask &= rr_out < (dst_inner_px - boundary_gap)
    outer_mask &= rr_out > (dst_inner_px + boundary_gap)
    outer_mask &= rr_out < (Rpx - settings.outer_feature_margin_mm * px_per_mm)

    # Remove specks and smooth tiny jagged scan artefacts.
    min_area_px = int(settings.remove_noise_below_mm2 * px_per_mm * px_per_mm)
    centre_mask = _remove_small_components(centre_mask, min_area_px)
    outer_mask = _remove_small_components(outer_mask, min_area_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    centre_mask = cv2.morphologyEx(centre_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
    outer_mask = cv2.morphologyEx(outer_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)

    # Guarantee that thin marker strokes survive at the requested printable minimum width.
    min_width_px = max(2, int(round(settings.min_line_width_mm * px_per_mm)))
    centre_mask = _ensure_minimum_width(centre_mask, min_width_px)
    outer_mask = _ensure_minimum_width(outer_mask, min_width_px)

    # Re-clip after thickening so the outer rim remains structurally clean.
    centre_mask &= rr_out < (dst_inner_px - 0.20 * px_per_mm)
    outer_mask &= rr_out > (dst_inner_px + 0.20 * px_per_mm)
    outer_mask &= rr_out < (Rpx - settings.outer_feature_margin_mm * px_per_mm)

    return warped, centre_mask, outer_mask, Rpx, px_per_mm, src_inner_ratio


# ------------------------------ height map ---------------------------------

def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_heightmap(centre_mask: np.ndarray, outer_mask: np.ndarray, settings: CoinSettings, px_per_mm: float) -> np.ndarray:
    bevel_px = max(1.0, settings.edge_bevel_mm * px_per_mm)
    dc = cv2.distanceTransform(centre_mask.astype(np.uint8), cv2.DIST_L2, 5)
    do = cv2.distanceTransform(outer_mask.astype(np.uint8), cv2.DIST_L2, 5)
    ac = _smoothstep01(dc / bevel_px)
    ao = _smoothstep01(do / bevel_px)
    return (
        settings.disc_thickness_mm
        - settings.centre_engrave_depth_mm * ac
        + settings.outer_emboss_height_mm * ao
    ).astype(np.float32)


# ------------------------------- STL mesh ----------------------------------

def _sample_height_polar(heightmap: np.ndarray, Rpx: float, Rmm: float, nr: int, nt: int):
    N = heightmap.shape[0]
    C = (N - 1) / 2.0
    radii = np.linspace(Rmm / nr, Rmm, nr, dtype=np.float32)
    theta = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False, dtype=np.float32)
    rr, tt = np.meshgrid(radii, theta, indexing="ij")
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    mapx = C + (x / Rmm) * Rpx
    # Image rows increase downward, while Cartesian Y increases upward.
    # Subtract here so an unmirrored STL matches the source when viewed from +Z.
    mapy = C - (y / Rmm) * Rpx
    z = cv2.remap(heightmap, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return x, y, z


def build_watertight_mesh(heightmap: np.ndarray, Rpx: float, settings: CoinSettings):
    R = settings.disc_diameter_mm / 2.0
    nr = settings.mesh_radial_rings
    nt = settings.mesh_angular_segments
    x, y, z = _sample_height_polar(heightmap, Rpx, R, nr, nt)

    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []

    # Top centre and top rings.
    top_center = len(vertices)
    vertices.append((0.0, 0.0, float(heightmap[heightmap.shape[0] // 2, heightmap.shape[1] // 2])))
    top_start = len(vertices)
    for j in range(nr):
        for i in range(nt):
            vertices.append((float(x[j, i]), float(y[j, i]), float(z[j, i])))

    def top_idx(j, i):
        return top_start + j * nt + (i % nt)

    # Centre fan, CCW when viewed from above.
    for i in range(nt):
        faces.append((top_center, top_idx(0, i), top_idx(0, i + 1)))
    # Annular top quads.
    for j in range(nr - 1):
        for i in range(nt):
            a = top_idx(j, i)
            b = top_idx(j + 1, i)
            c = top_idx(j + 1, i + 1)
            d = top_idx(j, i + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))

    chamfer = max(0.0, min(settings.base_chamfer_mm, R * 0.25, settings.disc_thickness_mm * 0.45))
    Rb = R - chamfer if chamfer > 1e-6 else R

    # Flat bottom uses only a centre fan (no need for thousands of coplanar rings).
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, 0.0))
    bottom_start = len(vertices)
    theta = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
    for t in theta:
        vertices.append((Rb * math.cos(t), Rb * math.sin(t), 0.0))

    def bot_idx(i):
        return bottom_start + (i % nt)

    # Clockwise when viewed from above => outward normal is -Z.
    for i in range(nt):
        faces.append((bottom_center, bot_idx(i + 1), bot_idx(i)))

    top_outer = [top_idx(nr - 1, i) for i in range(nt)]

    if chamfer > 1e-6:
        mid_start = len(vertices)
        for t in theta:
            vertices.append((R * math.cos(t), R * math.sin(t), chamfer))

        def mid_idx(i):
            return mid_start + (i % nt)

        # Vertical wall from chamfer height to top perimeter.
        for i in range(nt):
            a = top_outer[i]
            b = top_outer[(i + 1) % nt]
            c = mid_idx(i + 1)
            d = mid_idx(i)
            faces.append((a, c, b))
            faces.append((a, d, c))
        # Sloped base chamfer.
        for i in range(nt):
            a = mid_idx(i)
            b = mid_idx(i + 1)
            c = bot_idx(i + 1)
            d = bot_idx(i)
            faces.append((a, c, b))
            faces.append((a, d, c))
    else:
        # Straight side wall.
        for i in range(nt):
            a = top_outer[i]
            b = top_outer[(i + 1) % nt]
            c = bot_idx(i + 1)
            d = bot_idx(i)
            faces.append((a, c, b))
            faces.append((a, d, c))

    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray, header_text: str = "Coin STL Dropper") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = header_text.encode("ascii", errors="replace")[:80].ljust(80, b" ")
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(faces)))
        batch = bytearray()
        for tri in faces:
            v1, v2, v3 = vertices[tri]
            n = np.cross(v2 - v1, v3 - v1)
            norm = float(np.linalg.norm(n))
            if norm > 1e-12:
                n = n / norm
            else:
                n = np.zeros(3, dtype=np.float32)
            batch.extend(struct.pack("<12fH",
                                     float(n[0]), float(n[1]), float(n[2]),
                                     float(v1[0]), float(v1[1]), float(v1[2]),
                                     float(v2[0]), float(v2[1]), float(v2[2]),
                                     float(v3[0]), float(v3[1]), float(v3[2]),
                                     0))
        f.write(batch)


# ------------------------------ diagnostics --------------------------------

def save_diagnostic(path: Path, centre_mask: np.ndarray, outer_mask: np.ndarray, settings: CoinSettings, Rpx: float) -> None:
    N = centre_mask.shape[0]
    canvas = np.full((N, N, 3), 255, dtype=np.uint8)
    # Blue-ish centre, green-ish outer. This is only a visual verification image.
    canvas[centre_mask] = (210, 165, 80)  # BGR
    canvas[outer_mask] = (155, 220, 170)
    C = int(round((N - 1) / 2.0))
    outer_r = int(round(Rpx))
    inner_r = int(round(Rpx * settings.inner_design_diameter_mm / settings.disc_diameter_mm))
    cv2.circle(canvas, (C, C), outer_r, (40, 40, 40), 2)
    cv2.circle(canvas, (C, C), inner_r, (80, 80, 80), 2)
    ok, encoded = cv2.imencode(".png", canvas)
    if ok:
        encoded.tofile(str(path))


# ------------------------------ public API ---------------------------------

def _unique_output_stem(output_dir: Path, stem: str) -> str:
    candidate = stem
    n = 2
    while (output_dir / f"{candidate}.stl").exists() or (output_dir / f"{candidate}_CHECK.png").exists():
        candidate = f"{stem}__{n}"
        n += 1
    return candidate


def process_one(path: Path, output_dir: Path, settings: CoinSettings, progress: Optional[Callable[[str], None]] = None) -> ProcessResult:
    progress = progress or (lambda _msg: None)
    result = ProcessResult(input_file=str(path), success=False)
    try:
        progress(f"Reading {path.name}")
        img = load_input_image(path)
        img = rectify_a4_page(img)
        expected_ratio = settings.inner_design_diameter_mm / settings.disc_diameter_mm
        progress(f"Finding guide circles in {path.name}")
        det = detect_guide_circles(img, expected_ratio)
        result.detected_inner_outer_ratio = det.inner_radius / det.outer_radius

        progress(f"Extracting drawing from {path.name}")
        _, centre_mask, outer_mask, Rpx, px_per_mm, _ = build_design_masks(img, det, settings)
        if not centre_mask.any() and not outer_mask.any():
            raise ValueError("No black drawing was found inside either design zone after removing the guide circles.")

        # A physical stamp must be mirrored so the clay impression reads the same way
        # as the child's drawing.  Make this explicit instead of relying on image/mesh
        # coordinate conventions.  Diagnostics stay unmirrored for easy comparison.
        mesh_centre = np.fliplr(centre_mask) if settings.mirror_for_stamp else centre_mask
        mesh_outer = np.fliplr(outer_mask) if settings.mirror_for_stamp else outer_mask
        heightmap = build_heightmap(mesh_centre, mesh_outer, settings, px_per_mm)
        progress(f"Building watertight mesh for {path.name}")
        vertices, faces = build_watertight_mesh(heightmap, Rpx, settings)

        stem = _unique_output_stem(output_dir, path.stem)
        stl_path = output_dir / f"{stem}.stl"
        write_binary_stl(stl_path, vertices, faces, header_text=f"Coin STL Dropper: {stem}")
        result.stl_file = str(stl_path)

        if settings.make_diagnostic_png:
            diag_path = output_dir / f"{stem}_CHECK.png"
            save_diagnostic(diag_path, centre_mask, outer_mask, settings, Rpx)
            result.diagnostic_file = str(diag_path)

        result.success = True
        progress(f"✓ {path.name} → {stl_path.name}")
    except Exception as exc:
        result.error = str(exc)
        progress(f"✗ {path.name}: {exc}")
    return result


def expand_inputs(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in SUPPORTED_EXTENSIONS))
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            out.append(p)
    # preserve order but de-duplicate exact paths
    seen = set()
    unique = []
    for p in out:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def write_batch_report(output_dir: Path, results: Sequence[ProcessResult], settings: CoinSettings) -> None:
    report = output_dir / "batch_report.csv"
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["input_file", "success", "stl_file", "diagnostic_file", "detected_inner_outer_ratio", "error"])
        for r in results:
            w.writerow([r.input_file, r.success, r.stl_file, r.diagnostic_file, f"{r.detected_inner_outer_ratio:.4f}", r.error])

    settings_path = output_dir / "settings_used.txt"
    with open(settings_path, "w", encoding="utf-8") as f:
        for k, v in asdict(settings).items():
            f.write(f"{k}={v}\n")


def process_batch(paths: Sequence[Path], output_dir: Path, settings: Optional[CoinSettings] = None,
                  progress: Optional[Callable[[str], None]] = None) -> List[ProcessResult]:
    settings = settings or CoinSettings()
    settings.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = expand_inputs(paths)
    if not inputs:
        raise ValueError("No supported image/PDF files were selected.")
    results = []
    for i, path in enumerate(inputs, start=1):
        if progress:
            progress(f"[{i}/{len(inputs)}] {path.name}")
        results.append(process_one(path, output_dir, settings, progress=progress))
    write_batch_report(output_dir, results, settings)
    return results
