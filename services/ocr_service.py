# services/ocr_service.py

import os
os.environ["PPOCR_ENABLE_PADDLEX"] = "0"
os.environ["PPOCR_USE_PADDLEX"] = "0"

from paddleocr import PaddleOCR

import re
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

# --- add below your existing imports ---
# (you already have cv2, np, load_image_bgr, find_mrz_crop, bottom_crop, etc.)

def _find_mrz_crop_bbox(image: np.ndarray):
    """
    Like find_mrz_crop but also returns the bbox (x0,y0,x1,y1).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect)

    grad = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
    grad = cv2.convertScaleAbs(grad)
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9)), iterations=1)

    H, W = gray.shape
    bottom = bw[H//2:, :]
    cnts, _ = cv2.findContours(bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    best, best_score = None, -1
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        y += H//2
        ratio = w / max(1, h)
        bottomness = (y + h) / H
        score = (w/W) + 0.5*min(ratio/20.0, 1.0) + 0.5*bottomness
        if score > best_score:
            best_score, best = score, (x, y, w, h)

    if not best:
        return None, None
    x, y, w, h = best
    pad_y = int(h*0.25)
    y0, y1 = max(0, y - pad_y), min(H, y + h + pad_y)
    x0, x1 = max(0, int(x - 0.02*W)), min(W, int(x + w + 0.02*W))
    crop = image[y0:y1, x0:x1]
    return crop, (x0, y0, x1, y1)

def save_debug_overlay(image_path: str, out_path: str):
    """
    Draw the MRZ region used for OCR as a green rectangle and save alongside the image.
    """
    img = load_image_bgr(image_path)
    crop, bbox = _find_mrz_crop_bbox(img)
    if bbox is None:
        # fallback: a straight band at the bottom
        H, W = img.shape[:2]
        band_h = max(90, int(H*0.14))
        bbox = (30, H-band_h-30, W-30, H-30)
    x0, y0, x1, y1 = bbox
    dbg = img.copy()
    cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.imwrite(out_path, dbg)



# ---------- Public API ----------

def init_ocr() -> PaddleOCR:
    """Initialize OCR engine (English + angle classifier)."""
    return PaddleOCR(use_angle_cls=True, lang='en')

def load_image_bgr(path: str) -> np.ndarray:
    """Robust image loader to BGR (OpenCV)."""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is not None:
        return im
    # Pillow fallback (TIFF etc.)
    pil = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def extract_mrz_two_lines(image_bgr: np.ndarray, ocr: PaddleOCR) -> Tuple[str, str]:
    """
    Detect MRZ band, OCR it, pick best two consecutive lines.
    Returns (L1, L2) uppercase with spaces removed.
    Raises ValueError if two lines cannot be found.
    """
    crop = find_mrz_crop(image_bgr)
    if crop is None:
        crop = bottom_crop(image_bgr, 0.35)
    cands = ocr_lines_from_image(ocr, crop)
    pair = pick_best_two_mrz_lines(cands)

    if not pair and len(cands) == 1:
        # try full image as fallback once
        cands2 = ocr_lines_from_image(ocr, image_bgr)
        pair = pick_best_two_mrz_lines(cands2)

    if not pair:
        raise ValueError("Could not locate two MRZ lines")

    L1 = _normalize_line(pair[0])
    L2 = _normalize_line(pair[1])
    return L1, L2

# ---------- Internals (kept small & clear) ----------

def bottom_crop(image: np.ndarray, frac: float = 0.35) -> np.ndarray:
    h = image.shape[0]
    y0 = int(h*(1.0 - max(0.05, min(frac, 0.95))))
    return image[y0:, :]

def find_mrz_crop(image: np.ndarray) -> Optional[np.ndarray]:
    """Light-weight MRZ band detector near the bottom."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect)

    grad = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
    grad = cv2.convertScaleAbs(grad)
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9)), iterations=1)

    H, W = gray.shape
    bottom = bw[H//2:, :]
    cnts, _ = cv2.findContours(bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best, best_score = None, -1
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        y += H//2
        ratio = w / max(1, h)
        bottomness = (y + h) / H
        score = (w/W) + 0.5*min(ratio/20.0, 1.0) + 0.5*bottomness
        if score > best_score:
            best_score, best = score, (x, y, w, h)

    if not best:
        return None

    x, y, w, h = best
    pad_y = int(h*0.25)
    y0, y1 = max(0, y - pad_y), min(H, y + h + pad_y)
    x0, x1 = max(0, int(x - 0.02*W)), min(W, int(x + w + 0.02*W))
    return image[y0:y1, x0:x1]

def ocr_lines_from_image(ocr: PaddleOCR, image: np.ndarray) -> List[Tuple[float, str, float]]:
    """Return list of (cy, text, conf) sorted top->bottom."""
    out = []
    result = ocr.ocr(image, cls=True)
    for blocks in result or []:
        for box, (txt, conf) in (blocks or []):
            ys = [p[1] for p in box]
            cy = sum(ys) / 4.0
            out.append((cy, txt.strip(), float(conf or 0.0)))
    out.sort(key=lambda x: x[0])
    return out

def pick_best_two_mrz_lines(cands: List[Tuple[float, str, float]]) -> Optional[Tuple[str, str]]:
    """Pick two consecutive lines likely to be TD3 MRZ."""
    cands = [(cy, s, c) for (cy, s, c) in cands if _is_mrzish(s)]
    if len(cands) < 2:
        return None


    def score_line(s: str, conf: float) -> float:
        s_up = _normalize_line(s)
        length = -abs(len(s_up) - 44)
        lt_ratio = s_up.count("<") / max(1, len(s_up))
        upp_ratio = sum(ch.isalpha() and ch == ch.upper() for ch in s_up) / max(1, len(s_up))
        return length + 5.0*conf + 1.0*lt_ratio + 0.5*upp_ratio

    best = (-1e9, None, None)
    for i in range(len(cands)-1):
        _, s1, c1 = cands[i]
        _, s2, c2 = cands[i+1]
        sc = score_line(s1, c1) + score_line(s2, c2)
        sc += min(sum(ch.isdigit() for ch in s2), 18)*0.15  # L2 digit density
        if sc > best[0]:
            best = (sc, s1, s2)

    return (best[1], best[2]) if best[1] and best[2] else None

def _normalize_line(s: str) -> str:
    return re.sub(r"\s+", "", s.upper())

# add this import with your others
from parsers.td3 import parse_td3

# cache the OCR once
_OCR = None
def get_ocr():
    global _OCR
    if _OCR is None:
        _OCR = init_ocr()
    return _OCR


def find_mrz_crop_bbox(image: np.ndarray):
    """
    Same logic as find_mrz_crop, but also return bbox (x0,y0,x1,y1).
    Returns (crop, (x0,y0,x1,y1)) or (None, None).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect)

    grad = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
    grad = cv2.convertScaleAbs(grad)
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9)), iterations=1)

    H, W = gray.shape
    bottom = bw[H//2:, :]
    cnts, _ = cv2.findContours(bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    best, best_score = None, -1
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        y += H//2
        ratio = w / max(1, h)
        bottomness = (y + h) / H
        score = (w/W) + 0.5*min(ratio/20.0, 1.0) + 0.5*bottomness
        if score > best_score:
            best_score, best = score, (x, y, w, h)

    if not best:
        return None, None
    x, y, w, h = best
    pad_y = int(h*0.25)
    y0, y1 = max(0, y - pad_y), min(H, y + h + pad_y)
    x0, x1 = max(0, int(x - 0.02*W)), min(W, int(x + w + 0.02*W))
    crop = image[y0:y1, x0:x1]
    return crop, (x0, y0, x1, y1)

def _fix_common_mrz_confusions(L1: str, L2: str) -> tuple[str,str]:
    """
    Light corrections for typical OCR mistakes in numeric fields of L2.
    Only touch fields that should be digits/<:
      passport_no(0:9), birth(13:19), expiry(21:27), personal(28:42)
    """
    L1 = L1
    L2 = (L2 + "<"*44)[:44]
    def fix_digits(s: str):
        trans = str.maketrans({
            'O':'0','Q':'0','D':'0',
            'I':'1','L':'1',
            'Z':'2',
            'S':'5',
            'B':'8',
        })
        return ''.join(ch.translate(trans) if ch.isalpha() else ch for ch in s)
    parts = list(L2)
    # passport number
    parts[0:9]   = list(fix_digits(''.join(parts[0:9])))
    # birth date YYMMDD
    parts[13:19] = list(fix_digits(''.join(parts[13:19])))
    # expiry YYMMDD
    parts[21:27] = list(fix_digits(''.join(parts[21:27])))
    # personal number
    parts[28:42] = list(fix_digits(''.join(parts[28:42])))
    return L1, ''.join(parts)

def extract_passport(image_path: str, fix: bool = False) -> dict:
    """
    Read an image path, OCR the MRZ, parse TD3, and return a dict of fields.
    Set fix=True to apply light MRZ corrections before parsing.
    """
    ocr = get_ocr()
    img = load_image_bgr(image_path)
    L1, L2 = extract_mrz_two_lines(img, ocr)
    if fix:
        L1, L2 = _fix_common_mrz_confusions(L1, L2)
    return parse_td3(L1, L2)

def save_debug_overlay(image_path: str, out_path: str):
    """
    Save a debug image drawing the MRZ region we used.
    """
    img = load_image_bgr(image_path)
    crop, bbox = find_mrz_crop_bbox(img)
    if bbox is None:
        # fallback: bottom band
        H, W = img.shape[:2]
        band_h = max(90, int(H*0.14))
        bbox = (30, H-band_h-30, W-30, H-30)
    x0,y0,x1,y1 = bbox
    dbg = img.copy()
    cv2.rectangle(dbg, (x0,y0), (x1,y1), (0,255,0), 2)
    cv2.imwrite(out_path, dbg)

_MRZ_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ<0123456789")

def _is_mrzish(s: str) -> bool:
    u = _normalize_line(s)
    if any(ch not in _MRZ_ALLOWED for ch in u):
        return False
    if u.count("<") < 5:
        return False
    return 34 <= len(u) <= 47