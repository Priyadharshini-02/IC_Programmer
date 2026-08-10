"""
IC Inspection System
====================
Checks IC chips for:
  1. Presence / Absence  — YOLO (best.pt)
  2. Orientation         — Dominant-edge + asymmetry NCC voter
  3. Pin Health          — find_peaks peak-matching on gradient projection
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
SCORE_THRESHOLD         = 0.30
HARD_SPATIAL_RADIUS_PX  = 150.0
ORIENT_THRESHOLD_DEG    = 45.0   # |rotation| above this → WRONG ORIENTATION
PIN_HEALTH_WARN         = 0.70   # overall health below this → pin FAIL
PIN_HEALTH_SIDE_WARN    = 0.35   # any single side below this → pin FAIL
YOLO_PATH               = r'best.pt'
CONF_THRESHOLD          = 0.60
MIN_ROI_PIN_HEALTH_PX   = 200
PIN_HEALTH_MIN_CHIP_SIM = 0.50


# ─────────────────────────────────────────────────────────────────────────────
#  AdvancedImageAligner  — orientation via edge-direction + 4-way NCC voter
#
#  WHY NOT ORB:
#    QFP/LQFP chips have 4-fold symmetric pin patterns.  ORB matches pins on
#    opposite sides as valid feature pairs → RANSAC produces random rotations
#    (−15° to +68°) on the same correctly-placed chip, frame after frame.
#
#  WHY NOT LOG-POLAR PHASE CORRELATION:
#    IC chips have strong DC and rectangular-boundary artefacts in their FFT.
#    The log-polar spectrum always produces a ~45° shift regardless of actual
#    rotation (hits the ±45° clamp every time).
#
#  WHAT WORKS — two-signal voter:
#
#  Signal 1 — Dominant-edge direction (primary, fast)
#    The gradient magnitude image of a QFP chip is dominated by the pin rows.
#    We compute the dominant gradient angle in 4 oriented zones (top/bottom/
#    left/right strips) by finding the mode of the gradient orientation
#    histogram (36 bins, 0–180°).  The dominant angle shifts by exactly 90°
#    when the chip is rotated 90°.  We compare this angle to the reference
#    chip's dominant angle to find the rotation quadrant.
#
#  Signal 2 — 4-way NCC voter at full-resolution centre crop (confirmation)
#    The centre 60% of the chip contains the IC body marking, text, and chamfer
#    which are NOT 4-fold symmetric.  NCC at 4 rotations (0/90/180/270) on
#    this crop gives the correct quadrant reliably.  We use 3 scales (64/96/128)
#    for voting stability.
#
#  Decision: both signals must agree on the winning quadrant.
#    If they disagree or either is inconclusive → report 0° (no false positive).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RegistrationConfig:
    confidence_threshold: float = 0.40   # NCC gap needed to declare a winner


class AdvancedImageAligner:

    def __init__(self, config: RegistrationConfig = None):
        self.config = config or RegistrationConfig()

    def register_images(self, reference_img: np.ndarray,
                        test_img: np.ndarray) -> Dict:
        ref_gray  = self._preprocess(reference_img)
        test_gray = self._preprocess(test_img)
        return self._orientation_vote(ref_gray, test_gray)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
               if len(image.shape) == 3 else image.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    # ── Signal 1: dominant edge direction per strip zone ─────────────────
    def _dominant_edge_angle(self, gray: np.ndarray) -> float:
        """
        Return the dominant gradient orientation (0–180°) of the chip.
        Uses only the 4 border strips where pin rows live (same STRIP_FRAC
        as PinHealthChecker).  The mode of the gradient-orientation histogram
        (weighted by magnitude) gives the pin-row direction reliably.
        """
        h, w    = gray.shape[:2]
        sw_h    = max(6, int(h * 0.20))
        sw_w    = max(6, int(w * 0.20))

        strips = [
            gray[0:sw_h,      0:w],       # TOP
            gray[h-sw_h:h,    0:w],       # BOTTOM
            gray[0:h,         0:sw_w],    # LEFT
            gray[0:h,         w-sw_w:w],  # RIGHT
        ]

        all_mag = []
        all_ang = []
        for s in strips:
            gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx*gx + gy*gy)
            ang = (np.degrees(np.arctan2(np.abs(gy), np.abs(gx)))) % 180.0
            all_mag.append(mag.ravel())
            all_ang.append(ang.ravel())

        mag_all = np.concatenate(all_mag)
        ang_all = np.concatenate(all_ang)

        # Weighted histogram — bins every 5°
        hist, edges = np.histogram(ang_all, bins=36, range=(0, 180),
                                   weights=mag_all)
        peak_bin = int(np.argmax(hist))
        return float((edges[peak_bin] + edges[peak_bin + 1]) / 2.0)

    # ── Signal 2: 4-way NCC voter on centre-body crop ────────────────────
    def _body_ncc_voter(self, ref_gray: np.ndarray,
                        test_gray: np.ndarray) -> Dict:
        """
        Multi-scale NCC majority vote on the centre 60% crop of the chip.
        The IC body (text, dot, chamfer) is NOT 4-fold symmetric, making this
        crop much more discriminative than the full chip for 0 vs 180° and
        90 vs 270°.

        Returns dict with keys: success, rotation, confidence, message.
        """
        h, w  = ref_gray.shape[:2]
        cy0   = int(h * 0.20);  cy1 = int(h * 0.80)
        cx0   = int(w * 0.20);  cx1 = int(w * 0.80)
        ref_c = ref_gray[cy0:cy1, cx0:cx1]
        tst_c = test_gray[cy0:cy1, cx0:cx1]

        def _ncc(a, b):
            a = a.astype(np.float32) - a.mean()
            b = b.astype(np.float32) - b.mean()
            denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b)) + 1e-8
            return float(np.sum(a * b) / denom)

        angles = [0, 90, 180, 270]
        votes  = {a: 0 for a in angles}
        scores_128 = {}

        for sz in [64, 96, 128]:
            r = cv2.resize(ref_c, (sz, sz))
            t = cv2.resize(tst_c, (sz, sz))
            sc = {0: _ncc(r, t)}
            for k, deg in ((1, 90), (2, 180), (3, 270)):
                sc[deg] = _ncc(r, np.rot90(t, k))
            winner = max(sc, key=sc.__getitem__)
            votes[winner] += 1
            if sz == 128:
                scores_128 = sc

        best_deg   = max(votes, key=votes.__getitem__)
        n_votes    = votes[best_deg]
        best_score = scores_128.get(best_deg, 0.0)
        sv         = sorted(scores_128.values(), reverse=True)
        ncc_gap    = sv[0] - sv[1] if len(sv) >= 2 else 0.0

        if n_votes < 2:
            return {'success': False, 'rotation': 0.0, 'confidence': 0.0,
                    'message': f'body-NCC: no majority  votes={votes}'}
        if ncc_gap < self.config.confidence_threshold:
            return {'success': False, 'rotation': 0.0, 'confidence': 0.0,
                    'message': f'body-NCC: ambiguous gap={ncc_gap:.3f}'}

        rot = float(best_deg if best_deg <= 180 else best_deg - 360)
        return {'success': True, 'rotation': rot,
                'confidence': float(ncc_gap),
                'message': f'body-NCC votes={votes} best={best_deg}° gap={ncc_gap:.3f}'}

    # ── Combined orientation decision ─────────────────────────────────────
    def _orientation_vote(self, ref_gray: np.ndarray,
                          test_gray: np.ndarray) -> Dict:
        """
        Combine dominant-edge angle and body-NCC voter.

        Edge-angle gives the raw angle difference between ref and test.
        Body-NCC voter gives the quadrant (0/90/180/270).
        They must agree on the nearest 90° quadrant; if not → report 0°.
        """
        # Signal 1: edge angle difference
        ref_angle  = self._dominant_edge_angle(ref_gray)
        tst_angle  = self._dominant_edge_angle(test_gray)
        raw_diff   = tst_angle - ref_angle
        # Normalise to (−90, +90] — edge orientation is 0–180° so ambiguity
        # is ±90°; the NCC voter resolves the 0/180 ambiguity.
        while raw_diff >  90.0: raw_diff -= 180.0
        while raw_diff <= -90.0: raw_diff += 180.0
        edge_quad = int(round(raw_diff / 90.0)) * 90  # nearest 0 or ±90

        # Signal 2: body NCC voter
        ncc_res  = self._body_ncc_voter(ref_gray, test_gray)
        ncc_rot  = float(ncc_res['rotation'])
        ncc_quad = int(round(ncc_rot / 90.0)) * 90

        print(f"  [Orient] edge_diff={raw_diff:+.1f}° edge_quad={edge_quad}°"
              f"  ncc_rot={ncc_rot:+.0f}° ncc_quad={ncc_quad}°"
              f"  ncc_ok={ncc_res['success']}  gap={ncc_res['confidence']:.3f}")

        # Agreement check
        if not ncc_res['success']:
            # NCC inconclusive — trust edge angle alone only if it's clearly
            # non-zero (>30°); otherwise assume correct.
            if abs(raw_diff) > 30.0:
                final_rot = float(edge_quad)
                msg = f'NCC inconclusive, edge-only rot={final_rot}'
            else:
                final_rot = 0.0
                msg = 'NCC inconclusive, edge small → assumed correct'
        elif edge_quad == ncc_quad:
            # Both agree → high confidence
            final_rot = float(ncc_quad)
            msg = f'edge+NCC agree rot={final_rot}'
        else:
            # Disagreement → do not flag wrong orientation
            final_rot = 0.0
            msg = (f'edge({edge_quad}°) vs NCC({ncc_quad}°) disagree → assumed correct')

        return {
            'success':     True,
            'rotation':    final_rot,
            'confidence':  ncc_res['confidence'],
            'num_matches': 0,
            'num_inliers': 0,
            'message':     msg,
        }

    @staticmethod
    def _fail(message: str) -> Dict:
        return {'success': False, 'rotation': 0.0, 'confidence': 0.0,
                'num_matches': 0, 'num_inliers': 0, 'message': message}


# ─────────────────────────────────────────────────────────────────────────────
#  IC Detector  (YOLO best.pt)
# ─────────────────────────────────────────────────────────────────────────────
class ICDetector:
    def __init__(self, model_path: str = YOLO_PATH, conf: float = CONF_THRESHOLD):
        self.conf        = conf
        print("model confidence: ", conf)
        self._model      = None
        self._model_path = model_path

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")
        mp = Path(self._model_path)
        if not mp.exists():
            raise FileNotFoundError(f"YOLO model not found: {mp.resolve()}")
        self._model = YOLO(str(mp))
        print(f"[YOLO] Loaded model: {mp}")
        return self._model

    def detect(self, image: np.ndarray) -> List[Dict]:
        model   = self._get_model()
        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb, conf=self.conf, verbose=False)
        dets    = []
        for r in results:
            for b in r.boxes:
                dets.append({'box':   [float(v) for v in b.xyxy[0].tolist()],
                             'score': float(b.conf[0]),
                             'class': 'IC_chip'})
        print(f"[YOLO] Detected {len(dets)} IC_chip(s)  (conf>={self.conf})")
        return dets


# ─────────────────────────────────────────────────────────────────────────────
#  Pin Health Checker — direct peak-matching on gradient projection
#
#  Per-strip pipeline:
#    1. CLAHE grayscale
#    2. Upscale 4x → Sobel magnitude → 1-D mean projection ⊥ to pin rows
#    3. Gaussian smooth (sigma = SMOOTH_SIGMA)
#    4. find_peaks on the REFERENCE projection
#         prominence ≥ PROM_FRAC × projection range
#         distance   ≥ estimated pitch × 0.5  (pitch from autocorrelation)
#    5. For each reference peak, sample the TEST projection at the same X:
#         ratio = test_height / ref_height
#         ratio < MISS_RATIO  → MISSING
#         ratio < BENT_RATIO  → BENT
#         else                → healthy
#    6. Bridge: valley between consecutive ref peaks is
#         > BRIDGE_RATIO × ref_valley AND > 50% of ref peak mean
#    7. side_score = healthy_pins / total_ref_pins
#
#  health_score = mean(4 side scores)
# ─────────────────────────────────────────────────────────────────────────────
class PinHealthChecker:

    STRIP_FRAC   = 0.20    # border strip as fraction of chip dimension
    UPSAMPLE     = 4       # upscale before gradient projection
    CLAHE_CLIP   = 3.0
    CLAHE_TILE   = (4, 4)
    SMOOTH_SIGMA = 1.5     # Gaussian sigma on upscaled projection

    # Peak detection
    PROM_FRAC = 0.08       # prominence >= this × projection range
    MIN_PEAKS = 2          # skip analysis if fewer ref peaks found

    # Defect classification
    MISS_RATIO   = 0.35    # test/ref peak ratio below this → MISSING
    BENT_RATIO   = 0.65    # test/ref peak ratio below this → BENT
    BRIDGE_RATIO = 1.40    # test valley / ref valley above this → BRIDGED

    def check(self, test_chip: np.ndarray,
              ref_chip: np.ndarray = None) -> dict:
        if ref_chip  is None or ref_chip.size  == 0: return self._empty()
        if test_chip is None or test_chip.size == 0: return self._empty()
        if test_chip.shape[0] < 20 or test_chip.shape[1] < 20: return self._empty()

        if ref_chip.shape != test_chip.shape:
            test_chip = cv2.resize(test_chip,
                                   (ref_chip.shape[1], ref_chip.shape[0]),
                                   interpolation=cv2.INTER_AREA)

        ref_g  = self._clahe_gray(ref_chip)
        test_g = self._clahe_gray(test_chip)

        h, w  = ref_g.shape[:2]
        sw_h  = max(6, int(h * self.STRIP_FRAC))
        sw_w  = max(6, int(w * self.STRIP_FRAC))

        strips = {
            "TOP":    (ref_g[0:sw_h,       0:w],      test_g[0:sw_h,       0:w],      1),
            "BOTTOM": (ref_g[h-sw_h:h,     0:w],      test_g[h-sw_h:h,     0:w],      1),
            "LEFT":   (ref_g[0:h,          0:sw_w],   test_g[0:h,          0:sw_w],   0),
            "RIGHT":  (ref_g[0:h,          w-sw_w:w], test_g[0:h,          w-sw_w:w], 0),
        }

        side_scores  = {}
        side_details = {}
        n_miss = n_bent = n_bridge = 0

        for side, (rs, ts, axis) in strips.items():
            d = self._analyse_strip(rs, ts, axis)
            side_scores[side]  = d["score"]
            side_details[side] = d
            n_miss   += d["missing"]
            n_bent   += d["bent"]
            n_bridge += d["bridged"]

        health = float(np.mean(list(side_scores.values())))
        total  = n_miss + n_bent + n_bridge

        print("  [PinHealth] " +
              "  ".join(f"{s}:{v:.2f}" for s, v in side_scores.items()) +
              f"  overall:{health:.2f}")
        for s, d in side_details.items():
            if d["missing"] or d["bent"] or d["bridged"]:
                print(f"    [{s}] miss={d['missing']} bent={d['bent']}"
                      f" bridge={d['bridged']}  n_pins={d['n_ref_pins']}"
                      f"  score={d['score']:.2f}")

        return {
            "health_score": health,  "ssim_score": health,  "is_ic": True,
            "total_pins":   total,   "missing":    n_miss,  "bent":  n_bent,
            "bridged":      n_bridge,
            "ok":           1 if total == 0 else 0,
            "defect_pixels": total,
            "side_scores":  side_scores,
            "side_details": side_details,
        }

    # ─────────────────────────────────────────────────────────────────────
    def _analyse_strip(self, ref_strip: np.ndarray,
                       test_strip: np.ndarray, axis: int) -> dict:
        empty = {"score": 1.0, "missing": 0, "bent": 0, "bridged": 0,
                 "n_ref_pins": 0, "fft": 1.0, "hog": 1.0, "ncc": 1.0,
                 "ssim": 1.0, "grad": 1.0}

        rp = self._gradient_projection(ref_strip,  axis)
        tp = self._gradient_projection(test_strip, axis)
        if rp is None or tp is None:
            return empty

        from scipy.signal import find_peaks

        rng = float(rp.max() - rp.min())
        if rng < 5.0:
            return empty   # flat strip — no pins on this side

        # ── Estimate pin pitch via autocorrelation ────────────────────────
        rp_mc = rp - rp.mean()
        acorr = np.correlate(rp_mc, rp_mc, mode='full')[len(rp_mc) - 1:]
        acorr = acorr / (acorr[0] + 1e-8)
        from scipy.signal import find_peaks as _fp
        pitch_cands, _ = _fp(acorr[1:], height=0.10)
        pitch = int(pitch_cands[0]) + 1 if len(pitch_cands) > 0 else max(4, len(rp) // 12)

        min_dist = max(3, int(pitch * 0.5))
        prom     = max(3.0, rng * self.PROM_FRAC)

        ref_peaks, _ = find_peaks(rp, prominence=prom, distance=min_dist)
        if len(ref_peaks) < self.MIN_PEAKS:
            return empty

        # ── Per-peak defect classification ───────────────────────────────
        n_miss = n_bent = healthy = 0
        for pk in ref_peaks:
            rh    = float(rp[pk])
            th    = float(tp[pk])
            ratio = th / (rh + 1e-6)
            if   ratio < self.MISS_RATIO: n_miss  += 1
            elif ratio < self.BENT_RATIO: n_bent  += 1
            else:                         healthy  += 1

        # ── Bridge detection ──────────────────────────────────────────────
        n_bridge = 0
        for k in range(len(ref_peaks) - 1):
            pa, pb       = ref_peaks[k], ref_peaks[k + 1]
            ref_valley   = float(rp[pa:pb + 1].min())
            tst_valley   = float(tp[pa:pb + 1].min())
            ref_pk_mean  = float((rp[pa] + rp[pb]) / 2.0)
            if (ref_pk_mean > 5.0
                    and ref_valley > 1e-6
                    and tst_valley > ref_valley * self.BRIDGE_RATIO
                    and tst_valley > ref_pk_mean * 0.50):
                n_bridge += 1

        n_ref = len(ref_peaks)
        score = float(healthy) / float(n_ref)

        return {
            "score":      float(np.clip(score, 0.0, 1.0)),
            "missing":    n_miss,
            "bent":       n_bent,
            "bridged":    n_bridge,
            "n_ref_pins": n_ref,
            "fft": score, "hog": score, "ncc": score,
            "ssim": score, "grad": score,
        }

    # ─────────────────────────────────────────────────────────────────────
    def _gradient_projection(self, strip: np.ndarray, axis: int):
        """Upscale → Sobel magnitude → mean projection → Gaussian smooth."""
        from scipy.ndimage import gaussian_filter1d
        if strip.size == 0 or strip.shape[0] < 3 or strip.shape[1] < 3:
            return None
        us  = self.UPSAMPLE
        sup = cv2.resize(strip,
                         (strip.shape[1] * us, strip.shape[0] * us),
                         interpolation=cv2.INTER_CUBIC).astype(np.float32)
        gx  = cv2.Sobel(sup, cv2.CV_32F, 1, 0, ksize=3)
        gy  = cv2.Sobel(sup, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        proj = mag.mean(axis=axis)
        return gaussian_filter1d(proj, sigma=self.SMOOTH_SIGMA * us)

    def _clahe_gray(self, img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        return cv2.createCLAHE(clipLimit=self.CLAHE_CLIP,
                               tileGridSize=self.CLAHE_TILE).apply(g)

    @staticmethod
    def _empty() -> dict:
        return {
            "health_score": 1.0,  "ssim_score": 1.0,  "is_ic": False,
            "total_pins":   0,    "missing":    0,     "bent":  0,
            "bridged":      0,    "ok":         0,     "defect_pixels": 0,
            "side_scores":  {"TOP": 1.0, "BOTTOM": 1.0, "LEFT": 1.0, "RIGHT": 1.0},
            "side_details": {},
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Polarity Detector
# ─────────────────────────────────────────────────────────────────────────────
class PolarityDetector:
    POLAR_CLASSES = {'capacitor', 'electrolytic', 'diode', 'led',
                     'transistor', 'ic', 'connector'}

    def __init__(self, angle_threshold: float = 30.0):
        self.angle_threshold = angle_threshold
        self.aligner = AdvancedImageAligner()

    def detect_polarity(self, component_image: np.ndarray,
                        component_class: str) -> dict:
        has_polarity = any(pc in component_class.lower()
                           for pc in self.POLAR_CLASSES)
        return {'has_polarity': has_polarity,
                'chip': component_image,
                'component_class': component_class}

    def compare_polarity(self, ref_polarity: dict, test_polarity: dict,
                         angle_threshold: float = None) -> dict:
        threshold = angle_threshold if angle_threshold is not None \
                    else self.angle_threshold
        if not ref_polarity.get('has_polarity') or \
           not test_polarity.get('has_polarity'):
            return {'correct': True, 'angle_difference': 0.0,
                    'confidence': 1.0, 'num_matches': 0,
                    'method': 'skipped_non_polar',
                    'message': 'Component has no polarity — check skipped'}

        align = self.aligner.register_images(ref_polarity['chip'],
                                             test_polarity['chip'])
        if not align['success']:
            return {'correct': True, 'angle_difference': 0.0,
                    'confidence': 0.0, 'num_matches': 0,
                    'method': 'registration_failed',
                    'message': align['message']}

        raw = float(align['rotation'])
        while raw >  180.0: raw -= 360.0
        while raw <= -180.0: raw += 360.0
        is_wrong = abs(raw) > threshold
        method = 'body_ncc_voter'
        print(f"  [Polarity] class={ref_polarity.get('component_class', '?')}"
              f"  rotation={raw:+.1f}°  conf={align['confidence']:.2f}"
              f"  → {'WRONG' if is_wrong else 'OK'}")
        return {'correct': not is_wrong, 'angle_difference': raw,
                'confidence': float(align['confidence']),
                'num_matches': align['num_matches'],
                'method': method, 'message': align['message']}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def hungarian_assignment(score_matrix: np.ndarray) -> List[Tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(-score_matrix)
        return list(zip(ri.tolist(), ci.tolist()))
    except ImportError:
        used: set = set(); pairs = []
        for i in np.argsort(-score_matrix.max(axis=1)):
            row = score_matrix[i].copy(); row[list(used)] = -1
            j = int(np.argmax(row))
            if row[j] >= 0: pairs.append((int(i), j)); used.add(j)
        return pairs


def box_iou_with_margin(a, b, margin=40):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1 - margin), max(ay1, by1 - margin)
    ix2, iy2 = min(ax2, bx2 + margin), min(ay2, by2 + margin)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    cont = (ix2 - ix1) * (iy2 - iy1) / area_a if ix2 > ix1 and iy2 > iy1 else 0.0
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    diag = float(np.sqrt((bx2 - bx1) ** 2 + (by2 - by1) ** 2)) + 1e-6
    prox = float(np.exp(-0.5 * (float(np.sqrt((acx - bcx) ** 2 +
                                               (acy - bcy) ** 2)) / (diag / 2)) ** 2))
    return float(np.clip(0.7 * cont + 0.3 * prox, 0.0, 1.0))


def ncc_similarity(chip1, chip2):
    if chip1 is None or chip2 is None or chip1.size == 0 or chip2.size == 0:
        return 0.0
    g1 = cv2.cvtColor(chip1, cv2.COLOR_BGR2GRAY) if len(chip1.shape) == 3 else chip1
    g2 = cv2.cvtColor(chip2, cv2.COLOR_BGR2GRAY) if len(chip2.shape) == 3 else chip2
    g1 = cv2.createCLAHE(2.0, (4, 4)).apply(cv2.resize(g1, (96, 96))).astype(np.float32)
    g2 = cv2.createCLAHE(2.0, (4, 4)).apply(cv2.resize(g2, (96, 96))).astype(np.float32)
    g1 -= g1.mean(); g2 -= g2.mean()
    denom = float(np.linalg.norm(g1)) * float(np.linalg.norm(g2)) + 1e-8
    return float(np.clip((float(np.sum(g1 * g2)) / denom + 1.0) / 2.0, 0.0, 1.0))


def crop_chip(image, box, size=128):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    chip = image[y1:y2, x1:x2]
    if chip.size == 0: return np.zeros((size, size, 3), dtype=np.uint8)
    chip = cv2.resize(chip, (size, size))
    lab = cv2.cvtColor(chip, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(2.0, (4, 4)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def crop_chip_native(image, box) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    chip = image[y1:y2, x1:x2]
    if chip.size == 0: return np.zeros((64, 64, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(chip, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(chip, 1.5, blurred, -0.5, 0)


# ─────────────────────────────────────────────────────────────────────────────
#  IC Inspector
# ─────────────────────────────────────────────────────────────────────────────
PIN_STRIP_CFG = {
    "strip_frac": 0.18, "upsample": 3, "proj_smooth_sigma": 1.5,
    "min_proj_amp": 5.0, "peak_prominence": 0.10,
    "missing_thresh": 0.45, "bent_thresh": 0.65, "bridge_fill": 0.60,
    "side_ncc_fail": 0.60, "clahe_clip": 3.0, "clahe_tile": (4, 4),
    "ecc_iterations": 80, "ecc_epsilon": 1e-3,
}


class ICInspector:
    def __init__(self, yolo_path=YOLO_PATH, conf_threshold=CONF_THRESHOLD, roi=None):
        self.detector   = ICDetector(model_path=yolo_path, conf=conf_threshold)
        self.aligner    = AdvancedImageAligner()
        self.pin_health = PinHealthChecker()
        self.roi        = roi
        self.reference_image          = None
        self.reference_detections     = None
        self.ref_chips_cache          = []
        self.ref_chips_native_cache   = []
        self.ref_from_file            = False
        self.conf                     = conf_threshold
        print("model confidence: ", conf_threshold)

    def _filter_by_roi(self, dets):
        if self.roi is None: return dets
        rx1, ry1, rx2, ry2 = self.roi
        kept = [d for d in dets
                if rx1 <= (d['box'][0] + d['box'][2]) / 2 <= rx2
                and ry1 <= (d['box'][1] + d['box'][3]) / 2 <= ry2]
        if len(dets) - len(kept):
            print(f"[ROI] Filtered {len(dets) - len(kept)} outside")
        return kept

    def _roi_large_enough_for_pin_health(self) -> bool:
        if self.roi is None: return True
        rx1, ry1, rx2, ry2 = self.roi
        roi_w, roi_h = rx2 - rx1, ry2 - ry1
        print("roi_w: ", roi_w, "roi_h: ", roi_h)
        ok = (roi_w >= MIN_ROI_PIN_HEALTH_PX and roi_h >= MIN_ROI_PIN_HEALTH_PX)
        if not ok:
            print(f"[PinHealth] SKIPPED — ROI too small"
                  f" ({roi_w}x{roi_h} px < {MIN_ROI_PIN_HEALTH_PX} px)")
        return ok

    def set_reference(self, image: np.ndarray):
        print("\nProcessing reference image...")
        dets = self._filter_by_roi(self.detector.detect(image))
        if not dets:
            print("Warning: No IC components detected inside ROI in reference.")
            return
        ref_det         = dets[0]
        ref_chip        = crop_chip(image, ref_det['box'])
        ref_chip_native = crop_chip_native(image, ref_det['box'])
        self.reference_image        = ref_chip
        self.reference_detections   = dets
        self.ref_chips_cache        = [ref_chip]
        self.ref_chips_native_cache = [ref_chip_native]
        self.ref_from_file          = False
        Path('ref').mkdir(exist_ok=True)
        cv2.imwrite('ref/input_frame.png', image)
        cv2.imwrite('ref/chip_0.png', ref_chip)
        cv2.imwrite('ref/chip_0_native.png', ref_chip_native)
        roi_vis = image.copy()
        x1, y1, x2, y2 = map(int, ref_det['box'])
        cv2.rectangle(roi_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(roi_vis, 'REF', (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.roi:
            rx1, ry1, rx2, ry2 = self.roi
            cv2.rectangle(roi_vis, (rx1, ry1), (rx2, ry2), (255, 165, 0), 2)
        cv2.imwrite('ref/roi_frame.png', roi_vis)
        print('Saved: ref/input_frame.png  ref/chip_0.png  '
              'ref/chip_0_native.png  ref/roi_frame.png')
        print(f'Reference set with {len(dets)} IC component(s) inside ROI')

    def load_reference_from_file(self, ref_chip_path: str):
        print(f"\nLoading reference chip from: {ref_chip_path}")
        chip = cv2.imread(ref_chip_path)
        if chip is None:
            print(f"Error: Cannot read reference chip from '{ref_chip_path}'")
            return
        chip = crop_chip(chip, [0, 0, chip.shape[1], chip.shape[0]])
        native_path = str(ref_chip_path).replace('chip_0.png', 'chip_0_native.png')
        ni = cv2.imread(native_path)
        if ni is not None:
            native_chip = crop_chip_native(ni, [0, 0, ni.shape[1], ni.shape[0]])
            print(f'  Native ref loaded: {native_path}')
        else:
            native_chip = crop_chip_native(chip, [0, 0, chip.shape[1], chip.shape[0]])
            print('  Native ref not found, using 128x128 fallback')
        self.reference_image        = chip
        self.ref_chips_cache        = [chip]
        self.ref_chips_native_cache = [native_chip]
        self.reference_detections   = [{'box': [0, 0, chip.shape[1], chip.shape[0]],
                                         'score': 1.0, 'class': 'IC_chip'}]
        self.ref_from_file = True
        print('Reference chip loaded — YOLO/ROI check skipped')

    def detect_and_compare(self, test_image: np.ndarray, save_images=False) -> dict:
        if self.reference_image is None:
            return {'status': 'error', 'message': 'No reference image set'}

        test_dets = self._filter_by_roi(self.detector.detect(test_image))
        if not test_dets:
            missing = [{'ref_id': i, 'box': d['box'], 'status': 'absent'}
                       for i, d in enumerate(self.reference_detections)]
            return {'status': 'success', 'num_components': 0, 'components': [],
                    'missing_components': missing,
                    'comparison': {'matched': 0, 'extra': 0,
                                   'missing': len(self.reference_detections),
                                   'wrong_orientation': 0, 'pin_failures': 0}}

        test_chips        = [crop_chip(test_image, d['box']) for d in test_dets]
        test_chips_native = [crop_chip_native(test_image, d['box']) for d in test_dets]
        N_test, N_ref = len(test_dets), len(self.reference_detections)

        score_matrix = np.zeros((N_test, N_ref), dtype=np.float32)
        iou_matrix   = np.zeros((N_test, N_ref), dtype=np.float32)
        test_centres = [((d['box'][0] + d['box'][2]) / 2,
                         (d['box'][1] + d['box'][3]) / 2) for d in test_dets]
        ref_centres  = [((d['box'][0] + d['box'][2]) / 2,
                         (d['box'][1] + d['box'][3]) / 2) for d in self.reference_detections]

        for i in range(N_test):
            tcx, tcy = test_centres[i]
            for j in range(N_ref):
                rcx, rcy = ref_centres[j]
                if not self.ref_from_file:
                    if float(np.sqrt((tcx - rcx) ** 2 +
                                     (tcy - rcy) ** 2)) > HARD_SPATIAL_RADIUS_PX:
                        continue
                sim = ncc_similarity(test_chips[i], self.ref_chips_cache[j])
                iou = box_iou_with_margin(test_dets[i]['box'],
                                          self.reference_detections[j]['box'])
                iou_matrix[i, j]   = iou
                score_matrix[i, j] = 0.65 * sim + 0.35 * iou

        assignment_map: Dict[int, int] = {}
        for ti, rj in hungarian_assignment(score_matrix):
            if score_matrix[ti, rj] < SCORE_THRESHOLD: continue
            if not self.ref_from_file:
                tcx, tcy = test_centres[ti]; rcx, rcy = ref_centres[rj]
                if float(np.sqrt((tcx - rcx) ** 2 +
                                 (tcy - rcy) ** 2)) > HARD_SPATIAL_RADIUS_PX:
                    continue
            assignment_map[ti] = rj

        components_info = []

        for i in range(N_test):
            ref_idx  = assignment_map.get(i, -1)
            best_sim = ncc_similarity(test_chips[i],
                                      self.ref_chips_cache[ref_idx]) if ref_idx >= 0 else 0.0

            # ── Orientation ───────────────────────────────────────────────
            ic_orient = {
                'correct': True, 'rotation_deg': 0.0, 'rotation_step': 0,
                'confidence': 0.0, 'num_matches': 0, 'reason': 'not_checked'
            }
            if ref_idx >= 0:
                ref_c = self.ref_chips_cache[ref_idx]
                tst_c = test_chips[i]
                align = self.aligner.register_images(ref_c, tst_c)
                raw   = float(align['rotation'])
                while raw >  180.0: raw -= 360.0
                while raw <= -180.0: raw += 360.0
                is_wrong = abs(raw) > ORIENT_THRESHOLD_DEG
                _sr      = int(round(raw / 90.0)) * 90
                rot_step = _sr if _sr != -180 else 180
                ic_orient = {
                    'correct':         not is_wrong,
                    'rotation_deg':    raw,
                    'rotation_step':   rot_step,
                    'confidence':      float(align['confidence']),
                    'num_matches':     align['num_matches'],
                    'reason':          'ok' if not is_wrong else f'rotated_{rot_step}deg',
                }
                print(f"  [Orient] comp_id={i} rot={raw:+.1f}° step={rot_step}°"
                      f"  conf={align['confidence']:.3f}"
                      f"  → {'WRONG' if is_wrong else 'OK'}"
                      f"  [{align['message']}]")

            ic_orient_wrong = (
                not ic_orient['correct']
                and ic_orient['reason'] != 'not_checked'
            )

            # ── Pin health ────────────────────────────────────────────────
            chip_sim_ok  = best_sim >= PIN_HEALTH_MIN_CHIP_SIM
            run_pin_health = (
                ref_idx >= 0
                and self._roi_large_enough_for_pin_health()
                and not ic_orient_wrong
                and chip_sim_ok
            )
            if run_pin_health:
                if ref_idx < len(self.ref_chips_native_cache):
                    ref_native = self.ref_chips_native_cache[ref_idx]
                else:
                    c = self.ref_chips_cache[ref_idx]
                    ref_native = crop_chip_native(c, [0, 0, c.shape[1], c.shape[0]])
                pin = self.pin_health.check(test_chips_native[i], ref_native)
            else:
                if ic_orient_wrong:
                    print(f"  [PinHealth] SKIPPED — orientation wrong"
                          f" ({ic_orient['rotation_deg']:+.1f}°)")
                elif not chip_sim_ok:
                    print(f"  [PinHealth] SKIPPED — chip sim too low"
                          f" ({best_sim:.3f} < {PIN_HEALTH_MIN_CHIP_SIM})")
                pin = self.pin_health._empty()

            _ss     = pin.get('side_scores', {})
            _worst  = min(_ss.values()) if _ss else 1.0
            _n_def  = pin.get('missing', 0) + pin.get('bent', 0) + pin.get('bridged', 0)
            pin_ok  = (
                (pin.get('health_score', 1.0) >= PIN_HEALTH_WARN
                 and _worst >= PIN_HEALTH_SIDE_WARN)
                or _n_def == 0
            )

            status = ('wrong_orientation' if ic_orient_wrong else 'present') \
                     if ref_idx >= 0 else 'extra'

            components_info.append({
                'id': i, 'box': test_dets[i]['box'],
                'score': test_dets[i]['score'],
                'class': test_dets[i].get('class', 'IC_chip'),
                'similarity': best_sim,
                'iou_score': float(iou_matrix[i, ref_idx]) if ref_idx >= 0 else 0.0,
                'status': status, 'matched_ref_id': ref_idx if ref_idx >= 0 else None,
                'ic_orient_correct':    ic_orient['correct'],
                'ic_orient_deg':        ic_orient['rotation_deg'],
                'ic_orient_step':       ic_orient['rotation_step'],
                'ic_orient_confidence': ic_orient['confidence'],
                'ic_orient_matches':    ic_orient['num_matches'],
                'ic_orient_reason':     ic_orient['reason'],
                'pin_health_score': pin.get('health_score', 1.0),
                'pin_health_is_ic':  pin.get('is_ic', False),
                'pin_health_ok':     pin_ok,
                'pin_side_scores':   pin.get('side_scores', {}),
                'pin_total':   pin.get('total_pins', 0),
                'pin_missing': pin.get('missing', 0),
                'pin_bent':    pin.get('bent', 0),
                'pin_bridged': pin.get('bridged', 0),
                'pin_ok':      pin.get('ok', 0),
            })

        matched_ref = set(assignment_map.values())
        missing_components = [
            {'ref_id': j, 'box': self.reference_detections[j]['box'], 'status': 'absent'}
            for j in range(N_ref) if j not in matched_ref
        ]

        present_count  = sum(1 for c in components_info if c['status'] == 'present')
        wrong_o_count  = sum(1 for c in components_info if c['status'] == 'wrong_orientation')
        extra_count    = sum(1 for c in components_info if c['status'] == 'extra')
        pin_fail_count = sum(1 for c in components_info
                             if c['pin_health_is_ic'] and not c['pin_health_ok'])

        try:
            from pathlib import Path as _Path
            import time as _time
            _Path('output').mkdir(exist_ok=True)
            _vis = self._create_annotated_image(test_image, components_info, missing_components)
            if self.roi:
                _rx1, _ry1, _rx2, _ry2 = self.roi
                cv2.rectangle(_vis, (_rx1, _ry1), (_rx2, _ry2), (255, 165, 0), 2)
                cv2.putText(_vis, 'ROI', (_rx1, max(0, _ry1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            _ts = _time.strftime('%Y%m%d_%H%M%S')
            _out_path = f'output/result_{_ts}.png'
            cv2.imwrite(_out_path, _vis)
            print(f'[Output] Saved annotated result: {_out_path}')
        except Exception as _e:
            print(f'[Output] Warning: could not save result image: {_e}')

        return {
            'status': 'success', 'num_components': len(components_info),
            'components': components_info, 'missing_components': missing_components,
            'comparison': {'matched': present_count, 'wrong_orientation': wrong_o_count,
                           'extra': extra_count, 'missing': len(missing_components),
                           'pin_failures': pin_fail_count},
        }

    def _create_annotated_image(self, image, components, missing):
        vis = image.copy()
        colours = {'present': (0, 220, 0), 'wrong_orientation': (0, 165, 255),
                   'extra': (0, 0, 255)}
        for c in components:
            x1, y1, x2, y2 = map(int, c['box'])
            if c.get('pin_health_is_ic') and not c.get('pin_health_ok', True):
                col = (0, 200, 255)
            else:
                col = colours.get(c['status'], (200, 200, 200))
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            label = c['status'].upper()
            if c['status'] == 'wrong_orientation':
                label += f" {c.get('ic_orient_deg', 0.0):+.0f}deg"
            if c.get('pin_health_is_ic'):
                ph_ok  = c.get('pin_health_ok', True)
                miss   = c.get('pin_missing', 0)
                bent   = c.get('pin_bent', 0)
                bridge = c.get('pin_bridged', 0)
                label += (f" PIN_FAIL miss={miss} bent={bent} brdg={bridge}"
                          if not ph_ok else " PIN_OK")
            ly = y1 - 6 if y1 > 16 else y2 + 14
            cv2.putText(vis, label, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)
        for m in missing:
            x1, y1, x2, y2 = map(int, m['box'])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (128, 0, 128), 2)
            cv2.putText(vis, 'ABSENT', (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 0, 128), 1)
        return vis


# ─────────────────────────────────────────────────────────────────────────────
#  Interactive CLI app
# ─────────────────────────────────────────────────────────────────────────────
class ICInspectionApp:
    def __init__(self, yolo_path=YOLO_PATH, conf_threshold=CONF_THRESHOLD, roi=None):
        self.conf      = conf_threshold
        print("model confidence: ", conf_threshold)
        self.inspector = ICInspector(yolo_path, conf_threshold, roi)
        self._last_result = None; self._last_image = None

    def run_interactive(self):
        print("\nIC Inspection — orientation via edge-direction + body-NCC voter")
        print("Commands: ref <img>  |  test <img>  |  load_ref <chip>  |  quit\n")
        while True:
            try:    raw = input(">> ").strip()
            except (EOFError, KeyboardInterrupt): break
            if not raw: continue
            cmd = raw.split(); sub = cmd[0].lower()
            if sub in ('quit', 'q', 'exit'): break
            elif sub == 'ref':
                if len(cmd) < 2: print("Usage: ref <image_path>"); continue
                img = cv2.imread(cmd[1])
                if img is None: print(f"Cannot read: {cmd[1]}"); continue
                self.inspector.set_reference(img)
            elif sub == 'load_ref':
                if len(cmd) < 2: print("Usage: load_ref <chip_path>"); continue
                self.inspector.load_reference_from_file(cmd[1])
            elif sub == 'test':
                if len(cmd) < 2: print("Usage: test <image_path>"); continue
                img = cv2.imread(cmd[1])
                if img is None: print(f"Cannot read: {cmd[1]}"); continue
                self._last_image  = img
                self._last_result = self.inspector.detect_and_compare(img)
                self._display_results(self._last_result)
            elif sub == 'teach': self._handle_teach(cmd)
            else: print(f"Unknown command: {sub}")

    def _handle_teach(self, cmd):
        if len(cmd) < 2: print("Usage: teach <image_path>"); return
        img = cv2.imread(cmd[1])
        if img is None: print(f"Cannot read: {cmd[1]}"); return
        self.inspector.set_reference(img)
        print(f"Reference updated from: {cmd[1]}")

    def _display_results(self, result):
        if result['status'] != 'success':
            print(f"Error: {result.get('message', '?')}")
            print("\n[RESULT_BEGIN]")
            print("IC Present (matched): 0"); print("IC Absent  (missing): 1")
            print("IC Extra   (unexpected): 0"); print("Wrong orientation: 0")
            print("Pin health failures: 0"); print("[RESULT_END]")
            return

        comps = result.get('components', [])
        miss  = result.get('missing_components', [])
        cmp   = result.get('comparison', {})

        overall_pass = (cmp.get('wrong_orientation', 0) == 0
                        and cmp.get('missing', 0) == 0
                        and cmp.get('extra', 0) == 0
                        and cmp.get('pin_failures', 0) == 0)

        print("\n" + "=" * 60)
        print(f"INSPECTION RESULT:  {'✓ PASS' if overall_pass else '✗ FAIL'}")
        print("=" * 60)
        print(f"  IC Present (matched):     {cmp.get('matched', 0)}"
              f"  /  {len(self.inspector.reference_detections or [])}")
        print(f"  IC Absent  (missing):     {cmp.get('missing', 0)}")
        print(f"  IC Extra   (unexpected):  {cmp.get('extra', 0)}")
        print(f"  Wrong orientation:        {cmp.get('wrong_orientation', 0)}")
        print(f"  Pin health failures:      {cmp.get('pin_failures', 0)}")
        print("=" * 60)

        sym = {'present': '✓', 'wrong_orientation': '⟳', 'extra': '+', 'absent': '✗'}
        if comps:
            print("\nComponent Details:")
            for c in comps:
                s  = c['status']
                ln = (f"  {sym.get(s, '?')} [{s.upper()}]"
                      f"  id={c['id']}"
                      f"  sim={c['similarity']:.3f}"
                      f"  iou={c.get('iou_score', 0):.2f}")
                deg  = c.get('ic_orient_deg', 0.0)
                conf = c.get('ic_orient_confidence', 0.0)
                nm   = c.get('ic_orient_matches', 0)
                if not c.get('ic_orient_correct', True):
                    ln += (f"\n      Orientation: WRONG"
                           f"  rotation={deg:+.1f}°"
                           f"  step~{c['ic_orient_step']}°"
                           f"  conf={conf:.2f}")
                else:
                    reason = c.get('ic_orient_reason', '')
                    if 'aligner_failed' in reason:
                        ln += f"\n      Orientation: NOT CHECKED  ({reason})"
                    else:
                        ln += (f"\n      Orientation: OK"
                               f"  rotation={deg:+.1f}°"
                               f"  conf={conf:.2f}")
                if c.get('pin_health_is_ic'):
                    ok = c.get('pin_health_ok', True)
                    ln += (f"\n      Pin health:  {'OK' if ok else 'FAIL'}"
                           f"  score={c['pin_health_score']:.0%}"
                           f"  total={c['pin_total']}"
                           f"  miss={c['pin_missing']}"
                           f"  bent={c['pin_bent']}"
                           f"  bridge={c['pin_bridged']}")
                else:
                    ln += "\n      Pin health:  (not an IC package)"
                print(ln)
        if miss:
            print("\nAbsent ICs:")
            for m in miss:
                print(f"  ✗ [ABSENT]  ref_id={m['ref_id']}"
                      f"  box={[round(v) for v in m['box']]}")
        print()

        total_pin_missing = sum(c['pin_missing'] for c in comps if c.get('pin_health_is_ic'))
        total_pin_bent    = sum(c['pin_bent']    for c in comps if c.get('pin_health_is_ic'))
        total_pin_bridge  = sum(c['pin_bridged'] for c in comps if c.get('pin_health_is_ic'))

        worst_orient = None
        for c in comps:
            if not c.get('ic_orient_correct', True):
                if worst_orient is None or abs(c['ic_orient_deg']) > abs(worst_orient['ic_orient_deg']):
                    worst_orient = c
        if worst_orient is None and comps:
            worst_orient = comps[0]

        rotation_deg  = worst_orient['ic_orient_deg']  if worst_orient else 0.0
        rotation_step = worst_orient['ic_orient_step'] if worst_orient else 0

        print("[RESULT_BEGIN]")
        print(f"IC Present (matched): {cmp.get('matched', 0)}")
        print(f"IC Absent  (missing): {cmp.get('missing', 0)}")
        print(f"IC Extra   (unexpected): {cmp.get('extra', 0)}")
        print(f"Wrong orientation: {cmp.get('wrong_orientation', 0)}")
        print(f"Pin health failures: {cmp.get('pin_failures', 0)}")
        print(f"pin_missing={total_pin_missing}")
        print(f"pin_bent={total_pin_bent}")
        print(f"pin_bridge={total_pin_bridge}")
        print(f"rotation={rotation_deg:.2f} step~{rotation_step}")
        print("[RESULT_END]")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    for d in ('output', 'tmp', 'images', 'ref'): Path(d).mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description='IC Inspection')
    parser.add_argument('--yolo-path',      default=YOLO_PATH)
    parser.add_argument('--conf-threshold', type=float, default=CONF_THRESHOLD)
    parser.add_argument('--mode',           default='interactive',
                        choices=['interactive', 'check_roi'])
    parser.add_argument('--image',          default=None)
    parser.add_argument('--roi-coords',     default=None)
    parser.add_argument('--roi-file',       default=None)
    args = parser.parse_args()

    def _load_roi(path):
        if not path: return None
        try:
            vals = open(path).read().split()
            return (int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]))
        except Exception as e:
            print(f"[ROI] Cannot read '{path}': {e}"); return None

    if args.mode == 'check_roi':
        if not args.image or not args.roi_coords:
            print("IC_ABSENT"); raise SystemExit(1)
        img = cv2.imread(args.image)
        if img is None: print("IC_ABSENT"); raise SystemExit(1)
        try:
            vals = args.roi_coords.split()
            if len(vals) != 4: raise ValueError(f"Expected 4 values, got {len(vals)}")
        except Exception as e:
            print(f"IC_ABSENT  # invalid --roi-coords: {e}"); raise SystemExit(1)
        rx1, ry1 = max(0, int(vals[0])), max(0, int(vals[1]))
        rx2, ry2 = min(img.shape[1], int(vals[2])), min(img.shape[0], int(vals[3]))
        try:
            dets  = ICDetector(model_path=args.yolo_path,
                               conf=args.conf_threshold).detect(img)
            found = any(rx1 <= (d['box'][0] + d['box'][2]) // 2 <= rx2
                        and ry1 <= (d['box'][1] + d['box'][3]) // 2 <= ry2
                        for d in dets)
            print("IC_PRESENT" if found else "IC_ABSENT")
        except Exception as e:
            print(f"IC_ABSENT  # error: {e}")
        raise SystemExit(0)

    roi = _load_roi(args.roi_file)
    if roi: print(f"[ROI] x1={roi[0]} y1={roi[1]} x2={roi[2]} y2={roi[3]}")
    try:
        ICInspectionApp(yolo_path=args.yolo_path,
                        conf_threshold=args.conf_threshold,
                        roi=roi).run_interactive()
    except Exception as _e:
        import traceback as _tb
        print(f'[FATAL] {_e}', flush=True); _tb.print_exc()
        print('[RESULT_BEGIN]')
        print('IC Present (matched): 0'); print('IC Absent  (missing): 1')
        print('IC Extra   (unexpected): 0'); print('Wrong orientation: 0')
        print('Pin health failures: 0'); print('pin_missing=0')
        print('pin_bent=0'); print('pin_bridge=0')
        print('rotation=0.00 step~0'); print('[RESULT_END]')"""
IC Inspection System
====================
Checks IC chips for:
  1. Presence / Absence  — YOLO (best.pt)
  2. Orientation         — Dominant-edge + asymmetry NCC voter
  3. Pin Health          — find_peaks peak-matching on gradient projection
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
SCORE_THRESHOLD         = 0.30
HARD_SPATIAL_RADIUS_PX  = 150.0
ORIENT_THRESHOLD_DEG    = 45.0   # |rotation| above this → WRONG ORIENTATION
PIN_HEALTH_WARN         = 0.70   # overall health below this → pin FAIL
PIN_HEALTH_SIDE_WARN    = 0.35   # any single side below this → pin FAIL
YOLO_PATH               = r'best.pt'
CONF_THRESHOLD          = 0.60
MIN_ROI_PIN_HEALTH_PX   = 200
PIN_HEALTH_MIN_CHIP_SIM = 0.50


# ─────────────────────────────────────────────────────────────────────────────
#  AdvancedImageAligner  — orientation via edge-direction + 4-way NCC voter
#
#  WHY NOT ORB:
#    QFP/LQFP chips have 4-fold symmetric pin patterns.  ORB matches pins on
#    opposite sides as valid feature pairs → RANSAC produces random rotations
#    (−15° to +68°) on the same correctly-placed chip, frame after frame.
#
#  WHY NOT LOG-POLAR PHASE CORRELATION:
#    IC chips have strong DC and rectangular-boundary artefacts in their FFT.
#    The log-polar spectrum always produces a ~45° shift regardless of actual
#    rotation (hits the ±45° clamp every time).
#
#  WHAT WORKS — two-signal voter:
#
#  Signal 1 — Dominant-edge direction (primary, fast)
#    The gradient magnitude image of a QFP chip is dominated by the pin rows.
#    We compute the dominant gradient angle in 4 oriented zones (top/bottom/
#    left/right strips) by finding the mode of the gradient orientation
#    histogram (36 bins, 0–180°).  The dominant angle shifts by exactly 90°
#    when the chip is rotated 90°.  We compare this angle to the reference
#    chip's dominant angle to find the rotation quadrant.
#
#  Signal 2 — 4-way NCC voter at full-resolution centre crop (confirmation)
#    The centre 60% of the chip contains the IC body marking, text, and chamfer
#    which are NOT 4-fold symmetric.  NCC at 4 rotations (0/90/180/270) on
#    this crop gives the correct quadrant reliably.  We use 3 scales (64/96/128)
#    for voting stability.
#
#  Decision: both signals must agree on the winning quadrant.
#    If they disagree or either is inconclusive → report 0° (no false positive).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RegistrationConfig:
    confidence_threshold: float = 0.40   # NCC gap needed to declare a winner


class AdvancedImageAligner:

    def __init__(self, config: RegistrationConfig = None):
        self.config = config or RegistrationConfig()

    def register_images(self, reference_img: np.ndarray,
                        test_img: np.ndarray) -> Dict:
        ref_gray  = self._preprocess(reference_img)
        test_gray = self._preprocess(test_img)
        return self._orientation_vote(ref_gray, test_gray)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
               if len(image.shape) == 3 else image.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    # ── Signal 1: dominant edge direction per strip zone ─────────────────
    def _dominant_edge_angle(self, gray: np.ndarray) -> float:
        """
        Return the dominant gradient orientation (0–180°) of the chip.
        Uses only the 4 border strips where pin rows live (same STRIP_FRAC
        as PinHealthChecker).  The mode of the gradient-orientation histogram
        (weighted by magnitude) gives the pin-row direction reliably.
        """
        h, w    = gray.shape[:2]
        sw_h    = max(6, int(h * 0.20))
        sw_w    = max(6, int(w * 0.20))

        strips = [
            gray[0:sw_h,      0:w],       # TOP
            gray[h-sw_h:h,    0:w],       # BOTTOM
            gray[0:h,         0:sw_w],    # LEFT
            gray[0:h,         w-sw_w:w],  # RIGHT
        ]

        all_mag = []
        all_ang = []
        for s in strips:
            gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx*gx + gy*gy)
            ang = (np.degrees(np.arctan2(np.abs(gy), np.abs(gx)))) % 180.0
            all_mag.append(mag.ravel())
            all_ang.append(ang.ravel())

        mag_all = np.concatenate(all_mag)
        ang_all = np.concatenate(all_ang)

        # Weighted histogram — bins every 5°
        hist, edges = np.histogram(ang_all, bins=36, range=(0, 180),
                                   weights=mag_all)
        peak_bin = int(np.argmax(hist))
        return float((edges[peak_bin] + edges[peak_bin + 1]) / 2.0)

    # ── Signal 2: 4-way NCC voter on centre-body crop ────────────────────
    def _body_ncc_voter(self, ref_gray: np.ndarray,
                        test_gray: np.ndarray) -> Dict:
        """
        Multi-scale NCC majority vote on the centre 60% crop of the chip.
        The IC body (text, dot, chamfer) is NOT 4-fold symmetric, making this
        crop much more discriminative than the full chip for 0 vs 180° and
        90 vs 270°.

        Returns dict with keys: success, rotation, confidence, message.
        """
        h, w  = ref_gray.shape[:2]
        cy0   = int(h * 0.20);  cy1 = int(h * 0.80)
        cx0   = int(w * 0.20);  cx1 = int(w * 0.80)
        ref_c = ref_gray[cy0:cy1, cx0:cx1]
        tst_c = test_gray[cy0:cy1, cx0:cx1]

        def _ncc(a, b):
            a = a.astype(np.float32) - a.mean()
            b = b.astype(np.float32) - b.mean()
            denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b)) + 1e-8
            return float(np.sum(a * b) / denom)

        angles = [0, 90, 180, 270]
        votes  = {a: 0 for a in angles}
        scores_128 = {}

        for sz in [64, 96, 128]:
            r = cv2.resize(ref_c, (sz, sz))
            t = cv2.resize(tst_c, (sz, sz))
            sc = {0: _ncc(r, t)}
            for k, deg in ((1, 90), (2, 180), (3, 270)):
                sc[deg] = _ncc(r, np.rot90(t, k))
            winner = max(sc, key=sc.__getitem__)
            votes[winner] += 1
            if sz == 128:
                scores_128 = sc

        best_deg   = max(votes, key=votes.__getitem__)
        n_votes    = votes[best_deg]
        best_score = scores_128.get(best_deg, 0.0)
        sv         = sorted(scores_128.values(), reverse=True)
        ncc_gap    = sv[0] - sv[1] if len(sv) >= 2 else 0.0

        if n_votes < 2:
            return {'success': False, 'rotation': 0.0, 'confidence': 0.0,
                    'message': f'body-NCC: no majority  votes={votes}'}
        if ncc_gap < self.config.confidence_threshold:
            return {'success': False, 'rotation': 0.0, 'confidence': 0.0,
                    'message': f'body-NCC: ambiguous gap={ncc_gap:.3f}'}

        rot = float(best_deg if best_deg <= 180 else best_deg - 360)
        return {'success': True, 'rotation': rot,
                'confidence': float(ncc_gap),
                'message': f'body-NCC votes={votes} best={best_deg}° gap={ncc_gap:.3f}'}

    # ── Combined orientation decision ─────────────────────────────────────
    def _orientation_vote(self, ref_gray: np.ndarray,
                          test_gray: np.ndarray) -> Dict:
        """
        Combine dominant-edge angle and body-NCC voter.

        Edge-angle gives the raw angle difference between ref and test.
        Body-NCC voter gives the quadrant (0/90/180/270).
        They must agree on the nearest 90° quadrant; if not → report 0°.
        """
        # Signal 1: edge angle difference
        ref_angle  = self._dominant_edge_angle(ref_gray)
        tst_angle  = self._dominant_edge_angle(test_gray)
        raw_diff   = tst_angle - ref_angle
        # Normalise to (−90, +90] — edge orientation is 0–180° so ambiguity
        # is ±90°; the NCC voter resolves the 0/180 ambiguity.
        while raw_diff >  90.0: raw_diff -= 180.0
        while raw_diff <= -90.0: raw_diff += 180.0
        edge_quad = int(round(raw_diff / 90.0)) * 90  # nearest 0 or ±90

        # Signal 2: body NCC voter
        ncc_res  = self._body_ncc_voter(ref_gray, test_gray)
        ncc_rot  = float(ncc_res['rotation'])
        ncc_quad = int(round(ncc_rot / 90.0)) * 90

        print(f"  [Orient] edge_diff={raw_diff:+.1f}° edge_quad={edge_quad}°"
              f"  ncc_rot={ncc_rot:+.0f}° ncc_quad={ncc_quad}°"
              f"  ncc_ok={ncc_res['success']}  gap={ncc_res['confidence']:.3f}")

        # Agreement check
        if not ncc_res['success']:
            # NCC inconclusive — trust edge angle alone only if it's clearly
            # non-zero (>30°); otherwise assume correct.
            if abs(raw_diff) > 30.0:
                final_rot = float(edge_quad)
                msg = f'NCC inconclusive, edge-only rot={final_rot}'
            else:
                final_rot = 0.0
                msg = 'NCC inconclusive, edge small → assumed correct'
        elif edge_quad == ncc_quad:
            # Both agree → high confidence
            final_rot = float(ncc_quad)
            msg = f'edge+NCC agree rot={final_rot}'
        else:
            # Disagreement → do not flag wrong orientation
            final_rot = 0.0
            msg = (f'edge({edge_quad}°) vs NCC({ncc_quad}°) disagree → assumed correct')

        return {
            'success':     True,
            'rotation':    final_rot,
            'confidence':  ncc_res['confidence'],
            'num_matches': 0,
            'num_inliers': 0,
            'message':     msg,
        }

    @staticmethod
    def _fail(message: str) -> Dict:
        return {'success': False, 'rotation': 0.0, 'confidence': 0.0,
                'num_matches': 0, 'num_inliers': 0, 'message': message}


# ─────────────────────────────────────────────────────────────────────────────
#  IC Detector  (YOLO best.pt)
# ─────────────────────────────────────────────────────────────────────────────
class ICDetector:
    def __init__(self, model_path: str = YOLO_PATH, conf: float = CONF_THRESHOLD):
        self.conf        = conf
        print("model confidence: ", conf)
        self._model      = None
        self._model_path = model_path

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")
        mp = Path(self._model_path)
        if not mp.exists():
            raise FileNotFoundError(f"YOLO model not found: {mp.resolve()}")
        self._model = YOLO(str(mp))
        print(f"[YOLO] Loaded model: {mp}")
        return self._model

    def detect(self, image: np.ndarray) -> List[Dict]:
        model   = self._get_model()
        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb, conf=self.conf, verbose=False)
        dets    = []
        for r in results:
            for b in r.boxes:
                dets.append({'box':   [float(v) for v in b.xyxy[0].tolist()],
                             'score': float(b.conf[0]),
                             'class': 'IC_chip'})
        print(f"[YOLO] Detected {len(dets)} IC_chip(s)  (conf>={self.conf})")
        return dets


# ─────────────────────────────────────────────────────────────────────────────
#  Pin Health Checker — direct peak-matching on gradient projection
#
#  Per-strip pipeline:
#    1. CLAHE grayscale
#    2. Upscale 4x → Sobel magnitude → 1-D mean projection ⊥ to pin rows
#    3. Gaussian smooth (sigma = SMOOTH_SIGMA)
#    4. find_peaks on the REFERENCE projection
#         prominence ≥ PROM_FRAC × projection range
#         distance   ≥ estimated pitch × 0.5  (pitch from autocorrelation)
#    5. For each reference peak, sample the TEST projection at the same X:
#         ratio = test_height / ref_height
#         ratio < MISS_RATIO  → MISSING
#         ratio < BENT_RATIO  → BENT
#         else                → healthy
#    6. Bridge: valley between consecutive ref peaks is
#         > BRIDGE_RATIO × ref_valley AND > 50% of ref peak mean
#    7. side_score = healthy_pins / total_ref_pins
#
#  health_score = mean(4 side scores)
# ─────────────────────────────────────────────────────────────────────────────
class PinHealthChecker:

    STRIP_FRAC   = 0.20    # border strip as fraction of chip dimension
    UPSAMPLE     = 4       # upscale before gradient projection
    CLAHE_CLIP   = 3.0
    CLAHE_TILE   = (4, 4)
    SMOOTH_SIGMA = 1.5     # Gaussian sigma on upscaled projection

    # Peak detection
    PROM_FRAC = 0.08       # prominence >= this × projection range
    MIN_PEAKS = 2          # skip analysis if fewer ref peaks found

    # Defect classification
    MISS_RATIO   = 0.35    # test/ref peak ratio below this → MISSING
    BENT_RATIO   = 0.65    # test/ref peak ratio below this → BENT
    BRIDGE_RATIO = 1.40    # test valley / ref valley above this → BRIDGED

    def check(self, test_chip: np.ndarray,
              ref_chip: np.ndarray = None) -> dict:
        if ref_chip  is None or ref_chip.size  == 0: return self._empty()
        if test_chip is None or test_chip.size == 0: return self._empty()
        if test_chip.shape[0] < 20 or test_chip.shape[1] < 20: return self._empty()

        if ref_chip.shape != test_chip.shape:
            test_chip = cv2.resize(test_chip,
                                   (ref_chip.shape[1], ref_chip.shape[0]),
                                   interpolation=cv2.INTER_AREA)

        ref_g  = self._clahe_gray(ref_chip)
        test_g = self._clahe_gray(test_chip)

        h, w  = ref_g.shape[:2]
        sw_h  = max(6, int(h * self.STRIP_FRAC))
        sw_w  = max(6, int(w * self.STRIP_FRAC))

        strips = {
            "TOP":    (ref_g[0:sw_h,       0:w],      test_g[0:sw_h,       0:w],      1),
            "BOTTOM": (ref_g[h-sw_h:h,     0:w],      test_g[h-sw_h:h,     0:w],      1),
            "LEFT":   (ref_g[0:h,          0:sw_w],   test_g[0:h,          0:sw_w],   0),
            "RIGHT":  (ref_g[0:h,          w-sw_w:w], test_g[0:h,          w-sw_w:w], 0),
        }

        side_scores  = {}
        side_details = {}
        n_miss = n_bent = n_bridge = 0

        for side, (rs, ts, axis) in strips.items():
            d = self._analyse_strip(rs, ts, axis)
            side_scores[side]  = d["score"]
            side_details[side] = d
            n_miss   += d["missing"]
            n_bent   += d["bent"]
            n_bridge += d["bridged"]

        health = float(np.mean(list(side_scores.values())))
        total  = n_miss + n_bent + n_bridge

        print("  [PinHealth] " +
              "  ".join(f"{s}:{v:.2f}" for s, v in side_scores.items()) +
              f"  overall:{health:.2f}")
        for s, d in side_details.items():
            if d["missing"] or d["bent"] or d["bridged"]:
                print(f"    [{s}] miss={d['missing']} bent={d['bent']}"
                      f" bridge={d['bridged']}  n_pins={d['n_ref_pins']}"
                      f"  score={d['score']:.2f}")

        return {
            "health_score": health,  "ssim_score": health,  "is_ic": True,
            "total_pins":   total,   "missing":    n_miss,  "bent":  n_bent,
            "bridged":      n_bridge,
            "ok":           1 if total == 0 else 0,
            "defect_pixels": total,
            "side_scores":  side_scores,
            "side_details": side_details,
        }

    # ─────────────────────────────────────────────────────────────────────
    def _analyse_strip(self, ref_strip: np.ndarray,
                       test_strip: np.ndarray, axis: int) -> dict:
        empty = {"score": 1.0, "missing": 0, "bent": 0, "bridged": 0,
                 "n_ref_pins": 0, "fft": 1.0, "hog": 1.0, "ncc": 1.0,
                 "ssim": 1.0, "grad": 1.0}

        rp = self._gradient_projection(ref_strip,  axis)
        tp = self._gradient_projection(test_strip, axis)
        if rp is None or tp is None:
            return empty

        from scipy.signal import find_peaks

        rng = float(rp.max() - rp.min())
        if rng < 5.0:
            return empty   # flat strip — no pins on this side

        # ── Estimate pin pitch via autocorrelation ────────────────────────
        rp_mc = rp - rp.mean()
        acorr = np.correlate(rp_mc, rp_mc, mode='full')[len(rp_mc) - 1:]
        acorr = acorr / (acorr[0] + 1e-8)
        from scipy.signal import find_peaks as _fp
        pitch_cands, _ = _fp(acorr[1:], height=0.10)
        pitch = int(pitch_cands[0]) + 1 if len(pitch_cands) > 0 else max(4, len(rp) // 12)

        min_dist = max(3, int(pitch * 0.5))
        prom     = max(3.0, rng * self.PROM_FRAC)

        ref_peaks, _ = find_peaks(rp, prominence=prom, distance=min_dist)
        if len(ref_peaks) < self.MIN_PEAKS:
            return empty

        # ── Per-peak defect classification ───────────────────────────────
        n_miss = n_bent = healthy = 0
        for pk in ref_peaks:
            rh    = float(rp[pk])
            th    = float(tp[pk])
            ratio = th / (rh + 1e-6)
            if   ratio < self.MISS_RATIO: n_miss  += 1
            elif ratio < self.BENT_RATIO: n_bent  += 1
            else:                         healthy  += 1

        # ── Bridge detection ──────────────────────────────────────────────
        n_bridge = 0
        for k in range(len(ref_peaks) - 1):
            pa, pb       = ref_peaks[k], ref_peaks[k + 1]
            ref_valley   = float(rp[pa:pb + 1].min())
            tst_valley   = float(tp[pa:pb + 1].min())
            ref_pk_mean  = float((rp[pa] + rp[pb]) / 2.0)
            if (ref_pk_mean > 5.0
                    and ref_valley > 1e-6
                    and tst_valley > ref_valley * self.BRIDGE_RATIO
                    and tst_valley > ref_pk_mean * 0.50):
                n_bridge += 1

        n_ref = len(ref_peaks)
        score = float(healthy) / float(n_ref)

        return {
            "score":      float(np.clip(score, 0.0, 1.0)),
            "missing":    n_miss,
            "bent":       n_bent,
            "bridged":    n_bridge,
            "n_ref_pins": n_ref,
            "fft": score, "hog": score, "ncc": score,
            "ssim": score, "grad": score,
        }

    # ─────────────────────────────────────────────────────────────────────
    def _gradient_projection(self, strip: np.ndarray, axis: int):
        """Upscale → Sobel magnitude → mean projection → Gaussian smooth."""
        from scipy.ndimage import gaussian_filter1d
        if strip.size == 0 or strip.shape[0] < 3 or strip.shape[1] < 3:
            return None
        us  = self.UPSAMPLE
        sup = cv2.resize(strip,
                         (strip.shape[1] * us, strip.shape[0] * us),
                         interpolation=cv2.INTER_CUBIC).astype(np.float32)
        gx  = cv2.Sobel(sup, cv2.CV_32F, 1, 0, ksize=3)
        gy  = cv2.Sobel(sup, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        proj = mag.mean(axis=axis)
        return gaussian_filter1d(proj, sigma=self.SMOOTH_SIGMA * us)

    def _clahe_gray(self, img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        return cv2.createCLAHE(clipLimit=self.CLAHE_CLIP,
                               tileGridSize=self.CLAHE_TILE).apply(g)

    @staticmethod
    def _empty() -> dict:
        return {
            "health_score": 1.0,  "ssim_score": 1.0,  "is_ic": False,
            "total_pins":   0,    "missing":    0,     "bent":  0,
            "bridged":      0,    "ok":         0,     "defect_pixels": 0,
            "side_scores":  {"TOP": 1.0, "BOTTOM": 1.0, "LEFT": 1.0, "RIGHT": 1.0},
            "side_details": {},
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Polarity Detector
# ─────────────────────────────────────────────────────────────────────────────
class PolarityDetector:
    POLAR_CLASSES = {'capacitor', 'electrolytic', 'diode', 'led',
                     'transistor', 'ic', 'connector'}

    def __init__(self, angle_threshold: float = 30.0):
        self.angle_threshold = angle_threshold
        self.aligner = AdvancedImageAligner()

    def detect_polarity(self, component_image: np.ndarray,
                        component_class: str) -> dict:
        has_polarity = any(pc in component_class.lower()
                           for pc in self.POLAR_CLASSES)
        return {'has_polarity': has_polarity,
                'chip': component_image,
                'component_class': component_class}

    def compare_polarity(self, ref_polarity: dict, test_polarity: dict,
                         angle_threshold: float = None) -> dict:
        threshold = angle_threshold if angle_threshold is not None \
                    else self.angle_threshold
        if not ref_polarity.get('has_polarity') or \
           not test_polarity.get('has_polarity'):
            return {'correct': True, 'angle_difference': 0.0,
                    'confidence': 1.0, 'num_matches': 0,
                    'method': 'skipped_non_polar',
                    'message': 'Component has no polarity — check skipped'}

        align = self.aligner.register_images(ref_polarity['chip'],
                                             test_polarity['chip'])
        if not align['success']:
            return {'correct': True, 'angle_difference': 0.0,
                    'confidence': 0.0, 'num_matches': 0,
                    'method': 'registration_failed',
                    'message': align['message']}

        raw = float(align['rotation'])
        while raw >  180.0: raw -= 360.0
        while raw <= -180.0: raw += 360.0
        is_wrong = abs(raw) > threshold
        method = 'body_ncc_voter'
        print(f"  [Polarity] class={ref_polarity.get('component_class', '?')}"
              f"  rotation={raw:+.1f}°  conf={align['confidence']:.2f}"
              f"  → {'WRONG' if is_wrong else 'OK'}")
        return {'correct': not is_wrong, 'angle_difference': raw,
                'confidence': float(align['confidence']),
                'num_matches': align['num_matches'],
                'method': method, 'message': align['message']}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def hungarian_assignment(score_matrix: np.ndarray) -> List[Tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(-score_matrix)
        return list(zip(ri.tolist(), ci.tolist()))
    except ImportError:
        used: set = set(); pairs = []
        for i in np.argsort(-score_matrix.max(axis=1)):
            row = score_matrix[i].copy(); row[list(used)] = -1
            j = int(np.argmax(row))
            if row[j] >= 0: pairs.append((int(i), j)); used.add(j)
        return pairs


def box_iou_with_margin(a, b, margin=40):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1 - margin), max(ay1, by1 - margin)
    ix2, iy2 = min(ax2, bx2 + margin), min(ay2, by2 + margin)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    cont = (ix2 - ix1) * (iy2 - iy1) / area_a if ix2 > ix1 and iy2 > iy1 else 0.0
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    diag = float(np.sqrt((bx2 - bx1) ** 2 + (by2 - by1) ** 2)) + 1e-6
    prox = float(np.exp(-0.5 * (float(np.sqrt((acx - bcx) ** 2 +
                                               (acy - bcy) ** 2)) / (diag / 2)) ** 2))
    return float(np.clip(0.7 * cont + 0.3 * prox, 0.0, 1.0))


def ncc_similarity(chip1, chip2):
    if chip1 is None or chip2 is None or chip1.size == 0 or chip2.size == 0:
        return 0.0
    g1 = cv2.cvtColor(chip1, cv2.COLOR_BGR2GRAY) if len(chip1.shape) == 3 else chip1
    g2 = cv2.cvtColor(chip2, cv2.COLOR_BGR2GRAY) if len(chip2.shape) == 3 else chip2
    g1 = cv2.createCLAHE(2.0, (4, 4)).apply(cv2.resize(g1, (96, 96))).astype(np.float32)
    g2 = cv2.createCLAHE(2.0, (4, 4)).apply(cv2.resize(g2, (96, 96))).astype(np.float32)
    g1 -= g1.mean(); g2 -= g2.mean()
    denom = float(np.linalg.norm(g1)) * float(np.linalg.norm(g2)) + 1e-8
    return float(np.clip((float(np.sum(g1 * g2)) / denom + 1.0) / 2.0, 0.0, 1.0))


def crop_chip(image, box, size=128):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    chip = image[y1:y2, x1:x2]
    if chip.size == 0: return np.zeros((size, size, 3), dtype=np.uint8)
    chip = cv2.resize(chip, (size, size))
    lab = cv2.cvtColor(chip, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(2.0, (4, 4)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def crop_chip_native(image, box) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    chip = image[y1:y2, x1:x2]
    if chip.size == 0: return np.zeros((64, 64, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(chip, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(chip, 1.5, blurred, -0.5, 0)


# ─────────────────────────────────────────────────────────────────────────────
#  IC Inspector
# ─────────────────────────────────────────────────────────────────────────────
PIN_STRIP_CFG = {
    "strip_frac": 0.18, "upsample": 3, "proj_smooth_sigma": 1.5,
    "min_proj_amp": 5.0, "peak_prominence": 0.10,
    "missing_thresh": 0.45, "bent_thresh": 0.65, "bridge_fill": 0.60,
    "side_ncc_fail": 0.60, "clahe_clip": 3.0, "clahe_tile": (4, 4),
    "ecc_iterations": 80, "ecc_epsilon": 1e-3,
}


class ICInspector:
    def __init__(self, yolo_path=YOLO_PATH, conf_threshold=CONF_THRESHOLD, roi=None):
        self.detector   = ICDetector(model_path=yolo_path, conf=conf_threshold)
        self.aligner    = AdvancedImageAligner()
        self.pin_health = PinHealthChecker()
        self.roi        = roi
        self.reference_image          = None
        self.reference_detections     = None
        self.ref_chips_cache          = []
        self.ref_chips_native_cache   = []
        self.ref_from_file            = False
        self.conf                     = conf_threshold
        print("model confidence: ", conf_threshold)

    def _filter_by_roi(self, dets):
        if self.roi is None: return dets
        rx1, ry1, rx2, ry2 = self.roi
        kept = [d for d in dets
                if rx1 <= (d['box'][0] + d['box'][2]) / 2 <= rx2
                and ry1 <= (d['box'][1] + d['box'][3]) / 2 <= ry2]
        if len(dets) - len(kept):
            print(f"[ROI] Filtered {len(dets) - len(kept)} outside")
        return kept

    def _roi_large_enough_for_pin_health(self) -> bool:
        if self.roi is None: return True
        rx1, ry1, rx2, ry2 = self.roi
        roi_w, roi_h = rx2 - rx1, ry2 - ry1
        print("roi_w: ", roi_w, "roi_h: ", roi_h)
        ok = (roi_w >= MIN_ROI_PIN_HEALTH_PX and roi_h >= MIN_ROI_PIN_HEALTH_PX)
        if not ok:
            print(f"[PinHealth] SKIPPED — ROI too small"
                  f" ({roi_w}x{roi_h} px < {MIN_ROI_PIN_HEALTH_PX} px)")
        return ok

    def set_reference(self, image: np.ndarray):
        print("\nProcessing reference image...")
        dets = self._filter_by_roi(self.detector.detect(image))
        if not dets:
            print("Warning: No IC components detected inside ROI in reference.")
            return
        ref_det         = dets[0]
        ref_chip        = crop_chip(image, ref_det['box'])
        ref_chip_native = crop_chip_native(image, ref_det['box'])
        self.reference_image        = ref_chip
        self.reference_detections   = dets
        self.ref_chips_cache        = [ref_chip]
        self.ref_chips_native_cache = [ref_chip_native]
        self.ref_from_file          = False
        Path('ref').mkdir(exist_ok=True)
        cv2.imwrite('ref/input_frame.png', image)
        cv2.imwrite('ref/chip_0.png', ref_chip)
        cv2.imwrite('ref/chip_0_native.png', ref_chip_native)
        roi_vis = image.copy()
        x1, y1, x2, y2 = map(int, ref_det['box'])
        cv2.rectangle(roi_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(roi_vis, 'REF', (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.roi:
            rx1, ry1, rx2, ry2 = self.roi
            cv2.rectangle(roi_vis, (rx1, ry1), (rx2, ry2), (255, 165, 0), 2)
        cv2.imwrite('ref/roi_frame.png', roi_vis)
        print('Saved: ref/input_frame.png  ref/chip_0.png  '
              'ref/chip_0_native.png  ref/roi_frame.png')
        print(f'Reference set with {len(dets)} IC component(s) inside ROI')

    def load_reference_from_file(self, ref_chip_path: str):
        print(f"\nLoading reference chip from: {ref_chip_path}")
        chip = cv2.imread(ref_chip_path)
        if chip is None:
            print(f"Error: Cannot read reference chip from '{ref_chip_path}'")
            return
        chip = crop_chip(chip, [0, 0, chip.shape[1], chip.shape[0]])
        native_path = str(ref_chip_path).replace('chip_0.png', 'chip_0_native.png')
        ni = cv2.imread(native_path)
        if ni is not None:
            native_chip = crop_chip_native(ni, [0, 0, ni.shape[1], ni.shape[0]])
            print(f'  Native ref loaded: {native_path}')
        else:
            native_chip = crop_chip_native(chip, [0, 0, chip.shape[1], chip.shape[0]])
            print('  Native ref not found, using 128x128 fallback')
        self.reference_image        = chip
        self.ref_chips_cache        = [chip]
        self.ref_chips_native_cache = [native_chip]
        self.reference_detections   = [{'box': [0, 0, chip.shape[1], chip.shape[0]],
                                         'score': 1.0, 'class': 'IC_chip'}]
        self.ref_from_file = True
        print('Reference chip loaded — YOLO/ROI check skipped')

    def detect_and_compare(self, test_image: np.ndarray, save_images=False) -> dict:
        if self.reference_image is None:
            return {'status': 'error', 'message': 'No reference image set'}

        test_dets = self._filter_by_roi(self.detector.detect(test_image))
        if not test_dets:
            missing = [{'ref_id': i, 'box': d['box'], 'status': 'absent'}
                       for i, d in enumerate(self.reference_detections)]
            return {'status': 'success', 'num_components': 0, 'components': [],
                    'missing_components': missing,
                    'comparison': {'matched': 0, 'extra': 0,
                                   'missing': len(self.reference_detections),
                                   'wrong_orientation': 0, 'pin_failures': 0}}

        test_chips        = [crop_chip(test_image, d['box']) for d in test_dets]
        test_chips_native = [crop_chip_native(test_image, d['box']) for d in test_dets]
        N_test, N_ref = len(test_dets), len(self.reference_detections)

        score_matrix = np.zeros((N_test, N_ref), dtype=np.float32)
        iou_matrix   = np.zeros((N_test, N_ref), dtype=np.float32)
        test_centres = [((d['box'][0] + d['box'][2]) / 2,
                         (d['box'][1] + d['box'][3]) / 2) for d in test_dets]
        ref_centres  = [((d['box'][0] + d['box'][2]) / 2,
                         (d['box'][1] + d['box'][3]) / 2) for d in self.reference_detections]

        for i in range(N_test):
            tcx, tcy = test_centres[i]
            for j in range(N_ref):
                rcx, rcy = ref_centres[j]
                if not self.ref_from_file:
                    if float(np.sqrt((tcx - rcx) ** 2 +
                                     (tcy - rcy) ** 2)) > HARD_SPATIAL_RADIUS_PX:
                        continue
                sim = ncc_similarity(test_chips[i], self.ref_chips_cache[j])
                iou = box_iou_with_margin(test_dets[i]['box'],
                                          self.reference_detections[j]['box'])
                iou_matrix[i, j]   = iou
                score_matrix[i, j] = 0.65 * sim + 0.35 * iou

        assignment_map: Dict[int, int] = {}
        for ti, rj in hungarian_assignment(score_matrix):
            if score_matrix[ti, rj] < SCORE_THRESHOLD: continue
            if not self.ref_from_file:
                tcx, tcy = test_centres[ti]; rcx, rcy = ref_centres[rj]
                if float(np.sqrt((tcx - rcx) ** 2 +
                                 (tcy - rcy) ** 2)) > HARD_SPATIAL_RADIUS_PX:
                    continue
            assignment_map[ti] = rj

        components_info = []

        for i in range(N_test):
            ref_idx  = assignment_map.get(i, -1)
            best_sim = ncc_similarity(test_chips[i],
                                      self.ref_chips_cache[ref_idx]) if ref_idx >= 0 else 0.0

            # ── Orientation ───────────────────────────────────────────────
            ic_orient = {
                'correct': True, 'rotation_deg': 0.0, 'rotation_step': 0,
                'confidence': 0.0, 'num_matches': 0, 'reason': 'not_checked'
            }
            if ref_idx >= 0:
                ref_c = self.ref_chips_cache[ref_idx]
                tst_c = test_chips[i]
                align = self.aligner.register_images(ref_c, tst_c)
                raw   = float(align['rotation'])
                while raw >  180.0: raw -= 360.0
                while raw <= -180.0: raw += 360.0
                is_wrong = abs(raw) > ORIENT_THRESHOLD_DEG
                _sr      = int(round(raw / 90.0)) * 90
                rot_step = _sr if _sr != -180 else 180
                ic_orient = {
                    'correct':         not is_wrong,
                    'rotation_deg':    raw,
                    'rotation_step':   rot_step,
                    'confidence':      float(align['confidence']),
                    'num_matches':     align['num_matches'],
                    'reason':          'ok' if not is_wrong else f'rotated_{rot_step}deg',
                }
                print(f"  [Orient] comp_id={i} rot={raw:+.1f}° step={rot_step}°"
                      f"  conf={align['confidence']:.3f}"
                      f"  → {'WRONG' if is_wrong else 'OK'}"
                      f"  [{align['message']}]")

            ic_orient_wrong = (
                not ic_orient['correct']
                and ic_orient['reason'] != 'not_checked'
            )

            # ── Pin health ────────────────────────────────────────────────
            chip_sim_ok  = best_sim >= PIN_HEALTH_MIN_CHIP_SIM
            run_pin_health = (
                ref_idx >= 0
                and self._roi_large_enough_for_pin_health()
                and not ic_orient_wrong
                and chip_sim_ok
            )
            if run_pin_health:
                if ref_idx < len(self.ref_chips_native_cache):
                    ref_native = self.ref_chips_native_cache[ref_idx]
                else:
                    c = self.ref_chips_cache[ref_idx]
                    ref_native = crop_chip_native(c, [0, 0, c.shape[1], c.shape[0]])
                pin = self.pin_health.check(test_chips_native[i], ref_native)
            else:
                if ic_orient_wrong:
                    print(f"  [PinHealth] SKIPPED — orientation wrong"
                          f" ({ic_orient['rotation_deg']:+.1f}°)")
                elif not chip_sim_ok:
                    print(f"  [PinHealth] SKIPPED — chip sim too low"
                          f" ({best_sim:.3f} < {PIN_HEALTH_MIN_CHIP_SIM})")
                pin = self.pin_health._empty()

            _ss     = pin.get('side_scores', {})
            _worst  = min(_ss.values()) if _ss else 1.0
            _n_def  = pin.get('missing', 0) + pin.get('bent', 0) + pin.get('bridged', 0)
            pin_ok  = (
                (pin.get('health_score', 1.0) >= PIN_HEALTH_WARN
                 and _worst >= PIN_HEALTH_SIDE_WARN)
                or _n_def == 0
            )

            status = ('wrong_orientation' if ic_orient_wrong else 'present') \
                     if ref_idx >= 0 else 'extra'

            components_info.append({
                'id': i, 'box': test_dets[i]['box'],
                'score': test_dets[i]['score'],
                'class': test_dets[i].get('class', 'IC_chip'),
                'similarity': best_sim,
                'iou_score': float(iou_matrix[i, ref_idx]) if ref_idx >= 0 else 0.0,
                'status': status, 'matched_ref_id': ref_idx if ref_idx >= 0 else None,
                'ic_orient_correct':    ic_orient['correct'],
                'ic_orient_deg':        ic_orient['rotation_deg'],
                'ic_orient_step':       ic_orient['rotation_step'],
                'ic_orient_confidence': ic_orient['confidence'],
                'ic_orient_matches':    ic_orient['num_matches'],
                'ic_orient_reason':     ic_orient['reason'],
                'pin_health_score': pin.get('health_score', 1.0),
                'pin_health_is_ic':  pin.get('is_ic', False),
                'pin_health_ok':     pin_ok,
                'pin_side_scores':   pin.get('side_scores', {}),
                'pin_total':   pin.get('total_pins', 0),
                'pin_missing': pin.get('missing', 0),
                'pin_bent':    pin.get('bent', 0),
                'pin_bridged': pin.get('bridged', 0),
                'pin_ok':      pin.get('ok', 0),
            })

        matched_ref = set(assignment_map.values())
        missing_components = [
            {'ref_id': j, 'box': self.reference_detections[j]['box'], 'status': 'absent'}
            for j in range(N_ref) if j not in matched_ref
        ]

        present_count  = sum(1 for c in components_info if c['status'] == 'present')
        wrong_o_count  = sum(1 for c in components_info if c['status'] == 'wrong_orientation')
        extra_count    = sum(1 for c in components_info if c['status'] == 'extra')
        pin_fail_count = sum(1 for c in components_info
                             if c['pin_health_is_ic'] and not c['pin_health_ok'])

        try:
            from pathlib import Path as _Path
            import time as _time
            _Path('output').mkdir(exist_ok=True)
            _vis = self._create_annotated_image(test_image, components_info, missing_components)
            if self.roi:
                _rx1, _ry1, _rx2, _ry2 = self.roi
                cv2.rectangle(_vis, (_rx1, _ry1), (_rx2, _ry2), (255, 165, 0), 2)
                cv2.putText(_vis, 'ROI', (_rx1, max(0, _ry1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            _ts = _time.strftime('%Y%m%d_%H%M%S')
            _out_path = f'output/result_{_ts}.png'
            cv2.imwrite(_out_path, _vis)
            print(f'[Output] Saved annotated result: {_out_path}')
        except Exception as _e:
            print(f'[Output] Warning: could not save result image: {_e}')

        return {
            'status': 'success', 'num_components': len(components_info),
            'components': components_info, 'missing_components': missing_components,
            'comparison': {'matched': present_count, 'wrong_orientation': wrong_o_count,
                           'extra': extra_count, 'missing': len(missing_components),
                           'pin_failures': pin_fail_count},
        }

    def _create_annotated_image(self, image, components, missing):
        vis = image.copy()
        colours = {'present': (0, 220, 0), 'wrong_orientation': (0, 165, 255),
                   'extra': (0, 0, 255)}
        for c in components:
            x1, y1, x2, y2 = map(int, c['box'])
            if c.get('pin_health_is_ic') and not c.get('pin_health_ok', True):
                col = (0, 200, 255)
            else:
                col = colours.get(c['status'], (200, 200, 200))
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            label = c['status'].upper()
            if c['status'] == 'wrong_orientation':
                label += f" {c.get('ic_orient_deg', 0.0):+.0f}deg"
            if c.get('pin_health_is_ic'):
                ph_ok  = c.get('pin_health_ok', True)
                miss   = c.get('pin_missing', 0)
                bent   = c.get('pin_bent', 0)
                bridge = c.get('pin_bridged', 0)
                label += (f" PIN_FAIL miss={miss} bent={bent} brdg={bridge}"
                          if not ph_ok else " PIN_OK")
            ly = y1 - 6 if y1 > 16 else y2 + 14
            cv2.putText(vis, label, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)
        for m in missing:
            x1, y1, x2, y2 = map(int, m['box'])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (128, 0, 128), 2)
            cv2.putText(vis, 'ABSENT', (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 0, 128), 1)
        return vis


# ─────────────────────────────────────────────────────────────────────────────
#  Interactive CLI app
# ─────────────────────────────────────────────────────────────────────────────
class ICInspectionApp:
    def __init__(self, yolo_path=YOLO_PATH, conf_threshold=CONF_THRESHOLD, roi=None):
        self.conf      = conf_threshold
        print("model confidence: ", conf_threshold)
        self.inspector = ICInspector(yolo_path, conf_threshold, roi)
        self._last_result = None; self._last_image = None

    def run_interactive(self):
        print("\nIC Inspection — orientation via edge-direction + body-NCC voter")
        print("Commands: ref <img>  |  test <img>  |  load_ref <chip>  |  quit\n")
        while True:
            try:    raw = input(">> ").strip()
            except (EOFError, KeyboardInterrupt): break
            if not raw: continue
            cmd = raw.split(); sub = cmd[0].lower()
            if sub in ('quit', 'q', 'exit'): break
            elif sub == 'ref':
                if len(cmd) < 2: print("Usage: ref <image_path>"); continue
                img = cv2.imread(cmd[1])
                if img is None: print(f"Cannot read: {cmd[1]}"); continue
                self.inspector.set_reference(img)
            elif sub == 'load_ref':
                if len(cmd) < 2: print("Usage: load_ref <chip_path>"); continue
                self.inspector.load_reference_from_file(cmd[1])
            elif sub == 'test':
                if len(cmd) < 2: print("Usage: test <image_path>"); continue
                img = cv2.imread(cmd[1])
                if img is None: print(f"Cannot read: {cmd[1]}"); continue
                self._last_image  = img
                self._last_result = self.inspector.detect_and_compare(img)
                self._display_results(self._last_result)
            elif sub == 'teach': self._handle_teach(cmd)
            else: print(f"Unknown command: {sub}")

    def _handle_teach(self, cmd):
        if len(cmd) < 2: print("Usage: teach <image_path>"); return
        img = cv2.imread(cmd[1])
        if img is None: print(f"Cannot read: {cmd[1]}"); return
        self.inspector.set_reference(img)
        print(f"Reference updated from: {cmd[1]}")

    def _display_results(self, result):
        if result['status'] != 'success':
            print(f"Error: {result.get('message', '?')}")
            print("\n[RESULT_BEGIN]")
            print("IC Present (matched): 0"); print("IC Absent  (missing): 1")
            print("IC Extra   (unexpected): 0"); print("Wrong orientation: 0")
            print("Pin health failures: 0"); print("[RESULT_END]")
            return

        comps = result.get('components', [])
        miss  = result.get('missing_components', [])
        cmp   = result.get('comparison', {})

        overall_pass = (cmp.get('wrong_orientation', 0) == 0
                        and cmp.get('missing', 0) == 0
                        and cmp.get('extra', 0) == 0
                        and cmp.get('pin_failures', 0) == 0)

        print("\n" + "=" * 60)
        print(f"INSPECTION RESULT:  {'✓ PASS' if overall_pass else '✗ FAIL'}")
        print("=" * 60)
        print(f"  IC Present (matched):     {cmp.get('matched', 0)}"
              f"  /  {len(self.inspector.reference_detections or [])}")
        print(f"  IC Absent  (missing):     {cmp.get('missing', 0)}")
        print(f"  IC Extra   (unexpected):  {cmp.get('extra', 0)}")
        print(f"  Wrong orientation:        {cmp.get('wrong_orientation', 0)}")
        print(f"  Pin health failures:      {cmp.get('pin_failures', 0)}")
        print("=" * 60)

        sym = {'present': '✓', 'wrong_orientation': '⟳', 'extra': '+', 'absent': '✗'}
        if comps:
            print("\nComponent Details:")
            for c in comps:
                s  = c['status']
                ln = (f"  {sym.get(s, '?')} [{s.upper()}]"
                      f"  id={c['id']}"
                      f"  sim={c['similarity']:.3f}"
                      f"  iou={c.get('iou_score', 0):.2f}")
                deg  = c.get('ic_orient_deg', 0.0)
                conf = c.get('ic_orient_confidence', 0.0)
                nm   = c.get('ic_orient_matches', 0)
                if not c.get('ic_orient_correct', True):
                    ln += (f"\n      Orientation: WRONG"
                           f"  rotation={deg:+.1f}°"
                           f"  step~{c['ic_orient_step']}°"
                           f"  conf={conf:.2f}")
                else:
                    reason = c.get('ic_orient_reason', '')
                    if 'aligner_failed' in reason:
                        ln += f"\n      Orientation: NOT CHECKED  ({reason})"
                    else:
                        ln += (f"\n      Orientation: OK"
                               f"  rotation={deg:+.1f}°"
                               f"  conf={conf:.2f}")
                if c.get('pin_health_is_ic'):
                    ok = c.get('pin_health_ok', True)
                    ln += (f"\n      Pin health:  {'OK' if ok else 'FAIL'}"
                           f"  score={c['pin_health_score']:.0%}"
                           f"  total={c['pin_total']}"
                           f"  miss={c['pin_missing']}"
                           f"  bent={c['pin_bent']}"
                           f"  bridge={c['pin_bridged']}")
                else:
                    ln += "\n      Pin health:  (not an IC package)"
                print(ln)
        if miss:
            print("\nAbsent ICs:")
            for m in miss:
                print(f"  ✗ [ABSENT]  ref_id={m['ref_id']}"
                      f"  box={[round(v) for v in m['box']]}")
        print()

        total_pin_missing = sum(c['pin_missing'] for c in comps if c.get('pin_health_is_ic'))
        total_pin_bent    = sum(c['pin_bent']    for c in comps if c.get('pin_health_is_ic'))
        total_pin_bridge  = sum(c['pin_bridged'] for c in comps if c.get('pin_health_is_ic'))

        worst_orient = None
        for c in comps:
            if not c.get('ic_orient_correct', True):
                if worst_orient is None or abs(c['ic_orient_deg']) > abs(worst_orient['ic_orient_deg']):
                    worst_orient = c
        if worst_orient is None and comps:
            worst_orient = comps[0]

        rotation_deg  = worst_orient['ic_orient_deg']  if worst_orient else 0.0
        rotation_step = worst_orient['ic_orient_step'] if worst_orient else 0

        print("[RESULT_BEGIN]")
        print(f"IC Present (matched): {cmp.get('matched', 0)}")
        print(f"IC Absent  (missing): {cmp.get('missing', 0)}")
        print(f"IC Extra   (unexpected): {cmp.get('extra', 0)}")
        print(f"Wrong orientation: {cmp.get('wrong_orientation', 0)}")
        print(f"Pin health failures: {cmp.get('pin_failures', 0)}")
        print(f"pin_missing={total_pin_missing}")
        print(f"pin_bent={total_pin_bent}")
        print(f"pin_bridge={total_pin_bridge}")
        print(f"rotation={rotation_deg:.2f} step~{rotation_step}")
        print("[RESULT_END]")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    for d in ('output', 'tmp', 'images', 'ref'): Path(d).mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description='IC Inspection')
    parser.add_argument('--yolo-path',      default=YOLO_PATH)
    parser.add_argument('--conf-threshold', type=float, default=CONF_THRESHOLD)
    parser.add_argument('--mode',           default='interactive',
                        choices=['interactive', 'check_roi'])
    parser.add_argument('--image',          default=None)
    parser.add_argument('--roi-coords',     default=None)
    parser.add_argument('--roi-file',       default=None)
    args = parser.parse_args()

    def _load_roi(path):
        if not path: return None
        try:
            vals = open(path).read().split()
            return (int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]))
        except Exception as e:
            print(f"[ROI] Cannot read '{path}': {e}"); return None

    if args.mode == 'check_roi':
        if not args.image or not args.roi_coords:
            print("IC_ABSENT"); raise SystemExit(1)
        img = cv2.imread(args.image)
        if img is None: print("IC_ABSENT"); raise SystemExit(1)
        try:
            vals = args.roi_coords.split()
            if len(vals) != 4: raise ValueError(f"Expected 4 values, got {len(vals)}")
        except Exception as e:
            print(f"IC_ABSENT  # invalid --roi-coords: {e}"); raise SystemExit(1)
        rx1, ry1 = max(0, int(vals[0])), max(0, int(vals[1]))
        rx2, ry2 = min(img.shape[1], int(vals[2])), min(img.shape[0], int(vals[3]))
        try:
            dets  = ICDetector(model_path=args.yolo_path,
                               conf=args.conf_threshold).detect(img)
            found = any(rx1 <= (d['box'][0] + d['box'][2]) // 2 <= rx2
                        and ry1 <= (d['box'][1] + d['box'][3]) // 2 <= ry2
                        for d in dets)
            print("IC_PRESENT" if found else "IC_ABSENT")
        except Exception as e:
            print(f"IC_ABSENT  # error: {e}")
        raise SystemExit(0)

    roi = _load_roi(args.roi_file)
    if roi: print(f"[ROI] x1={roi[0]} y1={roi[1]} x2={roi[2]} y2={roi[3]}")
    try:
        ICInspectionApp(yolo_path=args.yolo_path,
                        conf_threshold=args.conf_threshold,
                        roi=roi).run_interactive()
    except Exception as _e:
        import traceback as _tb
        print(f'[FATAL] {_e}', flush=True); _tb.print_exc()
        print('[RESULT_BEGIN]')
        print('IC Present (matched): 0'); print('IC Absent  (missing): 1')
        print('IC Extra   (unexpected): 0'); print('Wrong orientation: 0')
        print('Pin health failures: 0'); print('pin_missing=0')
        print('pin_bent=0'); print('pin_bridge=0')
        print('rotation=0.00 step~0'); print('[RESULT_END]')