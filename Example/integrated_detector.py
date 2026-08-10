import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

warnings.filterwarnings('ignore')
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

_builtin_print = print


def print(*args, **kwargs): 
    try:
        _builtin_print(*args, **kwargs)
    except UnicodeEncodeError:
        stream = kwargs.get('file', sys.stdout)
        enc = getattr(stream, 'encoding', None) or 'ascii'
        safe_args = [str(a).encode(enc, errors='replace').decode(enc) for a in args]
        _builtin_print(*safe_args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
SCORE_THRESHOLD         = 0.30
HARD_SPATIAL_RADIUS_PX  = 150.0
MIN_ROI_PIN_HEALTH_PX   = 200
PIN_HEALTH_MIN_CHIP_SIM = 0.50
ROI_DETECT_MARGIN_PX    = 150   
ROI_MIN_OVERLAP_RATIO   = 0.30  

# ── Global feature toggles ──────────────────────────────────────────────

#-------------------------------Presence/absence---------------------------

RFDETR_PATH             = r'checkpoint_best_ema.pth'
YOLO_PATH               = RFDETR_PATH   
CONF_THRESHOLD          = 0.70
ENABLE_NO_TEXT_ABSENT_CHECK = True
SAVE_NO_TEXT_ABSENT_PREPROCESSING = False   
NO_TEXT_MIN_MARK_FRAC       = 0.01     # fallback, used only if no reference frac is available
REF_FRAC_THRESHOLD_PCT      = 0.60      # dynamic threshold = REF_FRAC_THRESHOLD_PCT * ref_image_mark_frac

#------------------Orientation Related--------------------------------------
ENABLE_ORIENTATION_CHECK = True  
ORIENT_THRESHOLD_DEG    = 30.0   
DOT_MARKER_MIN_CONF      = 0.40 

ENABLE_SECONDARY_BOUNDARY_VERIFICATION = True   
SECONDARY_BOUNDARY_VERIFY_DEG          =5.0  

SAVE_IC_PREPROCESSING    = False      
SAVE_SECONDARY_BOUNDARY_PREPROCESSING  = False  

ENABLE_AREA_ROI_CHECK    = False
AREA_RATIO_TOLERANCE     = 0.10   
ROI_CONTAINMENT_MIN_FRAC = 0.90   


#------------------------------Pin health check ----------------------------------
MIN_IC_AREA_PIN_HEALTH_PX2 = 100000  
PIN_HEALTH_WARN         = 0.60   
PIN_HEALTH_SIDE_WARN    = 0.00 
ENABLE_PIN_HEALTH_CHECK  = False                                  
SAVE_PIN_HEALTH_PREPROCESSING = False   
   
#-----------------------------------------------------------------------
ENABLE_ORB_FALLBACK      = True    
ORB_MIN_MATCHES          = 10     
ORB_MIN_INLIERS          = 8       
ENABLE_BOUNDARY_ANGLE_CROSSCHECK = False            
BOUNDARY_ANGLE_CROSSCHECK_DEG    = 5  
ENABLE_EDGE_SCAN_CANDIDATE = True
EDGE_SCAN_CORNER_MARGIN_PX = 5   # trims this many points off each end of every                            
ENABLE_CORNER_NOTCH_STRIP    = True
CORNER_NOTCH_KERNEL_PX       = 31   # must be odd; larger removes bigger notches
                                    
@dataclass
class OrientationResult:
    success: bool
    rotation_deg: float = 0.0
    num_matches: int = 0
    num_inliers: int = 0
    message: str = ""


class ORBHomographyOrientation:

    def __init__(self,
                 n_features: int = 2000,
                 lowe_ratio: float = 0.75,
                 ransac_reproj_thresh: float = 5.0,
                 min_matches: int = ORB_MIN_MATCHES,
                 min_inliers: int = ORB_MIN_INLIERS):
        self.orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8,
                                   edgeThreshold=15, fastThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.lowe_ratio = lowe_ratio
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        den = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=5)
        return self._clahe.apply(den)

    def _match(self, des_ref: np.ndarray, des_test: np.ndarray):
        knn = self.bf.knnMatch(des_ref, des_test, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.lowe_ratio * n.distance:
                good.append(m)
        return good

    @staticmethod
    def _rotation_from_homography(H: np.ndarray) -> float:
        A = H[:2, :2].astype(np.float64)
        U, _, Vt = np.linalg.svd(A)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    def _template_rotation_fallback(self, ref_gray: np.ndarray, test_gray: np.ndarray):
        h, w = ref_gray.shape[:2]
        if h < 12 or w < 12:
            return None
        test_rs = cv2.resize(test_gray, (w, h), interpolation=cv2.INTER_AREA)

        def edge_map(g):
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        ref_edges = edge_map(ref_gray)
        best_angle, best_score = None, -1.0
        for angle in (0.0, 90.0, 180.0, 270.0):
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            rot = cv2.warpAffine(test_rs, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
            rot_edges = edge_map(rot)
            score = float(cv2.matchTemplate(ref_edges, rot_edges, cv2.TM_CCOEFF_NORMED)[0, 0])
            if score > best_score:
                best_score, best_angle = score, angle
        if best_angle is None or best_score < 0.25:
            return None
        return best_angle, best_score

    def estimate(self, reference_img: np.ndarray,
                 test_crop_img: np.ndarray) -> OrientationResult:
        ref_gray  = self._to_gray(reference_img)
        test_gray = self._to_gray(test_crop_img)

        ref_enh  = self._preprocess(ref_gray)
        test_enh = self._preprocess(test_gray)

        kp_ref,  des_ref  = self.orb.detectAndCompute(ref_enh, None)
        kp_test, des_test = self.orb.detectAndCompute(test_enh, None)

        if (des_ref is not None and des_test is not None
                and kp_ref is not None and kp_test is not None
                and len(kp_ref) >= 4 and len(kp_test) >= 4):
            good = self._match(des_ref, des_test)
            if len(good) >= self.min_matches:
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_test[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                              self.ransac_reproj_thresh)
                if H is not None:
                    num_inliers = int(mask.sum()) if mask is not None else 0
                    if num_inliers >= self.min_inliers:
                        rotation_deg = self._rotation_from_homography(H)
                        return OrientationResult(success=True, rotation_deg=rotation_deg,
                            num_matches=len(good), num_inliers=num_inliers,
                            message=f"ORB-homography rotation={rotation_deg:+.1f}° "
                                    f"(matches={len(good)}, inliers={num_inliers})")

        fallback = self._template_rotation_fallback(ref_enh, test_enh)
        if fallback is not None:
            angle, score = fallback
            return OrientationResult(success=True, rotation_deg=angle,
                num_matches=0, num_inliers=0,
                message=f"template-match rotation={angle:+.1f}° (edge-corr={score:.2f}, "
                        f"ORB fallback)")

        n_ref  = len(kp_ref)  if kp_ref  else 0
        n_test = len(kp_test) if kp_test else 0
        return OrientationResult(success=False,
            message=f"insufficient signal for orientation "
                    f"(ORB kp ref={n_ref} test={n_test}; template-match also inconclusive)")


class AdvancedImageAligner:

    def register_images(self, reference_img: np.ndarray,
                        test_img: np.ndarray) -> Dict:
        checker  = DotOrientationChecker(save_intermediate=SAVE_IC_PREPROCESSING)
        ref_res  = checker.check_orientation(reference_img, tag="ref")
        test_res = checker.check_orientation(test_img, tag="test")
        result = self._compare(ref_res, test_res)

        if result['success'] or not ENABLE_ORB_FALLBACK:
            return result

        orb_res = ORBHomographyOrientation().estimate(reference_img, test_img)
        if not orb_res.success:
            return AdvancedImageAligner._fail(
                f"{result['message']}; ORB fallback also failed ({orb_res.message})")

        return {'success': True, 'rotation': orb_res.rotation_deg,
                'confidence': min(1.0, orb_res.num_inliers / max(orb_res.num_matches, 1)),
                'num_matches': orb_res.num_matches, 'num_inliers': orb_res.num_inliers,
                'message': f"dot-check failed ({result['message']}); "
                           f"used ORB fallback: {orb_res.message}"}

    @staticmethod
    def _compare(ref_res: Dict, test_res: Dict) -> Dict:
        if ref_res.get('angle_deg') is None or test_res.get('angle_deg') is None:
            return AdvancedImageAligner._fail(
                f"dot not found (ref_conf={ref_res.get('dot_confidence', 0.0):.2f} "
                f"test_conf={test_res.get('dot_confidence', 0.0):.2f})")

        conf = min(float(ref_res['dot_confidence']), float(test_res['dot_confidence']))
        if conf < DOT_MARKER_MIN_CONF:
            return AdvancedImageAligner._fail(
                f"dot low-confidence (ref={ref_res['dot_confidence']:.2f} "
                f"test={test_res['dot_confidence']:.2f})")

        rot = float(test_res['angle_deg']) - float(ref_res['angle_deg'])
        while rot >  180.0: rot -= 360.0
        while rot <= -180.0: rot += 360.0

        result = {'success': True, 'rotation': rot, 'confidence': float(conf),
                'num_matches': 0, 'num_inliers': 0,
                'message': f"dot-angle ref={ref_res['angle_deg']:+.1f}° "
                           f"test={test_res['angle_deg']:+.1f}° rot={rot:+.1f}°"}

        if ENABLE_BOUNDARY_ANGLE_CROSSCHECK:
            rb_ref  = ref_res.get('raw_boundary_angle_deg')
            rb_test = test_res.get('raw_boundary_angle_deg')
            if rb_ref is not None and rb_test is not None:
                b_diff = rb_test - rb_ref
                while b_diff >  45.0: b_diff -= 90.0
                while b_diff <= -45.0: b_diff += 90.0
                if abs(b_diff) > BOUNDARY_ANGLE_CROSSCHECK_DEG and abs(b_diff) > abs(rot):
                    result['rotation'] = b_diff
                    result['message'] += (f"; boundary cross-check overrides: "
                                           f"raw_boundary rot={b_diff:+.1f}° "
                                           f"(ref={rb_ref:+.1f}° test={rb_test:+.1f}°)")

        return result

    @staticmethod
    def _fail(message: str) -> Dict:
        return {'success': False, 'rotation': 0.0, 'confidence': 0.0,
                'num_matches': 0, 'num_inliers': 0, 'message': message}



@dataclass
class BoundaryVerificationResult:
    verified:       bool             # True if a usable outer-boundary rect was
                                      # found on BOTH the reference and test crops
    passed:         bool             # True if the check passed (or was skipped --
                                      # this stage fails OPEN when unverifiable)
    angle_diff_deg: float = 0.0      # normalised to 0-90 deg
    ref_angle_deg:  float = None
    test_angle_deg: float = None
    ref_rect:       Tuple = None     # raw cv2.minAreaRect() of the reference crop
    test_rect:      Tuple = None     # raw cv2.minAreaRect() of the test crop
    message:        str = ""


class ICBoundaryVerifier:
    def __init__(self,
                 canny_lo: float = None, canny_hi: float = None,
                 canny_hi_percentile: float = 85.0, canny_lo_ratio: float = 0.5,
                 morph_kernel: int = 7, morph_iterations: int = 3,
                 min_area_ratio: float = 0.15, max_aspect_ratio: float = 6.0,
                 min_solidity: float = 0.75,
                 max_border_touch_frac: float = 0.20,
                 min_border_edge_support: float = 0.30,
                 min_interior_area_ratio: float = 0.35,
                 min_component_span_frac: float = 0.30,
                 text_strip_dilate_k: int = 3,
                 text_strip_border_margin_frac: float = 0.06,
                 out_dir: str = "output/ic_boundary_verify",
                 save_intermediate: bool = SAVE_SECONDARY_BOUNDARY_PREPROCESSING):
        self.canny_lo             = canny_lo
        self.canny_hi              = canny_hi
        self.canny_hi_percentile   = canny_hi_percentile
        self.canny_lo_ratio        = canny_lo_ratio
        self.morph_kernel     = morph_kernel
        self.morph_iterations = morph_iterations
        self.min_area_ratio   = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_solidity     = min_solidity
        self.max_border_touch_frac  = max_border_touch_frac
        self.min_border_edge_support = min_border_edge_support
        self.min_interior_area_ratio = min_interior_area_ratio
        self.min_component_span_frac = min_component_span_frac
        self.text_strip_dilate_k     = text_strip_dilate_k
        self.text_strip_border_margin_frac = text_strip_border_margin_frac
        self.out_dir           = Path(out_dir)
        self.save_intermediate = save_intermediate
        if self.save_intermediate:
            self.out_dir.mkdir(parents=True, exist_ok=True)


        self.no_text_out_dir = Path("output/no_text_absent")
        if SAVE_NO_TEXT_ABSENT_PREPROCESSING:
            self.no_text_out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, name: str, img: np.ndarray) -> None:
        """imwrite helper -- only writes when save_intermediate is on, mirroring
        DotOrientationChecker's own _save() so both stages behave the same way."""
        if self.save_intermediate and img is not None and img.size:
            cv2.imwrite(str(self.out_dir / name), img)

    def _save_no_text(self, name: str, img: np.ndarray) -> None:
        if SAVE_NO_TEXT_ABSENT_PREPROCESSING and img is not None and img.size:
            cv2.imwrite(str(self.no_text_out_dir / name), img)

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img is not None and img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def _solidity(c) -> float:
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return 0.0
        return float(cv2.contourArea(c)) / hull_area

    @staticmethod
    def _border_touch_fraction(c, h: int, w: int, band: int = 2) -> float:
        pts = c.reshape(-1, 2)
        on_border = ((pts[:, 0] <= band) | (pts[:, 0] >= w - 1 - band) |
                     (pts[:, 1] <= band) | (pts[:, 1] >= h - 1 - band))
        return float(on_border.mean())

    @staticmethod
    def _border_edge_support(c, raw_edges: np.ndarray, h: int, w: int,
                              band: int = 2, tol: int = 2) -> float:
        pts = c.reshape(-1, 2)
        on_border = ((pts[:, 0] <= band) | (pts[:, 0] >= w - 1 - band) |
                     (pts[:, 1] <= band) | (pts[:, 1] >= h - 1 - band))
        border_pts = pts[on_border]
        if len(border_pts) == 0:
            return 1.0
        k = 2 * tol + 1
        dilated = cv2.dilate(raw_edges, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        supported = dilated[border_pts[:, 1], border_pts[:, 0]] > 0
        return float(supported.mean())

    def _strip_interior_text_edges(self, edges: np.ndarray) -> np.ndarray:
        h, w = edges.shape[:2]
        k = max(1, self.text_strip_dilate_k)
        dil = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)

        margin = int(self.text_strip_border_margin_frac * max(h, w))
        kept = np.zeros_like(edges)
        edge_mask = edges > 0
        for i in range(1, n):   # label 0 is background
            x, y, cw, ch, _area = stats[i]
            near_border = (x <= margin or y <= margin or
                           x + cw >= w - margin or y + ch >= h - margin)

            if near_border:
                comp_mask = (labels == i) & edge_mask   # keep original edge
                kept[comp_mask] = 255                   # pixels, not the dilation halo
        return kept

    def _adaptive_canny_thresholds(self, gray: np.ndarray) -> Tuple[float, float]:
        gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        hi = float(np.percentile(mag, self.canny_hi_percentile))
        hi = max(hi, 10.0)   # floor so a near-flat crop doesn't collapse to 0/0
        lo = hi * self.canny_lo_ratio
        return lo, hi

    def compute_mark_frac(self, crop: np.ndarray, tag: str = "crop") -> Optional[float]:
        """Compute the interior-marking pixel fraction for a crop.
        Returns None if the crop is invalid/too small (caller should treat
        this as "no reference frac available")."""
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 8 or w < 8:
            return None

        gray = self._to_gray(crop)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=40)

        if self.canny_lo is None or self.canny_hi is None:
            canny_lo, canny_hi = self._adaptive_canny_thresholds(gray)
        else:
            canny_lo, canny_hi = self.canny_lo, self.canny_hi
        edges = cv2.Canny(gray, int(canny_lo), int(canny_hi))
        edges_stripped = self._strip_interior_text_edges(edges)

        interior = cv2.bitwise_and(edges, cv2.bitwise_not(edges_stripped))
        frac = float(np.count_nonzero(interior)) / float(h * w)

        if self.save_intermediate:
            self._save(f"{tag}_text_interior_mask.png", interior)

        if SAVE_NO_TEXT_ABSENT_PREPROCESSING:
            self._save_no_text(f"{tag}_crop.png", crop)
            self._save_no_text(f"{tag}_edges.png", edges)
            self._save_no_text(f"{tag}_edges_stripped.png", edges_stripped)
            self._save_no_text(f"{tag}_interior_mask.png", interior)

        return frac

    def has_interior_text(self, crop: np.ndarray, tag: str = "crop",
                           threshold: Optional[float] = None) -> bool:
        """threshold: dynamic mark-fraction cutoff (e.g. REF_FRAC_THRESHOLD_PCT *
        reference frac). Falls back to NO_TEXT_MIN_MARK_FRAC if not provided."""
        frac = self.compute_mark_frac(crop, tag=tag)
        if frac is None:
            return False

        min_frac = threshold if threshold is not None else NO_TEXT_MIN_MARK_FRAC
        has_text = frac >= min_frac

        print(f"  [NoTextCheck] tag={tag}  interior_marking_frac={frac:.5f}"
              f"  (need >= {min_frac:.5f}"
              f"{'  [dynamic, ref-based]' if threshold is not None else '  [fallback constant]'})  -> "
              f"{'TEXT FOUND' if has_text else 'NO TEXT'}")
        return has_text

    @staticmethod
    def _strip_corner_notch(contour, shape: Tuple[int, int],
                             kernel_px: int = CORNER_NOTCH_KERNEL_PX):
        h, w = shape
        filled = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 255, thickness=-1)

        k = max(1, kernel_px | 1)   # force odd
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)

        cleaned_contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
        if not cleaned_contours:
            return None, None   # kernel too aggressive for this shape -- fail safe

        clean_contour = max(cleaned_contours, key=cv2.contourArea)
        if cv2.contourArea(clean_contour) <= 0:
            return None, None

        return cv2.minAreaRect(clean_contour), clean_contour

    @staticmethod
    def _boundary_scan_points(binary: np.ndarray,
                               margin: int = EDGE_SCAN_CORNER_MARGIN_PX) -> np.ndarray:
        h, w = binary.shape[:2]
        left_pts, right_pts, top_pts, bottom_pts = [], [], [], []

        for y in range(h):
            white = False
            for x in range(w):
                if binary[y, x] == 255:
                    white = True
                elif white:
                    left_pts.append((x, y))
                    break

        for y in range(h):
            white = False
            for x in range(w - 1, -1, -1):
                if binary[y, x] == 255:
                    white = True
                elif white:
                    right_pts.append((x, y))
                    break

        for x in range(w):
            white = False
            for y in range(h):
                if binary[y, x] == 255:
                    white = True
                elif white:
                    top_pts.append((x, y))
                    break

        for x in range(w):
            white = False
            for y in range(h - 1, -1, -1):
                if binary[y, x] == 255:
                    white = True
                elif white:
                    bottom_pts.append((x, y))
                    break

        stacks = []
        for raw in (left_pts, right_pts, top_pts, bottom_pts):
            arr = np.array(raw, dtype=np.int32)
            if len(arr) > 2 * margin:
                arr = arr[margin:-margin]
            if len(arr):
                stacks.append(arr.reshape(-1, 2))

        if not stacks:
            return np.empty((0, 2), dtype=np.int32)
        return np.vstack(stacks)

    @staticmethod
    def _contour_from_binary_mask(mask: np.ndarray):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        raw_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        if not raw_contours:
            return None
        filled = closed.copy()
        cv2.drawContours(filled, raw_contours, -1, 255, thickness=cv2.FILLED)

        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    @staticmethod
    def _red_edge_mask(img: np.ndarray) -> np.ndarray:
        if img.ndim != 3:
            raise ValueError("_red_edge_mask expects a BGR image")
        b, g, r = cv2.split(img)
        red = ((r > 120) & (g < 100) & (b < 100)).astype(np.uint8) * 255
        return red

    @classmethod
    def _contour_from_scan_points(cls, pts: np.ndarray, h: int, w: int):
        if pts is None or len(pts) < 3:
            return None

        mask = np.zeros((h, w), dtype=np.uint8)
        xs = np.clip(pts[:, 0], 0, w - 1)
        ys = np.clip(pts[:, 1], 0, h - 1)
        mask[ys, xs] = 255

        return cls._contour_from_binary_mask(mask)

    def _candidate_from_scan_points(self, pts: np.ndarray, img_area: float,
                                     h: int, w: int, raw_edges: np.ndarray = None):
        contour = self._contour_from_scan_points(pts, h, w)
        if contour is None:
            return None

        area = cv2.contourArea(contour)
        if area < self.min_area_ratio * img_area:
            return None

        rect = cv2.minAreaRect(contour)
        (_, _), (rw, rh), _ = rect
        if rw <= 0 or rh <= 0:
            return None
        aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
        if aspect > self.max_aspect_ratio:
            return None

        sol = self._solidity(contour)
        if sol < self.min_solidity:
            return None

        border_frac = self._border_touch_fraction(contour, h, w)
        if border_frac > self.max_border_touch_frac:
            support = (self._border_edge_support(contour, raw_edges, h, w)
                       if raw_edges is not None else 0.0)
            if support < self.min_border_edge_support:
                return None

        if border_frac < 0.10 and area < self.min_interior_area_ratio * img_area:
            return None
        if ENABLE_CORNER_NOTCH_STRIP:
            stripped_rect, stripped_contour = self._strip_corner_notch(contour, (h, w))
            if stripped_rect is not None:
                rect, contour = stripped_rect, stripped_contour
                area = cv2.contourArea(contour)
                sol  = self._solidity(contour)
                border_frac = self._border_touch_fraction(contour, h, w)

        return rect, sol, area, [contour], border_frac

    def _candidate_rect(self, closed: np.ndarray, img_area: float,
                         raw_edges: np.ndarray = None):
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        h, w = closed.shape[:2]
        best, best_area, best_sol, best_border_frac = None, 0.0, 0.0, 0.0
        best_contour = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area_ratio * img_area:
                continue
            rect = cv2.minAreaRect(c)
            (_, _), (rw, rh), _ = rect
            if rw <= 0 or rh <= 0:
                continue
            aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
            if aspect > self.max_aspect_ratio:
                continue   # too sliver-like to be the package body
            sol = self._solidity(c)
            if sol < self.min_solidity:
                continue  
            border_frac = self._border_touch_fraction(c, h, w)
            if border_frac > self.max_border_touch_frac:
                support = (self._border_edge_support(c, raw_edges, h, w)
                           if raw_edges is not None else 0.0)
                if support < self.min_border_edge_support:
                    continue   
            if border_frac < 0.10 and area < self.min_interior_area_ratio * img_area:
                continue

            if area > best_area:
                best_area, best, best_sol, best_border_frac = area, rect, sol, border_frac
                best_contour = c

        if best is None:
            return None


        if ENABLE_CORNER_NOTCH_STRIP:
            stripped_rect, stripped_contour = self._strip_corner_notch(best_contour, (h, w))
            if stripped_rect is not None:
                best, best_contour = stripped_rect, stripped_contour
                best_area = cv2.contourArea(stripped_contour)
                best_sol  = self._solidity(stripped_contour)
                best_border_frac = self._border_touch_fraction(stripped_contour, h, w)

        return best, best_sol, best_area, contours, best_border_frac

    def _detect_boundary_rect(self, crop: np.ndarray, tag: str = "crop"):
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 8 or w < 8:
            return None

        # 1. Grayscale
        gray = self._to_gray(crop)
        self._save(f"{tag}_01_gray.png", gray)

        # 2. Gaussian + bilateral filtering (robust, edge-preserving smoothing)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=40)
        self._save(f"{tag}_02_filtered.png", gray)

        # 3. Canny edge detection -- adaptive thresholds unless the caller
        if self.canny_lo is None or self.canny_hi is None:
            canny_lo, canny_hi = self._adaptive_canny_thresholds(gray)
        else:
            canny_lo, canny_hi = self.canny_lo, self.canny_hi
        edges = cv2.Canny(gray, int(canny_lo), int(canny_hi))
        self._save(f"{tag}_03_edges.png", edges)


        edges_stripped = self._strip_interior_text_edges(edges)
        self._save(f"{tag}_03b_text_stripped.png", edges_stripped)

        # 4. Morphological closing to connect broken edges
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.morph_kernel, self.morph_kernel))
        closed = cv2.morphologyEx(edges_stripped, cv2.MORPH_CLOSE, kernel,
                                   iterations=self.morph_iterations)
        self._save(f"{tag}_04_closed.png", closed)


        scan_pts = (self._boundary_scan_points(closed)
                    if ENABLE_EDGE_SCAN_CANDIDATE else np.empty((0, 2), dtype=np.int32))
        if self.save_intermediate and len(scan_pts):
            scan_vis = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
            for (px, py) in scan_pts:
                cv2.circle(scan_vis, (int(px), int(py)), 2, (0, 0, 255), -1)
            self._save(f"{tag}_04b_edge_scan.png", scan_vis)

        img_area = float(h * w)
        cand_normal   = self._candidate_rect(closed, img_area, raw_edges=edges)
        cand_inverted = self._candidate_rect(cv2.bitwise_not(closed), img_area, raw_edges=edges)
        cand_scan     = (self._candidate_from_scan_points(scan_pts, img_area, h, w, raw_edges=edges)
                          if ENABLE_EDGE_SCAN_CANDIDATE else None)
        named_candidates = [
            ('normal',   cand_normal),
            ('inverted', cand_inverted),
            ('scan',     cand_scan),
        ]
        scored = [(name, cand) for name, cand in named_candidates if cand is not None]

        if not scored:
            best = None
            contours_for_vis = []
        else:
            def _key(item):
                _, cand = item
                _, sol, area, _, bf = cand
                return (-bf, sol, area)

            _winner_name, winner = max(scored, key=_key)
            best, _, _, contours_for_vis, _ = winner

        if self.save_intermediate and best is not None:
            vis = crop.copy() if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(vis, contours_for_vis, -1, (255, 0, 0), 1)
            box = cv2.boxPoints(best).astype(np.int32)
            cv2.drawContours(vis, [box], 0, (0, 220, 0), 2)
            self._save(f"{tag}_05_boundary_rect.png", vis)

        return best  # 7. minAreaRect result: (center, (w, h), angle)

    def detect_rect_from_closed(self, closed: np.ndarray,
                                 raw_edges: np.ndarray = None):
        if closed is None or closed.size == 0:
            return None, None, None
        img_area = float(closed.shape[0] * closed.shape[1])

        cand_normal   = self._candidate_rect(closed, img_area, raw_edges=raw_edges)
        cand_inverted = self._candidate_rect(cv2.bitwise_not(closed), img_area,
                                              raw_edges=raw_edges)

        if cand_normal is None and cand_inverted is None:
            return None, None, None
        elif cand_normal is None:
            rect, polarity = cand_inverted[0], 'inverted'
        elif cand_inverted is None:
            rect, polarity = cand_normal[0], 'normal'
        else:
            rect_n, sol_n, area_n, _, bf_n = cand_normal
            rect_i, sol_i, area_i, _, bf_i = cand_inverted
            key_n = (-bf_n, sol_n, area_n)
            key_i = (-bf_i, sol_i, area_i)
            if key_i > key_n:
                rect, polarity = rect_i, 'inverted'
            else:
                rect, polarity = rect_n, 'normal'

        return rect, self._normalise_angle(rect), polarity

    def detect_rect_from_edge_scan_image(self, img: np.ndarray, tag: str = "edge_scan",
                                          apply_notch_strip: bool = True):
        if img is None or img.size == 0:
            return None, None, None
        h, w = img.shape[:2]

        red_mask = self._red_edge_mask(img)
        if not np.any(red_mask):
            return None, None, None

        contour = self._contour_from_binary_mask(red_mask)
        if contour is None:
            return None, None, None

        rect = cv2.minAreaRect(contour)

        if apply_notch_strip and ENABLE_CORNER_NOTCH_STRIP:
            stripped_rect, stripped_contour = self._strip_corner_notch(contour, (h, w))
            if stripped_rect is not None:
                rect, contour = stripped_rect, stripped_contour

        if self.save_intermediate:
            vis = img.copy()
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 1)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(vis, [box], 0, (255, 0, 255), 1)
            self._save(f"{tag}_04c_edge_scan_contour.png", vis)

        return rect, self._normalise_angle(rect), contour

    def detect_inner_cavity_rect(self, roi_gray: np.ndarray,
                                  thresh: int = None,
                                  morph_kernel: int = None,
                                  morph_iterations: int = None,
                                  min_area_ratio: float = None,
                                  tag: str = "cavity"):
        if roi_gray is None or roi_gray.size == 0:
            return None, None
        gray = self._to_gray(roi_gray)
        h, w = gray.shape[:2]
        img_area = float(h * w)

        k    = morph_kernel     if morph_kernel     is not None else self.morph_kernel
        iters = morph_iterations if morph_iterations is not None else self.morph_iterations
        min_area_ratio = min_area_ratio if min_area_ratio is not None else self.min_area_ratio

        # 1. Threshold the IC ROI to a binary silhouette.
        if thresh is None:
            _, binary = cv2.threshold(gray, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        self._save(f"{tag}_01_binary.png", binary)

        # 2. Morphological closing -- bridges pin gaps/noise so the body
        # forms one clean external contour.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iters)
        self._save(f"{tag}_02_closed.png", closed)

        # 3. Largest external contour = the IC body outline.
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        outer = max(contours, key=cv2.contourArea)
        if cv2.contourArea(outer) < min_area_ratio * img_area:
            return None, None   # nothing large enough to be the IC body

        # 4. Fill that contour solid.
        filled_mask = np.zeros_like(closed)
        cv2.drawContours(filled_mask, [outer], -1, 255, thickness=-1)
        self._save(f"{tag}_03_filled.png", filled_mask)

        # 5. holeMask = filledMask - binary. cv2.subtract (not plain `-`)
        # clips at 0 instead of wrapping, in case binary has stray
        # foreground pixels just outside the filled body.
        hole_mask = cv2.subtract(filled_mask, binary)
        self._save(f"{tag}_04_hole_mask.png", hole_mask)

        # 6. Largest contour in holeMask = the inner cavity/rectangle.
        hole_contours, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
        if not hole_contours:
            return None, None
        inner = max(hole_contours, key=cv2.contourArea)

        # 7. Fit minAreaRect to obtain the inner cavity/rectangle.
        rect = cv2.minAreaRect(inner)
        angle = self._normalise_angle(rect)

        if self.save_intermediate:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(vis, [outer], -1, (255, 0, 0), 1)
            cv2.drawContours(vis, [inner], -1, (0, 165, 255), 1)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(vis, [box], 0, (0, 220, 0), 2)
            self._save(f"{tag}_05_cavity_rect.png", vis)

        return rect, angle

    @staticmethod
    def _normalise_angle(rect) -> float:
        (_, _), (rw, rh), ang = rect
        if rw < rh:
            ang += 90.0
        while ang >= 45.0:
            ang -= 90.0
        while ang < -45.0:
            ang += 90.0
        return float(ang)

    def verify(self, ref_crop: np.ndarray, test_crop: np.ndarray) -> BoundaryVerificationResult:
        ref_rect  = self._detect_boundary_rect(ref_crop, tag="ref")
        test_rect = self._detect_boundary_rect(test_crop, tag="test")

        if ref_rect is None or test_rect is None:
            return BoundaryVerificationResult(
                verified=False, passed=True,
                message="secondary boundary verification skipped -- could not "
                        f"isolate a package contour (ref={'ok' if ref_rect is not None else 'none'}, "
                        f"test={'ok' if test_rect is not None else 'none'})")

        ref_angle  = self._normalise_angle(ref_rect)
        test_angle = self._normalise_angle(test_rect)


        diff = test_angle - ref_angle
        while diff > 90.0:  diff -= 180.0
        while diff <= -90.0: diff += 180.0
        diff = abs(diff)
        if diff > 90.0:
            diff = 180.0 - diff

        passed = diff <= SECONDARY_BOUNDARY_VERIFY_DEG

        return BoundaryVerificationResult(
            verified=True, passed=passed, angle_diff_deg=float(diff),
            ref_angle_deg=ref_angle, test_angle_deg=test_angle,
            ref_rect=ref_rect, test_rect=test_rect,
            message=(f"boundary-only rot: ref={ref_angle:+.1f}\u00b0 "
                     f"test={test_angle:+.1f}\u00b0 diff={diff:.1f}\u00b0 "
                     f"({'PASS' if passed else 'FAIL'} vs {SECONDARY_BOUNDARY_VERIFY_DEG}\u00b0 threshold)"))


class DotOrientationChecker:

    def __init__(self, out_dir: str = "output/ic_preprocessing",
                 save_intermediate: bool = True):
        self.out_dir = Path(out_dir)
        self.save_intermediate = save_intermediate
        if self.save_intermediate:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))

    def _save(self, name: str, img: np.ndarray) -> None:
        if self.save_intermediate and img is not None and img.size:
            cv2.imwrite(str(self.out_dir / name), img)
    EDGE_GRAD_PERCENTILE = 75.0   # "small threshold" -> keep top ~quarter of gradient mass
    RAMP_DIFF_MIN        = 10.0   # min intensity jump across a gradient-line pair to count as a real edge (Fig. 4)
    NEAR_SQUARE_ASPECT_RATIO = 1.08   # max(rw,rh)/min(rw,rh) below this -> too square to trust a fine rotation angle
    CORNER_TRUST_CIRCULARITY = 0.55   # a corner candidate this round is trusted outright over any side/notch candidate

    def _package_intensity_band(self, gray: np.ndarray):
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_s = cv2.GaussianBlur(hist.reshape(-1, 1), (1, 9), 0).flatten()
        Ip = int(np.argmax(hist_s[:160]))   # package peak, lower/darker half (Fig. 3)
        return max(0, Ip - 25), min(255, Ip + 25)

    def _boundary_points(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        # step 1: small threshold on the gradient image -> all ramp-edge candidates
        thr = np.percentile(mag, self.EDGE_GRAD_PERCENTILE)
        ys, xs = np.where(mag > max(thr, 1.0))
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.int32)

        lo, hi = self._package_intensity_band(gray)
        d = max(3, int(0.03 * min(h, w)))

        gxn, gyn = gx[ys, xs], gy[ys, xs]
        norm = np.hypot(gxn, gyn) + 1e-6
        ux, uy = gxn / norm, gyn / norm

        x1 = np.clip((xs + ux * d).astype(np.int32), 0, w - 1)
        y1 = np.clip((ys + uy * d).astype(np.int32), 0, h - 1)
        x2 = np.clip((xs - ux * d).astype(np.int32), 0, w - 1)
        y2 = np.clip((ys - uy * d).astype(np.int32), 0, h - 1)
        v1 = gray[y1, x1].astype(np.int32)
        v2 = gray[y2, x2].astype(np.int32)


        near_package = ((v1 >= lo) & (v1 <= hi)) | ((v2 >= lo) & (v2 <= hi))

        # step 3 (Fig. 4): the two points must actually differ enough in
        # intensity to represent a real ramp edge (package vs. background),
        # not just texture/markings sitting inside the package.
        is_ramp = np.abs(v1 - v2) >= self.RAMP_DIFF_MIN

        keep = near_package & is_ramp
        return np.stack([xs[keep], ys[keep]], axis=1)

    def _detect_package_boundary(self, gray: np.ndarray):
        """Returns cv2.minAreaRect()-style ((cx,cy),(w,h),angle) for the
        package body, or None if no plausible package boundary is found."""
        pts = self._boundary_points(gray)
        if len(pts) >= 20:
            rect = cv2.minAreaRect(pts.astype(np.float32))
            (_, _), (rw, rh), _ = rect
            if rw * rh >= 0.05 * gray.shape[0] * gray.shape[1]:
                return rect
        return self._detect_package_boundary_blob(gray)


    @staticmethod
    def _detect_package_boundary_blob(gray: np.ndarray):
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if mask.mean() > 255 * 0.5:
            mask = cv2.bitwise_not(mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 0.05 * gray.shape[0] * gray.shape[1]:
            return None
        return cv2.minAreaRect(c)


    def _align_package(self, gray: np.ndarray):
        rect = self._detect_package_boundary(gray)
        h, w = gray.shape[:2]
        if rect is None:
            return gray, 0.0   # fallback: assume already aligned

        (cx, cy), (rw, rh), angle = rect


        aspect = max(rw, rh) / max(1e-6, min(rw, rh))
        if aspect < self.NEAR_SQUARE_ASPECT_RATIO:
            s = int(max(rw, rh))
            x0 = max(0, int(cx - s / 2)); y0 = max(0, int(cy - s / 2))
            x1 = min(w, int(cx + s / 2)); y1 = min(h, int(cy + s / 2))
            if x1 - x0 >= 10 and y1 - y0 >= 10:
                return gray[y0:y1, x0:x1], 0.0
            return gray, 0.0

        # minAreaRect angle is only defined mod 90 (a rectangle looks the
        # same after a quarter turn); normalize to the smallest rotation
        # that makes the box's sides horizontal/vertical.
        theta = angle
        if rw < rh:
            theta = angle + 90.0
        while theta > 45.0:
            theta -= 90.0
        while theta <= -45.0:
            theta += 90.0

        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), theta, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

        rect2 = self._detect_package_boundary(rotated)
        if rect2 is None:
            return rotated, theta
        (cx2, cy2), (rw2, rh2), _ = rect2
        x0 = max(0, int(cx2 - rw2 / 2))
        y0 = max(0, int(cy2 - rh2 / 2))
        x1 = min(w, int(cx2 + rw2 / 2))
        y1 = min(h, int(cy2 + rh2 / 2))
        if x1 - x0 < 10 or y1 - y0 < 10:
            return rotated, theta
        return rotated[y0:y1, x0:x1], theta


    def _raw_boundary_rotation(self, gray: np.ndarray):
        rect = self._detect_package_boundary(gray)
        if rect is None:
            return None
        (_, _), (rw, rh), angle = rect
        theta = angle
        if rw < rh:
            theta += 90.0
        while theta > 45.0:
            theta -= 90.0
        while theta <= -45.0:
            theta += 90.0
        return float(theta)


    MARGIN_FRAC       = 0.08    # outer strip to skip (pin leads / edge glare)
    SQUARE_ASPECT_TOL = 0.10    # |w/h - 1| below this -> treat as "square" package

    def _corner_and_side_aois(self, pkg_gray: np.ndarray) -> List[Dict]:
        h, w = pkg_gray.shape[:2]
        short_side = min(h, w)
        is_square = abs(w / max(h, 1) - 1.0) <= self.SQUARE_ASPECT_TOL
        frac = 0.25 if is_square else (1.0 / 3.0)   # Sec. 2.3 sizing rule

        m = max(2, int(short_side * self.MARGIN_FRAC))
        s = max(10, int(short_side * frac))
        s = min(s, h - 2 * m, w - 2 * m)
        if s < 10:
            m, s = 0, min(h, w)

        aois = [
            # corners -> dimple candidates
            dict(name='TL', kind='corner', offset=0.0,   y0=m,             x0=m,             anchor=(0.0, 0.0), box=pkg_gray[m:m+s,             m:m+s]),
            dict(name='TR', kind='corner', offset=90.0,  y0=m,             x0=w-m-s,         anchor=(s, 0.0),   box=pkg_gray[m:m+s,             w-m-s:w-m]),
            dict(name='BR', kind='corner', offset=180.0, y0=h-m-s,         x0=w-m-s,         anchor=(s, s),     box=pkg_gray[h-m-s:h-m,         w-m-s:w-m]),
            dict(name='BL', kind='corner', offset=270.0, y0=h-m-s,         x0=m,             anchor=(0.0, s),   box=pkg_gray[h-m-s:h-m,         m:m+s]),
            # mid-sides -> notch candidates
            dict(name='TOP',   kind='side', offset=0.0,   y0=m,             x0=w//2-s//2, anchor=(s/2.0, 0.0), box=pkg_gray[m:m+s,             w//2-s//2:w//2-s//2+s]),
            dict(name='RIGHT', kind='side', offset=90.0,  y0=h//2-s//2,     x0=w-m-s,     anchor=(s, s/2.0),   box=pkg_gray[h//2-s//2:h//2-s//2+s, w-m-s:w-m]),
            dict(name='BOTTOM',kind='side', offset=180.0, y0=h-m-s,         x0=w//2-s//2, anchor=(s/2.0, s),   box=pkg_gray[h-m-s:h-m,         w//2-s//2:w//2-s//2+s]),
            dict(name='LEFT',  kind='side', offset=270.0, y0=h//2-s//2,     x0=m,         anchor=(0.0, s/2.0), box=pkg_gray[h//2-s//2:h//2-s//2+s, m:m+s]),
        ]
        return [a for a in aois if a['box'].size and min(a['box'].shape[:2]) >= 6]

    # ── P-tile threshold, per Eq. 3: find the intensity level Tn such that
    # the fraction Pn (%) of the histogram mass lies at or above Tn ────────
    @staticmethod
    def _p_tile_threshold(hist: np.ndarray, pct: float) -> int:
        pct = float(np.clip(pct, 0.1, 99.0))
        target = pct / 100.0 * hist.sum()
        cum = 0.0
        for level in range(255, -1, -1):
            cum += hist[level]
            if cum >= target:
                return level
        return 0


    def _adaptive_double_threshold(self, chan_u8: np.ndarray, kind: str = 'corner',
                                    raw_aoi: np.ndarray = None, anchor=None):
        h, w = chan_u8.shape[:2]
        hist = cv2.calcHist([chan_u8], [0], None, [256], [0, 256]).flatten()

        P1, P2 = 3.0, 10.0     # initial percentiles, per the paper's stated (3%, 10%)
        tau, Nc = 3.0, 5.0     # updating step size / gap-control constant, per Eq. 4
        N_prev = None

        for _ in range(6):     # bounded adaptive-threshold refinement loop
            Ta = self._p_tile_threshold(hist, P1)
            Tb = self._p_tile_threshold(hist, P2)
            T1, T2 = (Ta, Tb) if Ta >= Tb else (Tb, Ta)   # T1 = strict/high, T2 = loose/low

            R3 = (chan_u8 >= T1)
            R2 = (chan_u8 >= T2) & ~R3


            grown = R3.astype(np.uint8)
            for _ in range(4):
                dil = cv2.dilate(grown, np.ones((3, 3), np.uint8))
                added = (dil.astype(bool) & R2 & ~grown.astype(bool))
                if not added.any():
                    break
                grown[added] = 1

            mask = grown * 255

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best = self._best_notch_dimple(cnts, (h, w), kind=kind, raw_aoi=raw_aoi, anchor=anchor)
            if best is not None:
                return best

            # no valid feature yet -> update (P1, P2) per Eq. 4 and retry
            N_now = int(mask.sum() // 255)
            if N_prev is None:
                N_prev = max(N_now, 1)
            gaps = max(0, len(cnts) - 1)
            growth = N_prev / max(N_now, 1)
            P1 = P1 + tau * growth
            P2 = P2 + tau * growth * (1.0 + gaps / Nc)
            N_prev = N_now
            if P1 >= 60.0 or P2 >= 85.0:   # threshold reached its lower bound
                break

        return None

    def _enhance_aoi(self, aoi: np.ndarray) -> np.ndarray:
        den = cv2.bilateralFilter(aoi, d=5, sigmaColor=25, sigmaSpace=5)
        return self._clahe.apply(den)

    def _segment_edge(self, aoi: np.ndarray, kind: str, anchor=None):
        if aoi.size == 0 or min(aoi.shape[:2]) < 6:
            return None
        pre = self._enhance_aoi(aoi)
        gx = cv2.Sobel(pre, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(pre, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        grad_u8 = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        best = self._adaptive_double_threshold(grad_u8, kind=kind, raw_aoi=aoi, anchor=anchor)
        return best + ('edge',) if best is not None else None

    def _segment_intensity(self, aoi: np.ndarray, kind: str, anchor=None):
        if aoi.size == 0 or min(aoi.shape[:2]) < 6:
            return None
        pre = self._enhance_aoi(aoi)
        for src, extreme in ((pre, 'bright'), (255 - pre, 'dark')):
            best = self._adaptive_double_threshold(src, kind=kind, raw_aoi=aoi, anchor=anchor)
            if best is not None:
                return best + (f'intensity_{extreme}',)
        return None


    @staticmethod
    def _best_notch_dimple(contours, aoi_shape, kind: str = 'corner', raw_aoi: np.ndarray = None,
                            anchor=None):
        h, w = aoi_shape
        aoi_area = float(h * w)
        min_area = max(6.0, 0.010 * aoi_area)
        best, best_score = None, 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > 0.15 * aoi_area:
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circ = 4.0 * np.pi * area / (perim * perim)
            if circ < 0.30:      # too irregular to be a notch or dimple at all
                continue


            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 1e-6 else 0.0
            if solidity < 0.75:
                continue  
            (_, _), (rw_c, rh_c), _ = cv2.minAreaRect(c)
            elongation = max(rw_c, rh_c) / max(1e-6, min(rw_c, rh_c))
            if elongation > 1.8:
                continue   # too stroke-like/elongated to be a molded marker

            (cx, cy), r = cv2.minEnclosingCircle(c)

            standoff = max(1.0, 0.10 * r)
            if (cx < standoff or cx > w - standoff or
                    cy < standoff or cy > h - standoff):
                continue   # centre is essentially on the AOI boundary -- a cut-off fragment
            anchor_w = 1.0
            if anchor is not None:
                adist = float(np.hypot(cx - anchor[0], cy - anchor[1]))
                anchor_w = max(0.10, 1.0 - adist / (0.75 * max(w, h)))

            size_w = min(1.0, area / (0.03 * aoi_area))
            if kind == 'side':
                shape_w = max(0.0, 1.0 - abs(circ - 0.70) / 0.35)
                if circ < 0.30 or circ > 0.90:
                    continue
            else:
                shape_w = circ   # dimples: reward roundness directly

            contrast_w = 1.0
            edge_w = 1.0
            reflection_w = 1.0
            if raw_aoi is not None:
                yy, xx = np.mgrid[0:h, 0:w]
                rr = max(2.0, r)
                inner = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr * rr
                ring = ((xx - cx) ** 2 + (yy - cy) ** 2 <= (rr * 2.2) ** 2) & ~inner
                if inner.any() and ring.any():
                    inner_mean = float(raw_aoi[inner].mean())
                    ring_mean  = float(raw_aoi[ring].mean())
                    contrast_w = min(1.0, abs(inner_mean - ring_mean) / 25.0)   # <25 gray-level gap -> steeply discounted

                   
                    inner_std = float(raw_aoi[inner].std())
                    raggedness = inner_std / max(3.0, abs(inner_mean - ring_mean))
                    reflection_w = max(0.10, 1.0 - min(1.0, raggedness))


                    gx = cv2.Sobel(raw_aoi, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(raw_aoi, cv2.CV_32F, 0, 1, ksize=3)
                    mag = cv2.magnitude(gx, gy)
                    rim = ring & ((xx - cx) ** 2 + (yy - cy) ** 2 <= (rr * 1.6) ** 2)
                    if rim.any():
                        edge_w = min(1.0, float(mag[rim].mean()) / 40.0)

            quality_w = contrast_w * edge_w * reflection_w * anchor_w
            score = shape_w * size_w * quality_w * solidity
            if score > best_score:
                best_score = score
                best = (float(cx), float(cy), float(r), float(circ), float(area), float(quality_w))
        return best

    STAGE_CONF_WEIGHT = (1.00, 0.75, 0.55, 0.35)   # corner+edge, corner+intensity, side+edge, side+intensity

    def _find_dot(self, pkg_gray: np.ndarray, tag: str):
        aois = self._corner_and_side_aois(pkg_gray)
        for a in aois:
            self._save(f"{tag}_aoi_{a['name']}.png", a['box'])
        self._save(f"{tag}_dot_search_region.png",
                    aois[0]['box'] if aois else pkg_gray)

        corners = [a for a in aois if a['kind'] == 'corner']
        sides   = [a for a in aois if a['kind'] == 'side']
        stages = [(corners, self._segment_edge), (corners, self._segment_intensity),
                  (sides,   self._segment_edge), (sides,   self._segment_intensity)]

        for stage_idx, (aoi_list, segment_fn) in enumerate(stages):
            candidates = []
            for a in aoi_list:
                result = segment_fn(a['box'], a['kind'], a.get('anchor'))
                if result is None:
                    continue
                cx, cy, r, circ, area, contrast_w, seg_method = result
                candidates.append({
                    'name': a['name'], 'kind': a['kind'], 'offset': a['offset'],
                    'center_full': (a['x0'] + cx, a['y0'] + cy),
                    'radius': r, 'circularity': circ, 'area': area,
                    'contrast_w': contrast_w, 'seg_method': seg_method,
                })
            if not candidates:
                continue   # this whole stage found nothing anywhere -> try the next, less-trusted stage

            best = max(candidates, key=lambda c: c['circularity'] * min(1.0, c['area'] / 20.0) * c['contrast_w'])

            vis = cv2.cvtColor(pkg_gray, cv2.COLOR_GRAY2BGR)
            cv2.circle(vis, (int(best['center_full'][0]), int(best['center_full'][1])),
                       max(1, int(best['radius'])), (0, 255, 0), 2)
            self._save(f"{tag}_dot_detected.png", vis)

            confidence = float(min(1.0, best['circularity'] * best['contrast_w']
                                   * self.STAGE_CONF_WEIGHT[stage_idx]))
            return {'center_full': best['center_full'], 'radius': float(best['radius']),
                    'circularity': float(best['circularity']), 'confidence': confidence,
                    'offset': best['offset'],
                    'method': f"{best['kind']}_{best['name']}_{best['seg_method']}"}

        return None

    def check_orientation(self, image: np.ndarray, tag: str = "chip") -> Dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
               if image.ndim == 3 else image.copy()
        self._save(f"{tag}_gray.png", gray)
        raw_boundary = self._raw_boundary_rotation(gray)

        pkg_gray, theta = self._align_package(gray)
        self._save(f"{tag}_aligned.png", pkg_gray)

        dot = self._find_dot(pkg_gray, tag)

        if dot is None:
            result = {'dot_center': None, 'dot_confidence': 0.0,
                      'angle_deg': None, 'boundary_angle_deg': theta,
                      'raw_boundary_angle_deg': raw_boundary,
                      'method': 'not_found'}
        else:
            angle = theta + dot['offset']
            while angle > 180.0: angle -= 360.0
            while angle <= -180.0: angle += 360.0
            result = {'dot_center': dot['center_full'],
                      'dot_confidence': dot['confidence'],
                      'angle_deg': float(angle),
                      'boundary_angle_deg': float(theta),
                      'raw_boundary_angle_deg': raw_boundary,
                      'method': dot['method']}

        print(f"  [DotOrient] tag={tag}  dot={result['dot_center']}  "
              f"angle={result['angle_deg']}  boundary={result.get('boundary_angle_deg'):.1f}  "
              f"raw_boundary={raw_boundary if raw_boundary is None else f'{raw_boundary:.1f}'}  "
              f"conf={result['dot_confidence']:.3f}  via {result['method']}")
        return result

RFDETR_RESOLUTION       = 512    # must match the checkpoint (see note in _get_model)


class ICDetector:
    def __init__(self, model_path: str = RFDETR_PATH, conf: float = CONF_THRESHOLD):
        self.conf        = conf
        print("model confidence: ", conf)
        self._model      = None
        self._model_path = model_path

    def _get_model(self):
        if self._model is not None:
            return self._model
        import time as _time
        _t0 = _time.perf_counter()
        try:
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")   # suppress rfdetr FutureWarnings on import
                from rfdetr import RFDETRSmall
        except ImportError:
            raise ImportError("pip install rfdetr")
        mp = Path(self._model_path)
        if not mp.exists():
            raise FileNotFoundError(f"RF-DETR checkpoint not found: {mp.resolve()}")

        import shutil
        _roboflow_cache = Path.home() / ".roboflow" / "models"
        _roboflow_cache.mkdir(parents=True, exist_ok=True)
        _cached = _roboflow_cache / mp.name
        if not _cached.exists() or _cached.stat().st_size != mp.stat().st_size:
            print(f"[RFDETR] Copying checkpoint to roboflow cache: {_cached}")
            shutil.copy2(mp, _cached)
        else:
            print(f"[RFDETR] Checkpoint already in roboflow cache: {_cached}")

        self._model = RFDETRSmall(pretrain_weights=str(mp))

        if hasattr(self._model, "optimize_for_inference"):
            print("[RFDETR] Optimizing model for inference ...")
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                self._model.optimize_for_inference()
        else:
            print("[RFDETR] optimize_for_inference() not available -- skipping.")

        # Warmup pass — absorbs first-call JIT/graph-build cost here at load time
        # instead of on the operator's first 'ref'/'test' command.
        print("[RFDETR] Running warmup inference ...")
        _wt0 = _time.perf_counter()
        _dummy = np.zeros((RFDETR_RESOLUTION, RFDETR_RESOLUTION, 3), dtype=np.uint8)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            try:
                self._model.predict(_dummy, threshold=self.conf)
            except Exception:
                pass
        print(f"[RFDETR] Warmup done in {_time.perf_counter() - _wt0:.2f}s")

        print(f"[RFDETR] Loaded model: {mp}  (ready in {_time.perf_counter() - _t0:.2f}s)")
        return self._model

    def detect(self, image: np.ndarray) -> List[Dict]:
        model    = self._get_model()
        rgb      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result   = model.predict(rgb, threshold=self.conf)
        names    = list(getattr(model, 'class_names', None) or [])
        dets     = []
        for box, score, class_id in zip(result.xyxy, result.confidence, result.class_id):
            cls_name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
            b = [float(v) for v in box]
            area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            dets.append({'box': b, 'score': float(score), 'class': cls_name, 'area_px2': area})
            print(f"  [RFDETR] {cls_name}  conf={score:.2f}  bbox={[int(v) for v in b]}  area={area:.1f}px^2")
        print(f"[RFDETR] Detected {len(dets)} IC_chip(s)  (conf>={self.conf})")
        return dets

class PinHealthChecker:
    # Strip geometry
    STRIP_FRAC   = 0.20    
    MIN_STRIP_PX = 12      

    # preprocess()
    CLAHE_CLIP  = 2.0
    CLAHE_TILE  = (8, 8)
    BLUR_KSIZE  = (3, 3)
    MORPH_KSIZE = (3, 3)

    # diff / defect counting
    DIFF_THRESH              = 40      
    DIFF_OPEN_KSIZE           = (5, 5) 
    MIN_DEFECT_AREA          = 20      
    MIN_DEFECT_AREA_FRAC     = 0.0006  
    SCORE_PENALTY_PER_DEFECT = 0.20    
    ENABLE_BAND_ISOLATION = True
    BAND_ISOLATION_FRAC   = 0.20   
    BAND_ISOLATION_MARGIN = 3      
    ENABLE_PIN_LENGTH_CHECK = True
    MIN_PIN_AREA            = 15     
    ENABLE_SHORT_PIN_CHECK  = False  
    MIN_PIN_LENGTH_RATIO    = 0.60  
    PIN_MATCH_DIST_FRAC     = 0.60   
    PIN_MATCH_DIST_MIN      = 5.0    
    MAX_PIN_WIDTH_RATIO     = 1.6   
    FLAG_UNMATCHED_REF_PINS = False 
    def __init__(self, out_dir: str = "output/pin_health",
                 save_intermediate: bool = SAVE_PIN_HEALTH_PREPROCESSING):
        self.out_dir = Path(out_dir)
        self.save_intermediate = save_intermediate
        if self.save_intermediate:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self._n_calls = 0

    # ─────────────────────────────────────────────────────────────────────
    def check(self, test_chip: np.ndarray,
              ref_chip: np.ndarray = None) -> dict:
        if ref_chip  is None or ref_chip.size  == 0: return self._empty()
        if test_chip is None or test_chip.size == 0: return self._empty()
        if test_chip.shape[0] < 20 or test_chip.shape[1] < 20: return self._empty()

        if ref_chip.shape[:2] != test_chip.shape[:2]:
            test_chip = cv2.resize(test_chip,
                                   (ref_chip.shape[1], ref_chip.shape[0]),
                                   interpolation=cv2.INTER_AREA)

        self._n_calls += 1
        call_tag = f"{self._n_calls:04d}"

        ref_strips  = self._extract_strips(ref_chip)
        test_strips = self._extract_strips(test_chip)

        side_scores  = {}
        side_details = {}
        total_defects = 0
        total_short_pins = 0

        for side in ("TOP", "BOTTOM", "LEFT", "RIGHT"):
            d = self._compare_strip(side, ref_strips[side], test_strips[side], call_tag)
            side_scores[side]  = d["score"]
            side_details[side] = d
            total_defects += d["defects"]
            total_short_pins += d.get("short_pins", 0)

        health = float(np.mean(list(side_scores.values()))) if side_scores else 1.0

        print("  [PinHealth] " +
              "  ".join(f"{s}:{v:.2f}" for s, v in side_scores.items()) +
              f"  overall:{health:.2f}" +
              (f"  short_pins:{total_short_pins}" if total_short_pins else ""))
        for s, d in side_details.items():
            if d["defects"]:
                print(f"    [{s}] defects={d['defects']}  score={d['score']:.2f}")

        return {
            "health_score": health,  "ssim_score": health,  "is_ic": True,
            "total_pins":   total_defects, "missing": 0,
            "bent":         total_defects, "bridged": 0,
            "ok":           1 if total_defects == 0 else 0,
            "defect_pixels": total_defects,
            "short_pins":   total_short_pins,
            "side_scores":  side_scores,
            "side_details": side_details,
        }

    # ─────────────────────────────────────────────────────────────────────
    #  upin.py: extract_strips
    # ─────────────────────────────────────────────────────────────────────
    def _extract_strips(self, img: np.ndarray) -> dict:
        h, w = img.shape[:2]
        sh = max(self.MIN_STRIP_PX, int(h * self.STRIP_FRAC))
        sw = max(self.MIN_STRIP_PX, int(w * self.STRIP_FRAC))
        sh = min(sh, max(1, h // 2))
        sw = min(sw, max(1, w // 2))
        return {
            "TOP":    img[0:sh,      :],
            "BOTTOM": img[h - sh:h,  :],
            "LEFT":   img[:, 0:sw],
            "RIGHT":  img[:, w - sw:w],
        }

    # ─────────────────────────────────────────────────────────────────────
    #  preprocess: grayscale + CLAHE + blur (no thresholding yet)
    # ─────────────────────────────────────────────────────────────────────
    def _enhance_gray(self, strip: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip.copy()

        clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP, tileGridSize=self.CLAHE_TILE)
        gray  = clahe.apply(gray)

        gray = cv2.GaussianBlur(gray, self.BLUR_KSIZE, 0)
        return gray


    def _binarize(self, gray: np.ndarray, thresh: float = None):

        if thresh is None:
            otsu_val, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            otsu_val = thresh
            _, th = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

        kernel = np.ones(self.MORPH_KSIZE, np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

        return th, otsu_val

    def _preprocess(self, strip: np.ndarray) -> np.ndarray:
        th, _ = self._binarize(self._enhance_gray(strip))
        return th

    # ─────────────────────────────────────────────────────────────────────
    #  isolate the pin/lead comb from surrounding background noise
    # ─────────────────────────────────────────────────────────────────────
    def _isolate_band(self, mask: np.ndarray, axis: str) -> np.ndarray:
        if not self.ENABLE_BAND_ISOLATION:
            return mask

        fg = mask > 0
        profile = fg.sum(axis=1) if axis == "row" else fg.sum(axis=0)
        n = profile.shape[0]
        peak = int(profile.max()) if n else 0
        if peak == 0:
            return mask  # nothing detected at all -- leave as-is

        thresh = peak * self.BAND_ISOLATION_FRAC

        runs = []
        start = None
        for i in range(n):
            above = profile[i] >= thresh
            if above and start is None:
                start = i
            elif not above and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, n - 1))

        if not runs:
            return mask
        best = max(runs, key=lambda r: int(profile[r[0]:r[1] + 1].sum()))
        lo = max(0, best[0] - self.BAND_ISOLATION_MARGIN)
        hi = min(n, best[1] + self.BAND_ISOLATION_MARGIN + 1)

        cleaned = np.zeros_like(mask)
        if axis == "row":
            cleaned[lo:hi, :] = mask[lo:hi, :]
        else:
            cleaned[:, lo:hi] = mask[:, lo:hi]
        return cleaned

    # ─────────────────────────────────────────────────────────────────────
    #  per-pin length comparison
    # ─────────────────────────────────────────────────────────────────────
    def _measure_pins(self, mask: np.ndarray, axis: str) -> list:

        binary = (mask > 0).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        pins = []
        for label in range(1, num_labels):  # label 0 is background
            x, y, w, h, area = stats[label]
            if area < self.MIN_PIN_AREA:
                continue
            if axis == "row":
                length, pos = h, x + w / 2.0
            else:
                length, pos = w, y + h / 2.0
            pins.append({"pos": float(pos), "length": int(length),
                         "bbox": (int(x), int(y), int(w), int(h))})

        pins.sort(key=lambda p: p["pos"])
        return pins

    def _find_short_pins(self, ref_pins: list, test_pins: list, axis: str) -> list:
        if not ref_pins:
            return []
        if not test_pins:
            return ([{"ref": rp, "test": None, "ratio": 0.0, "reason": "missing"}
                      for rp in ref_pins] if self.FLAG_UNMATCHED_REF_PINS else [])

        if len(ref_pins) > 1:
            gaps = [ref_pins[i + 1]["pos"] - ref_pins[i]["pos"]
                    for i in range(len(ref_pins) - 1)]
            pitch = float(np.median(gaps))
        else:
            pitch = float("inf")
        max_dist = max(pitch * self.PIN_MATCH_DIST_FRAC, self.PIN_MATCH_DIST_MIN)
        thick_idx = 2 if axis == "row" else 3

        used = set()
        failures = []
        for rp in ref_pins:
            best_i, best_d = None, None
            for i, tp in enumerate(test_pins):
                if i in used:
                    continue
                d = abs(tp["pos"] - rp["pos"])
                if d > max_dist:
                    continue
                if best_d is None or d < best_d:
                    best_d, best_i = d, i

            if best_i is None:
                if self.FLAG_UNMATCHED_REF_PINS:
                    failures.append({"ref": rp, "test": None, "ratio": 0.0,
                                      "reason": "missing"})
                continue

            tp = test_pins[best_i]
            used.add(best_i)

            rp_thick = rp["bbox"][thick_idx]
            tp_thick = tp["bbox"][thick_idx]
            if rp_thick > 0 and tp_thick > rp_thick * self.MAX_PIN_WIDTH_RATIO:
                failures.append({"ref": rp, "test": tp,
                                  "ratio": tp_thick / rp_thick,
                                  "reason": "merged"})
                continue

            if not self.ENABLE_SHORT_PIN_CHECK:
                continue
            if rp["length"] <= 0:
                continue
            ratio = tp["length"] / rp["length"]
            if ratio < self.MIN_PIN_LENGTH_RATIO:
                failures.append({"ref": rp, "test": tp, "ratio": ratio,
                                  "reason": "short"})
        return failures

    # ─────────────────────────────────────────────────────────────────────
    #  align (ECC, rotation + translation)
    # ─────────────────────────────────────────────────────────────────────
    def _align(self, ref: np.ndarray, test: np.ndarray) -> np.ndarray:
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            100,
            1e-5,
        )
        try:
            _, warp = cv2.findTransformECC(
                ref.astype(np.float32),
                test.astype(np.float32),
                warp,
                cv2.MOTION_EUCLIDEAN,
                criteria,
            )
            aligned = cv2.warpAffine(
                test,
                warp,
                (ref.shape[1], ref.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            )
        except cv2.error:
            aligned = test.copy()

        return aligned

    # ─────────────────────────────────────────────────────────────────────
    #  upin.py: compare_strip
    # ─────────────────────────────────────────────────────────────────────
    def _compare_strip(self, name: str, ref_strip: np.ndarray,
                       test_strip: np.ndarray, call_tag: str) -> dict:
        empty = {"score": 1.0, "defects": 0, "n_contours": 0}

        if (ref_strip.size == 0 or test_strip.size == 0
                or ref_strip.shape[0] < 3 or ref_strip.shape[1] < 3):
            return empty

        if ref_strip.shape[:2] != test_strip.shape[:2]:
            test_strip = cv2.resize(test_strip,
                                    (ref_strip.shape[1], ref_strip.shape[0]),
                                    interpolation=cv2.INTER_AREA)

        ref_gray  = self._enhance_gray(ref_strip)
        test_gray = self._enhance_gray(test_strip)
        ref_bin,  ref_otsu = self._binarize(ref_gray)           # ref picks the cutoff
        test_bin, _        = self._binarize(test_gray, ref_otsu)  # test reuses it


        band_axis = "row" if name in ("TOP", "BOTTOM") else "col"
        ref_bin  = self._isolate_band(ref_bin,  axis=band_axis)
        test_bin = self._isolate_band(test_bin, axis=band_axis)

        aligned = self._align(ref_bin, test_bin)

        diff = cv2.absdiff(ref_bin, aligned)
        _, diff = cv2.threshold(diff, self.DIFF_THRESH, 255, cv2.THRESH_BINARY)

        diff_kernel = np.ones(self.DIFF_OPEN_KSIZE, np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, diff_kernel)

        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = test_strip.copy() if self.save_intermediate else None
        defect_count = 0

        strip_area = ref_strip.shape[0] * ref_strip.shape[1]
        min_defect_area = max(self.MIN_DEFECT_AREA,
                               strip_area * self.MIN_DEFECT_AREA_FRAC)

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_defect_area:
                continue
            defect_count += 1
            if self.save_intermediate:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        short_pin_failures = []
        if self.ENABLE_PIN_LENGTH_CHECK:
            ref_pins  = self._measure_pins(ref_bin, axis=band_axis)
            test_pins = self._measure_pins(aligned, axis=band_axis)
            short_pin_failures = self._find_short_pins(ref_pins, test_pins, axis=band_axis)
            defect_count += len(short_pin_failures)

            if self.save_intermediate:
                colors = {"short": (0, 255, 255),    # yellow
                          "merged": (0, 165, 255),    # orange
                          "missing": (255, 0, 255)}   # magenta
                for f in short_pin_failures:
                    if f["test"] is not None:
                        x, y, w, h = f["test"]["bbox"]
                    else:

                        rx, ry, rw, rh = f["ref"]["bbox"]
                        x, y, w, h = rx, ry, rw, rh
                    cv2.rectangle(result, (x, y), (x + w, y + h),
                                  colors.get(f["reason"], (0, 255, 255)), 2)

        score = float(np.clip(1.0 - defect_count * self.SCORE_PENALTY_PER_DEFECT, 0.0, 1.0))

        if self.save_intermediate:
            tag = f"{call_tag}_{name}"
            self._save(f"{tag}_ref.png", ref_strip)
            self._save(f"{tag}_test.png", test_strip)
            self._save(f"{tag}_ref_binary.png", ref_bin)
            self._save(f"{tag}_test_binary_isolated.png", test_bin)
            self._save(f"{tag}_aligned.png", aligned)
            self._save(f"{tag}_difference.png", diff)
            self._save(f"{tag}_result.png", result)

        if short_pin_failures:
            summary = ", ".join(f"{f['reason']}:{f['ratio']:.2f}" for f in short_pin_failures)
            print(f"    [{name}] pin failures={len(short_pin_failures)}  [{summary}]")

        return {
            "score": score,
            "defects": defect_count,
            "n_contours": len(contours),
            "short_pins": len(short_pin_failures),
            "short_pin_details": [
                {"pos": f["ref"]["pos"], "ref_length": f["ref"]["length"],
                 "test_length": f["test"]["length"] if f["test"] else None,
                 "ratio": f["ratio"], "reason": f["reason"]}
                for f in short_pin_failures
            ],
        }

    def _save(self, name: str, img: np.ndarray) -> None:
        if self.save_intermediate and img is not None and img.size:
            cv2.imwrite(str(self.out_dir / name), img)

    @staticmethod
    def _empty() -> dict:
        return {
            "health_score": 1.0,  "ssim_score": 1.0,  "is_ic": False,
            "total_pins":   0,    "missing":    0,     "bent":  0,
            "bridged":      0,    "ok":         0,     "defect_pixels": 0,
            "short_pins":   0,
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


def crop_chip_with_margin(image, box, size=128, margin_frac=0.35):
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    mx, my = bw * margin_frac, bh * margin_frac
    ix1 = int(round(x1 - mx)); iy1 = int(round(y1 - my))
    ix2 = int(round(x2 + mx)); iy2 = int(round(y2 + my))
    ix1, iy1 = max(0, ix1), max(0, iy1)
    ix2, iy2 = min(image.shape[1], ix2), min(image.shape[0], iy2)
    chip = image[iy1:iy2, ix1:ix2]
    if chip.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(chip, (size, size))


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
    def __init__(self, yolo_path=RFDETR_PATH, conf_threshold=CONF_THRESHOLD, roi=None, detector=None):
        self.detector   = detector if detector is not None else ICDetector(model_path=yolo_path, conf=conf_threshold)
        self.aligner    = AdvancedImageAligner()
        # Secondary orientation verification stage (IC outer boundary only).
        # Independent of self.aligner; only ever consulted after self.aligner
        # has already reported a component's orientation as PASS.
        self.boundary_verifier = ICBoundaryVerifier()
        self.pin_health = PinHealthChecker()
        self.roi        = roi
        self.reference_image          = None
        self.reference_frame          = None   # raw, uncropped reference frame --
                                                # kept alongside the tight chip crops
                                                # so the secondary boundary verifier
                                                # can re-crop with real background
                                                # margin (see crop_chip_with_margin)
        self.reference_detections     = None
        self.ref_chips_cache          = []
        self.ref_chips_native_cache   = []
        self.ref_from_file            = False
        self.conf                     = conf_threshold
        print("model confidence: ", conf_threshold)

    def _filter_by_roi(self, dets):

        if self.roi is None: return dets
        rx1, ry1, rx2, ry2 = self.roi
        kept = []
        for d in dets:
            bx1, by1, bx2, by2 = d['box']
            ix1, iy1 = max(bx1, rx1), max(by1, ry1)
            ix2, iy2 = min(bx2, rx2), min(by2, ry2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue   # zero overlap -- genuinely outside the ROI
            d['box'] = [ix1, iy1, ix2, iy2]   # clip to the ROI itself
            kept.append(d)
        if len(dets) - len(kept):
            print(f"[ROI] Filtered {len(dets) - len(kept)} with no overlap")
        return kept

    def _detect_in_roi(self, image, margin=0, label="input"):
        if self.roi is None:
            print(f"[ROI][{label}] no ROI set — running detection on full frame")
            return self._filter_by_roi(self.detector.detect(image))

        h, w = image.shape[:2]
        rx1, ry1, rx2, ry2 = self.roi
        print(f"[ROI][{label}] user ROI: x1={rx1} y1={ry1} x2={rx2} y2={ry2}"
              f"  (frame={w}x{h})")
        cx1 = max(0, int(rx1 - margin)); cy1 = max(0, int(ry1 - margin))
        cx2 = min(w, int(rx2 + margin)); cy2 = min(h, int(ry2 + margin))
        if cx2 <= cx1 or cy2 <= cy1:
            print(f"[ROI][{label}] degenerate ROI crop — "
                  f"falling back to full-frame detection")
            return self._filter_by_roi(self.detector.detect(image))

        print(f"[ROI][{label}] detect crop (ROI only, margin={margin}px): "
              f"x1={cx1} y1={cy1} x2={cx2} y2={cy2}  ({cx2-cx1}x{cy2-cy1})")
        crop = image[cy1:cy2, cx1:cx2]
        dets = self.detector.detect(crop)
        for d in dets:
            bx1, by1, bx2, by2 = d['box']
            d['box'] = [bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1]
        dets = self._filter_by_roi(dets)
        status = "PRESENT" if dets else "ABSENT"
        print(f"[ROI][{label}] IC {status} — {len(dets)} detection(s) in ROI crop")
        return dets

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

    def _check_area_and_roi_containment(self, test_box, ref_area_px2: float) -> Tuple[bool, str]:
        bx1, by1, bx2, by2 = test_box
        test_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        if ref_area_px2 and ref_area_px2 > 0 and test_area > 0:
            ratio = test_area / ref_area_px2
            lo, hi = 1.0 - AREA_RATIO_TOLERANCE, 1.0 + AREA_RATIO_TOLERANCE
            if not (lo <= ratio <= hi):
                return False, (f"area_mismatch (test={test_area:.0f}px^2 "
                                f"ref={ref_area_px2:.0f}px^2 ratio={ratio:.2f}, "
                                f"need {lo:.2f}-{hi:.2f})")

        if self.roi is not None and test_area > 0:
            rx1, ry1, rx2, ry2 = self.roi
            ix1, iy1 = max(bx1, rx1), max(by1, ry1)
            ix2, iy2 = min(bx2, rx2), min(by2, ry2)
            inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            containment = inter_area / test_area
            if containment < ROI_CONTAINMENT_MIN_FRAC:
                return False, (f"roi_containment_low ({containment:.0%} of the "
                                f"component is inside the ROI, need "
                                f">= {ROI_CONTAINMENT_MIN_FRAC:.0%})")

        return True, "ok"

    def set_reference(self, image: np.ndarray):
        print("\nProcessing reference image...")
        dets = self._detect_in_roi(image, label="ref")
        if not dets:
            print("Warning: No IC components detected inside ROI in reference.")
            return

        for d in dets:
            bx1, by1, bx2, by2 = d['box']
            d['area_px2'] = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            print(f"  [RefArea] comp bbox={[int(v) for v in d['box']]}  "
                  f"area={d['area_px2']:.1f}px^2  (from reference image)")

        # crop_chip_native() is applied directly to the raw frame + the
        # original detection box, so it stays at true native resolution --
        # NOT derived from the resized/CLAHE'd 128x128 `chip`. This is what
        # keeps ref_marking_frac comparable to offline's fast path (which
        # loads the genuine chip_*_native.png saved here from disk).
        ref_chips        = [crop_chip(image, d['box']) for d in dets]
        ref_chips_native = [crop_chip_native(image, d['box']) for d in dets]

        self.reference_image        = ref_chips[0]
        self.reference_frame        = image
        self.reference_detections   = dets
        self.ref_chips_cache        = ref_chips
        self.ref_chips_native_cache = ref_chips_native
        self.ref_from_file          = False
        Path('ref').mkdir(exist_ok=True)
        cv2.imwrite('ref/input_frame.png', image)
        for idx, (chip, chip_native) in enumerate(zip(ref_chips, ref_chips_native)):
            cv2.imwrite(f'ref/chip_{idx}.png', chip)
            cv2.imwrite(f'ref/chip_{idx}_native.png', chip_native)
        roi_vis = image.copy()
        for idx, d in enumerate(dets):
            x1, y1, x2, y2 = map(int, d['box'])
            cv2.rectangle(roi_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(roi_vis, f'REF{idx}', (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.roi:
            rx1, ry1, rx2, ry2 = self.roi
            cv2.rectangle(roi_vis, (rx1, ry1), (rx2, ry2), (255, 165, 0), 2)
        cv2.imwrite('ref/roi_frame.png', roi_vis)
        print(f'Saved: ref/input_frame.png  ref/chip_0..{len(dets)-1}.png  '
              'ref/chip_*_native.png  ref/roi_frame.png')
        print(f'Reference set with {len(dets)} IC component(s) inside ROI')

    def load_reference_from_file(self, ref_chip_path: str):
        print(f"\nLoading reference chip from: {ref_chip_path}")
        raw = cv2.imread(ref_chip_path)
        if raw is None:
            print(f"Error: Cannot read reference chip from '{ref_chip_path}'")
            return


        box = [0, 0, raw.shape[1], raw.shape[0]]
        try:
            dets = self.detector.detect(raw)
        except Exception as e:
            dets = []
            print(f"  [RefFile] detection on loaded reference failed ({e}); "
                  f"falling back to whole-image crop")
        if dets:
            best = max(dets, key=lambda d: d.get('score', 0.0))
            box = best['box']
            print(f"  [RefFile] IC detected in loaded reference "
                  f"(conf={best.get('score', 0.0):.2f}) bbox={box} "
                  f"— cropping to match live test-chip framing")
        else:
            print(f"  [RefFile] No IC detected inside loaded reference — "
                  f"using whole image as chip (orientation confidence may "
                  f"be degraded if this doesn't match test-chip framing)")

        chip = crop_chip(raw, box)
        ref_area_px2 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        print(f"  [RefArea] bbox={[int(v) for v in box]}  "
              f"area={ref_area_px2:.1f}px^2  (from reference image)")
        native_path = str(ref_chip_path).replace('chip_0.png', 'chip_0_native.png')
        ni = cv2.imread(native_path)
        if ni is not None:
            native_chip = crop_chip_native(ni, [0, 0, ni.shape[1], ni.shape[0]])
            print(f'  Native ref loaded: {native_path}')
        else:
            native_chip = crop_chip_native(raw, box)
            print('  Native ref not found, using cropped-from-source fallback')
        self.reference_image        = chip
        self.reference_frame        = raw
        self.ref_chips_cache        = [chip]
        self.ref_chips_native_cache = [native_chip]
        self.reference_detections   = [{'box': box, 'score': 1.0, 'class': 'IC_chip',
                                         'area_px2': ref_area_px2}]
        self.ref_from_file = True
        print('Reference chip loaded — framed to match live-detection crops')

    def _recheck_reference_live(self):
        """Re-runs the model on the stored reference frame at the top of every
        live/test call, so the reference crop used for this comparison
        reflects a fresh presence check and a freshly model-derived bounding
        box, rather than only ever trusting the crop taken when the
        reference was first set (or loaded from file).

        Two cases:
          - ref_from_file: the stored reference_frame is just the loaded
            chip/reference image itself, not a live-camera frame, so it is
            re-detected with a plain (no-ROI) model call, exactly like
            load_reference_from_file() does on first load.
          - otherwise: reference_frame is a full live-camera frame, so it is
            re-detected the same way live test frames are, via the ROI-aware
            detector.

        In either case, if the model doesn't find an IC on this recheck, the
        previously cached reference crop is kept as-is and a warning is
        printed rather than failing the run.
        """
        if self.reference_frame is None:
            return

        if self.ref_from_file:
            try:
                dets = self.detector.detect(self.reference_frame)
            except Exception as e:
                print(f"[RefLive] WARNING: re-detection on the file-loaded "
                      f"reference failed ({e}) -- keeping the previously "
                      f"cached reference crop.")
                return
            if not dets:
                print("[RefLive] WARNING: model did not detect an IC in the "
                      "file-loaded reference on this live check -- keeping "
                      "the previously cached reference crop.")
                return
            best = max(dets, key=lambda d: d.get('score', 0.0))
            box  = best['box']
            bx1, by1, bx2, by2 = box
            ref_area_px2 = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

            chip        = crop_chip(self.reference_frame, box)
            native_chip = crop_chip_native(self.reference_frame, box)

            self.reference_image        = chip
            self.ref_chips_cache        = [chip]
            self.ref_chips_native_cache = [native_chip]
            self.reference_detections   = [{'box': box, 'score': best.get('score', 1.0),
                                             'class': best.get('class', 'IC_chip'),
                                             'area_px2': ref_area_px2}]

            print(f"[RefLive] Re-confirmed IC present in file-loaded "
                  f"reference (conf={best.get('score', 0.0):.2f}); "
                  f"re-cropped via model bbox for this live check: "
                  f"{[int(v) for v in box]}")
            return

        ref_dets_live = self._detect_in_roi(self.reference_frame, label="ref-live")
        if not ref_dets_live:
            print("[RefLive] WARNING: model did not detect an IC in the stored "
                  "reference frame on this live check -- keeping the "
                  "previously cached reference crop(s).")
            return

        for d in ref_dets_live:
            bx1, by1, bx2, by2 = d['box']
            d['area_px2'] = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        ref_chips_live        = [crop_chip(self.reference_frame, d['box'])
                                  for d in ref_dets_live]
        ref_chips_native_live = [crop_chip_native(self.reference_frame, d['box'])
                                  for d in ref_dets_live]

        self.reference_detections   = ref_dets_live
        self.ref_chips_cache        = ref_chips_live
        self.ref_chips_native_cache = ref_chips_native_live
        self.reference_image        = ref_chips_live[0]

        print(f"[RefLive] Re-confirmed IC present in reference "
              f"({len(ref_dets_live)} component(s)); re-cropped via model "
              f"bbox for this live check: "
              f"{[[int(v) for v in d['box']] for d in ref_dets_live]}")



    def detect_and_compare(self, test_image: np.ndarray, save_images=False) -> dict:
        if self.reference_image is None:
            return {'status': 'error', 'message': 'No reference image set'}

        self._recheck_reference_live()

        test_dets = self._detect_in_roi(test_image, label="test")
        if not test_dets:
            missing = [{'ref_id': i, 'box': d['box'], 'status': 'absent'}
                       for i, d in enumerate(self.reference_detections)]
            return {'status': 'success', 'num_components': 0, 'components': [],
                    'missing_components': missing,
                    'comparison': {'matched': 0,
                                   'missing': len(self.reference_detections),
                                   'wrong_orientation': 0, 'pin_failures': 0}}

        test_chips        = [crop_chip(test_image, d['box']) for d in test_dets]
        test_chips_native = [crop_chip_native(test_image, d['box']) for d in test_dets]
        N_test, N_ref = len(test_dets), len(self.reference_detections)

        # ── Live-mode IC crop save ──────────────────────────────────────
        # Persist the model's detected bounding boxes for this live/test
        # frame as individual cropped images, mirroring what set_reference()
        # already does for the reference frame (ref/chip_*.png).
        try:
            import time as _time
            _ts       = _time.strftime('%Y%m%d_%H%M%S_') + f"{int(_time.time()*1000) % 1000:03d}"
            _test_dir = Path('output') / 'test_chips' / _ts
            _test_dir.mkdir(parents=False, exist_ok=False)
            #cv2.imwrite(str(_test_dir / 'input_frame.png'), test_image)
            for idx, (d, chip, chip_native) in enumerate(
                    zip(test_dets, test_chips, test_chips_native)):
                x1, y1, x2, y2 = map(int, d['box'])
                #cv2.imwrite(str(_test_dir / f'chip_{idx}.png'), chip)
                #cv2.imwrite(str(_test_dir / f'chip_{idx}_native.png'), chip_native)
                print(f"  [LiveCrop] comp_id={idx}  bbox=[{x1}, {y1}, {x2}, {y2}]  "
                      f"-> {_test_dir / f'chip_{idx}.png'}")
            print(f"[LiveCrop] Saved {len(test_dets)} detected IC crop(s) "
                  f"from live frame to {_test_dir}")
        except Exception as _e:
            print(f"[LiveCrop] Warning: could not save live test-chip crops: {_e}")

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


            ref_native = None
            if ref_idx >= 0:
                if ref_idx < len(self.ref_chips_native_cache):
                    ref_native = self.ref_chips_native_cache[ref_idx]
                else:
                    c = self.ref_chips_cache[ref_idx]
                    ref_native = crop_chip_native(c, [0, 0, c.shape[1], c.shape[0]])


            ref_area_px2 = (self.reference_detections[ref_idx].get('area_px2', 0.0)
                             if ref_idx >= 0 else 0.0)

            # Pin health is only ever applicable to this component if its
            # reference area clears MIN_IC_AREA_PIN_HEALTH_PX2 (mirrors the
            # `area_ok` gate computed again, later, right before the actual
            # pin-health call). When that's true and pin health is enabled,
            # pin health becomes the presence signal for this component and
            # the interior-text check is bypassed entirely.
            pin_health_applicable = (ENABLE_PIN_HEALTH_CHECK
                                      and ref_area_px2 >= MIN_IC_AREA_PIN_HEALTH_PX2)

            no_text_absent = False
            if ref_idx >= 0 and ENABLE_NO_TEXT_ABSENT_CHECK and not pin_health_applicable:
                dynamic_threshold = None
                if ref_native is not None:
                    ref_frac = self.boundary_verifier.compute_mark_frac(
                        ref_native, tag=f"ref{i}")
                    if ref_frac is not None and ref_frac > 0:
                        dynamic_threshold = REF_FRAC_THRESHOLD_PCT * ref_frac
                        print(f"  [NoTextCheck] comp_id={i} ref_marking_frac={ref_frac:.5f}"
                              f"  -> dynamic_threshold={dynamic_threshold:.5f}"
                              f"  ({REF_FRAC_THRESHOLD_PCT*100:.0f}% of ref)")

                has_text = self.boundary_verifier.has_interior_text(
                    test_chips_native[i], tag=f"test{i}", threshold=dynamic_threshold)
                if not has_text:
                    no_text_absent = True
                    print(f"  [PresenceCheck] comp_id={i} NO interior "
                          f"markings on test crop -> ABSENT "
                          f"(orientation checks skipped)")
            elif ref_idx >= 0 and ENABLE_NO_TEXT_ABSENT_CHECK and pin_health_applicable:
                print(f"  [PresenceCheck] comp_id={i} SKIPPED — pin health "
                      f"check applies for this component "
                      f"(ref_area={ref_area_px2:.0f}px^2 >= "
                      f"{MIN_IC_AREA_PIN_HEALTH_PX2}px^2), no-text absence "
                      f"check bypassed")

            # ── Orientation ───────────────────────────────────────────────
            ic_orient = {
                'correct': True, 'rotation_deg': 0.0, 'rotation_step': 0,
                'confidence': 0.0, 'num_matches': 0, 'reason': 'not_checked',
                'verified': False,
                'secondary_verified': False, 'secondary_passed': True,
                'secondary_angle_diff_deg': 0.0, 'secondary_message': '',
            }
            if no_text_absent:
                ic_orient['secondary_message'] = (
                    "no interior markings found on test crop -- "
                    "reporting ABSENT")
            if ref_idx >= 0 and not no_text_absent and ENABLE_ORIENTATION_CHECK:
                ref_c = self.ref_chips_cache[ref_idx]
                tst_c = test_chips[i]
                align = self.aligner.register_images(ref_c, tst_c)

                if not align['success']:
                    ic_orient = {
                        'correct':         True,   # fail-open: don't block the
                                                    # line on an undetermined
                                                    # check, but see 'reason'
                        'rotation_deg':    0.0,
                        'rotation_step':   0,
                        'confidence':      float(align['confidence']),
                        'num_matches':     align['num_matches'],
                        'reason':          'unverified',
                        'verified':        False,
                        'secondary_verified': False, 'secondary_passed': True,
                        'secondary_angle_diff_deg': 0.0, 'secondary_message':
                            'skipped -- primary orientation check unverified',
                    }
                    print(f"  [Orient] comp_id={i}  → UNVERIFIED "
                          f"(orientation could not be determined -- treated "
                          f"as pass, NOT a confirmed match)  [{align['message']}]")
                else:
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
                        'verified':        True,
                        'secondary_verified': False, 'secondary_passed': True,
                        'secondary_angle_diff_deg': 0.0, 'secondary_message': '',
                    }
                    print(f"  [Orient] comp_id={i} rot={raw:+.1f}° step={rot_step}°"
                          f"  conf={align['confidence']:.3f}"
                          f"  → {'WRONG' if is_wrong else 'OK'}"
                          f"  [{align['message']}]")

                if ic_orient['correct'] and not no_text_absent and ENABLE_SECONDARY_BOUNDARY_VERIFICATION:

                    sec = self.boundary_verifier.verify(ref_c, tst_c)
                    ic_orient['secondary_verified']       = sec.verified
                    ic_orient['secondary_passed']         = sec.passed
                    ic_orient['secondary_angle_diff_deg'] = sec.angle_diff_deg
                    ic_orient['secondary_message']        = sec.message
                    _primary_kind = 'verified PASS' if ic_orient['verified'] else 'unverified pass'

                    if sec.verified and not sec.passed:
                        # Boundary-only cross-check disagrees with the
                        # primary's current OK status beyond tolerance --
                        # override to wrong-orientation.
                        ic_orient['correct'] = False
                        ic_orient['reason']  = (
                            f"boundary_verify_failed_{sec.angle_diff_deg:.1f}deg")
                        print(f"  [OrientVerify] comp_id={i} SECONDARY "
                              f"boundary check OVERRIDES primary "
                              f"({_primary_kind}) -> WRONG  [{sec.message}]")
                    else:
                        print(f"  [OrientVerify] comp_id={i} secondary "
                              f"boundary check "
                              f"{'confirms' if sec.verified else 'inconclusive, fail-open confirms'} "
                              f"{_primary_kind}  [{sec.message}]")
            elif ref_idx >= 0 and not ENABLE_ORIENTATION_CHECK:
                print(f"  [Orient] comp_id={i} SKIPPED — orientation check disabled")

            if ref_idx >= 0 and ENABLE_AREA_ROI_CHECK:
                area_roi_ok, area_roi_msg = self._check_area_and_roi_containment(
                    test_dets[i]['box'], ref_area_px2)
                if not area_roi_ok:
                    ic_orient['correct'] = False
                    ic_orient['reason']  = f"area_roi_check_failed ({area_roi_msg})"
                    print(f"  [AreaROI] comp_id={i} FAILED -> WRONG ORIENTATION  [{area_roi_msg}]")
                else:
                    print(f"  [AreaROI] comp_id={i} OK  [{area_roi_msg}]")

            ic_orient_wrong = (
                not ic_orient['correct']
                and ic_orient['reason'] != 'not_checked'
            )

            # ── Pin health ────────────────────────────────────────────────
            area_ok      = ref_area_px2 >= MIN_IC_AREA_PIN_HEALTH_PX2
            chip_sim_ok  = best_sim >= PIN_HEALTH_MIN_CHIP_SIM
            pin_health_enabled = (
                ref_idx >= 0
                and area_ok
                and not ic_orient_wrong
                and not no_text_absent
            )
            print(f"  [PinHealth] comp_id={i}  ref_area={ref_area_px2:.0f}px^2"
                  f" (>= {MIN_IC_AREA_PIN_HEALTH_PX2}px^2: {area_ok})"
                  f"  orientation_ok={not ic_orient_wrong}"
                  f"  no_text_absent={no_text_absent}"
                  f"  -> pin_check={pin_health_enabled}")
            run_pin_health = (
                ENABLE_PIN_HEALTH_CHECK
                and pin_health_enabled
                and self._roi_large_enough_for_pin_health()
                and chip_sim_ok
            )
            if run_pin_health:
                pin = self.pin_health.check(test_chips_native[i], ref_native)
            else:
                if not ENABLE_PIN_HEALTH_CHECK:
                    print(f"  [PinHealth] SKIPPED — pin health check disabled")
                elif ref_idx < 0:
                    print(f"  [PinHealth] SKIPPED — no matched reference component")
                elif not area_ok:
                    print(f"  [PinHealth] SKIPPED — reference IC area too small"
                          f" ({ref_area_px2:.0f}px^2 < {MIN_IC_AREA_PIN_HEALTH_PX2}px^2)")
                elif no_text_absent:
                    print(f"  [PinHealth] SKIPPED — no interior markings found (ABSENT)")
                elif ic_orient_wrong:
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
                or pin.get('bent', 0) < 80
            )


            if no_text_absent:
                status = 'absent'
            elif ic_orient_wrong:
                status = 'wrong_orientation'
            else:
                status = 'present'

            components_info.append({
                'id': i, 'box': test_dets[i]['box'],
                'score': test_dets[i]['score'],
                'class': test_dets[i].get('class', 'IC_chip'),
                'similarity': best_sim,
                'iou_score': float(iou_matrix[i, ref_idx]) if ref_idx >= 0 else 0.0,
                'status': status, 'matched_ref_id': ref_idx if ref_idx >= 0 else None,
                'no_text_absent':       no_text_absent,
                'ic_orient_correct':    ic_orient['correct'],
                'ic_orient_deg':        ic_orient['rotation_deg'],
                'ic_orient_step':       ic_orient['rotation_step'],
                'ic_orient_confidence': ic_orient['confidence'],
                'ic_orient_matches':    ic_orient['num_matches'],
                'ic_orient_reason':     ic_orient['reason'],
                'ic_orient_verified':   ic_orient.get('verified', False),
                'ic_orient_secondary_verified':       ic_orient.get('secondary_verified', False),
                'ic_orient_secondary_passed':         ic_orient.get('secondary_passed', True),
                'ic_orient_secondary_angle_diff_deg': ic_orient.get('secondary_angle_diff_deg', 0.0),
                'ic_orient_secondary_message':        ic_orient.get('secondary_message', ''),
                'pin_health_score': pin.get('health_score', 1.0),
                'pin_health_is_ic':  pin.get('is_ic', False),
                'pin_health_ok':     pin_ok,
                'pin_health_enabled': pin_health_enabled,
                'ref_area_px2':       ref_area_px2,
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

        present_count      = sum(1 for c in components_info if c['status'] == 'present')
        wrong_o_count      = sum(1 for c in components_info if c['status'] == 'wrong_orientation')
        no_text_absent_count = sum(1 for c in components_info if c.get('no_text_absent'))
        pin_fail_count     = sum(1 for c in components_info
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
            #cv2.imwrite(_out_path, _vis)
            print(f'[Output] Saved annotated result: {_out_path}')
        except Exception as _e:
            print(f'[Output] Warning: could not save result image: {_e}')

        return {
            'status': 'success', 'num_components': len(components_info),
            'components': components_info, 'missing_components': missing_components,
            'comparison': {'matched': present_count, 'wrong_orientation': wrong_o_count,
                           # a detected-but-unmarked chip counts toward the
                           # overall ABSENT tally alongside components the
                           # detector never found at all
                           'missing': len(missing_components) + no_text_absent_count,
                           'no_text_absent': no_text_absent_count,
                           'pin_failures': pin_fail_count},
        }

    def _create_annotated_image(self, image, components, missing):
        vis = image.copy()
        colours = {'present': (0, 220, 0), 'wrong_orientation': (0, 165, 255),
                   'absent': (128, 0, 128)}
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
            if c.get('no_text_absent'):
                label += " NO_TEXT"
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
    def __init__(self, yolo_path=RFDETR_PATH, conf_threshold=CONF_THRESHOLD, roi=None, detector=None):
        self.conf      = conf_threshold
        print("model confidence: ", conf_threshold)
        self.inspector = ICInspector(yolo_path, conf_threshold, roi, detector=detector)
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
            print("Wrong orientation: 0")
            print("Pin health failures: 0"); print("[RESULT_END]")
            return

        comps = result.get('components', [])
        miss  = result.get('missing_components', [])
        cmp   = result.get('comparison', {})

        overall_pass = (cmp.get('wrong_orientation', 0) == 0
                        and cmp.get('missing', 0) == 0
                        and cmp.get('pin_failures', 0) == 0)

        print("\n" + "=" * 60)
        print(f"INSPECTION RESULT:  {'✓ PASS' if overall_pass else '✗ FAIL'}")
        print("=" * 60)
        print(f"  IC Present (matched):     {cmp.get('matched', 0)}"
              f"  /  {len(self.inspector.reference_detections or [])}")
        print(f"  IC Absent  (missing):     {cmp.get('missing', 0)}"
              f"  (no interior markings: {cmp.get('no_text_absent', 0)})")
        print(f"  Wrong orientation:        {cmp.get('wrong_orientation', 0)}")
        print(f"  Pin health failures:      {cmp.get('pin_failures', 0)}")
        print("=" * 60)

        sym = {'present': '✓', 'wrong_orientation': '⟳', 'absent': '✗'}
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
                if c.get('no_text_absent'):
                    ln += ("\n      Orientation: NOT CHECKED  (no interior "
                           "markings found on the test crop -- reported ABSENT)")
                elif not c.get('ic_orient_correct', True):
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
        print(f"Wrong orientation: {cmp.get('wrong_orientation', 0)}")
        print(f"Pin health failures: {cmp.get('pin_failures', 0)}")
        print(f"pin_missing={total_pin_missing}")
        print(f"pin_bent={total_pin_bent}")
        print(f"pin_bridge={total_pin_bridge}")
        print(f"rotation={rotation_deg:.2f} step~{rotation_step}")
        print("[RESULT_END]")

import socket as _sock, json as _json, base64 as _b64
import subprocess as _sp, os as _os, time as _time

_SERVER_INFO_FILE       = Path(__file__).resolve().parent / "rfdetr_server.json"
_SERVER_READY_FILE      = Path(__file__).resolve().parent / "rfdetr_server.ready"
_SERVER_LOG_FILE        = Path(__file__).resolve().parent / "rfdetr_server.log"
_SERVER_STARTUP_TIMEOUT = 90    # first load can be slow (backbone dl + warmup)
_SOCKET_TIMEOUT         = 30


class _RemoteICDetector:
    """Drop-in stand-in for ICDetector.detect() that forwards the image to the
    persistent background server instead of loading RF-DETR in this process."""
    def __init__(self, port: int, conf: float = CONF_THRESHOLD):
        self.port = port
        self.conf = conf

    def detect(self, image: np.ndarray) -> List[Dict]:
        ok, buf = cv2.imencode('.png', image)
        if not ok:
            raise RuntimeError("Failed to encode image to send to RF-DETR server")
        payload = _json.dumps({
            "conf":      self.conf,
            "image_b64": _b64.b64encode(buf.tobytes()).decode('ascii'),
        })
        with _sock.create_connection(("127.0.0.1", self.port), timeout=_SOCKET_TIMEOUT) as s:
            s.sendall((payload + "\n").encode())
            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
        resp = _json.loads(data.decode(errors='replace').strip())
        if 'error' in resp:
            raise RuntimeError(f"RF-DETR server error: {resp['error']}")
        dets = resp['dets']
        for d in dets:
            b = d['box']
            area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            d['area_px2'] = area
            print(f"  [RFDETR] {d['class']}  conf={d['score']:.2f}  bbox={[int(v) for v in b]}  area={area:.1f}px^2")
        print(f"[RFDETR] Detected {len(dets)} IC_chip(s)  (conf>={self.conf})  [server:{self.port}]")
        return dets


def _socket_server_mode(port: int, model_path: str, conf_threshold: float) -> None:
    """Runs forever in the background: loads RF-DETR once, then serves detect()
    requests over a local TCP socket, one connection/job at a time."""
    _SERVER_READY_FILE.unlink(missing_ok=True)

    srv = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    srv.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(32)

    print(f"[RFDETR-SERVER] Loading model (pid={_os.getpid()}) ...", flush=True)
    detector = ICDetector(model_path=model_path, conf=conf_threshold)
    detector._get_model()   # force load + optimize + warmup now, not on first job
    _SERVER_READY_FILE.write_text("ready")
    print("[RFDETR-SERVER] READY -- waiting for jobs on "
          f"127.0.0.1:{port}", flush=True)

    while True:
        try:
            conn, addr = srv.accept()
        except Exception as ex:
            print(f"[RFDETR-SERVER] accept() failed: {ex}", flush=True)
            break
        with conn:
            try:
                data = b""
                while not data.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                line = data.decode(errors='replace').strip()
                if not line:
                    continue
                if line.upper() == "EXIT":
                    _SERVER_READY_FILE.unlink(missing_ok=True)
                    conn.sendall(b'{"ok": true}\n')
                    break
                job = _json.loads(line)
                if job.get('_ping'):
                    conn.sendall(b'{"ok": true}\n')
                    continue
                img_bytes   = _b64.b64decode(job['image_b64'])
                arr         = np.frombuffer(img_bytes, dtype=np.uint8)
                image       = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                detector.conf = float(job.get('conf', detector.conf))
                dets = detector.detect(image)
                conn.sendall((_json.dumps({"dets": dets}) + "\n").encode())
            except Exception as e:
                import traceback
                print(f"[RFDETR-SERVER] ERROR: {e}", flush=True)
                traceback.print_exc()
                try:
                    conn.sendall((_json.dumps({"error": str(e)}) + "\n").encode())
                except Exception:
                    pass


def _server_info() -> dict:
    try:
        return _json.loads(_SERVER_INFO_FILE.read_text())
    except Exception:
        return {}


def _pid_alive(pid: int) -> bool:
    """Windows-compatible process alive check (falls back to os.kill elsewhere)."""
    try:
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return exit_code.value == 259   # STILL_ACTIVE
    except Exception:
        try:
            _os.kill(pid, 0)
            return True
        except Exception:
            return False


def _server_alive(info: dict) -> bool:
    if not info:
        return False
    pid, port = info.get('pid'), info.get('port')
    if not pid or not port:
        return False
    if not _pid_alive(pid):
        return False
    if not _SERVER_READY_FILE.exists():
        return False
    try:
        s = _sock.create_connection(("127.0.0.1", port), timeout=1)
        s.close()
        return True
    except OSError:
        return False


def _spawn_socket_server(model_path: str, conf_threshold: float) -> int:
    with _sock.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    cmd = [sys.executable, str(Path(__file__).resolve()), '--_socket_server', str(port),
           '--yolo-path', model_path, '--conf-threshold', str(conf_threshold)]

    log_file = open(str(_SERVER_LOG_FILE), 'w')
    kwargs = {}
    if sys.platform == 'win32':
        kwargs['creationflags'] = _sp.DETACHED_PROCESS | _sp.CREATE_NEW_PROCESS_GROUP

    proc = _sp.Popen(cmd, stdout=log_file, stderr=log_file,
                     cwd=str(Path(__file__).resolve().parent),
                     close_fds=True, **kwargs)
    _SERVER_INFO_FILE.write_text(_json.dumps({"pid": proc.pid, "port": port}))
    return port


def _wait_for_server(port: int, timeout: float = _SERVER_STARTUP_TIMEOUT) -> bool:
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        if _SERVER_READY_FILE.exists():
            try:
                s = _sock.create_connection(("127.0.0.1", port), timeout=1)
                s.close()
                return True
            except OSError:
                pass
        _time.sleep(0.3)
    return False


def _get_remote_detector(model_path: str, conf_threshold: float):
    """Ensures a persistent background RF-DETR server is running and returns a
    _RemoteICDetector proxy pointed at it. Returns None on any failure so the
    caller can fall back to loading RF-DETR directly in this process."""
    try:
        info = _server_info()
        if not _server_alive(info):
            print("[AUTO-SERVER] Starting background RF-DETR model server "
                  "(first call, or server restarted -- loads model once) ...")
            t0   = _time.time()
            port = _spawn_socket_server(model_path, conf_threshold)
            if not _wait_for_server(port):
                print("[AUTO-SERVER] Server did not become ready in time -- "
                      "loading the model directly in this process instead.")
                return None
            print(f"[AUTO-SERVER] Server ready in {_time.time() - t0:.1f}s")
        else:
            port = info['port']
            print(f"[AUTO-SERVER] Reusing warm model server on port {port}")
        return _RemoteICDetector(port, conf_threshold)
    except Exception as e:
        print(f"[AUTO-SERVER] Could not use background server ({e}) -- "
              f"loading the model directly in this process instead.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    for d in ('output', 'tmp', 'images', 'ref'): Path(d).mkdir(exist_ok=True)

    # Hidden mode: this is the persistent background server process itself.
    if '--_socket_server' in sys.argv:
        _idx  = sys.argv.index('--_socket_server')
        _port = int(sys.argv[_idx + 1])
        _p2   = argparse.ArgumentParser(add_help=False)
        _p2.add_argument('--yolo-path',      default=RFDETR_PATH)
        _p2.add_argument('--conf-threshold', type=float, default=CONF_THRESHOLD)
        _sargs, _ = _p2.parse_known_args(sys.argv[1:_idx] + sys.argv[_idx + 2:])
        _socket_server_mode(_port, _sargs.yolo_path, _sargs.conf_threshold)
        raise SystemExit(0)

    parser = argparse.ArgumentParser(description='IC Inspection')
    parser.add_argument('--yolo-path',      default=RFDETR_PATH,
                        help='Path to the RF-DETR checkpoint (e.g. checkpoint_best_ema.pth). '
                             'Flag kept as --yolo-path for launcher/GUI backward-compat.')
    parser.add_argument('--conf-threshold', type=float, default=CONF_THRESHOLD)
    parser.add_argument('--mode',           default='interactive',
                        choices=['interactive', 'check_roi'])
    parser.add_argument('--image',          default=None)
    parser.add_argument('--roi-coords',     default=None)
    parser.add_argument('--roi-file',       default=None)
    parser.add_argument('--no-server',      action='store_true',
                        help='Skip the persistent background model server and '
                             'load RF-DETR directly in this process (slower per call).')
    args = parser.parse_args()

    detector = None if args.no_server else _get_remote_detector(args.yolo_path, args.conf_threshold)

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
            det   = detector if detector is not None else ICDetector(
                        model_path=args.yolo_path, conf=args.conf_threshold)
            dets  = det.detect(img)
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
                        roi=roi, detector=detector).run_interactive()
    except Exception as _e:
        import traceback as _tb
        print(f'[FATAL] {_e}', flush=True); _tb.print_exc()
        print('[RESULT_BEGIN]')
        print('IC Present (matched): 0'); print('IC Absent  (missing): 1')
        print('Wrong orientation: 0')
        print('Pin health failures: 0'); print('pin_missing=0')
        print('pin_bent=0'); print('pin_bridge=0')
        print('rotation=0.00 step~0'); print('[RESULT_END]')