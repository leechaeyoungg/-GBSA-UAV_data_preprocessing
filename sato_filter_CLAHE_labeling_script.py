import os
import cv2
import glob
import numpy as np

from skimage import io, color, img_as_float
from skimage.filters import sato, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects

# ============================
# 사용자 설정
# ============================
SRC_DIR = r"C:\Users\dromii\Downloads\20250710_ori_migum\crack_test_dataset" #라벨링할 이미지 경로
DST_DIR = r"C:\Users\dromii\Downloads\20250710_ori_migum\crack_test_dataset\crack_test_dataset_sato_CLAHE_mask" #마스크 저장 경로

WINDOW_MAIN = "Sato+CLAHE Mask (L: Original Only, R: Overlay on Gray/CLAHE)"
WINDOW_CTRL = "Control"

VIEW_SCALE = 0.6  # 메인창 표시 스케일(계산/저장은 원본 해상도)
# ============================

def list_images(folder):
    exts = ("*.jpg","*.jpeg","*.png","*.tif","*.tiff","*.bmp")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, e))
    return sorted(files)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def to_u8_gray(f01):
    return (np.clip(f01, 0, 1) * 255).astype(np.uint8)

def colorize_gray(f01, scale=1.0):
    g8 = to_u8_gray(f01)
    if scale != 1.0:
        g8 = cv2.resize(g8, (int(g8.shape[1]*scale), int(g8.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)

def overlay_mask(gray_f01, mask_u8, alpha=0.45, color=(0,255,255), scale=1.0):
    base = colorize_gray(gray_f01, scale=scale)
    if scale != 1.0:
        mu = cv2.resize(mask_u8, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mu = mask_u8
    out = base.copy()
    sel = mu > 0
    out[sel] = (out[sel]*(1-alpha) + np.array(color)*alpha).astype(np.uint8)
    return out

def put_label(img, text):
    out = img.copy()
    cv2.putText(out, text, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(out, text, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255,255,255), 1, cv2.LINE_AA)
    return out

class App:
    def __init__(self, files, dst_dir):
        self.files = files
        self.dst   = dst_dir
        ensure_dir(self.dst)

        self.idx = 0
        self.img = None
        self.gray = None      # 원본 gray (0..1)
        self.gray_proc = None # CLAHE 적용 시 여기에 결과 저장
        self.H = self.W = 0

        # 편집(지우개/복원) 마스크: 0=변경 없음, 255=유저가 '지움'
        self.erase_mask = None

        # Undo 스택: (changed_idx_bool, prev_vals_u8)
        self.undo_stack = []
        self._stroke_mask_tmp = None
        self._erase_before = None

        # 창/트랙바
        self.setup_windows()

        # 첫 이미지 로드
        self.load(self.files[self.idx])

    # ---------- 창/트랙바 ----------
    def setup_windows(self):
        cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_MAIN, 1600, 900)
        cv2.moveWindow(WINDOW_MAIN, 40, 40)

        cv2.namedWindow(WINDOW_CTRL, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_CTRL, 380, 520)
        cv2.moveWindow(WINDOW_CTRL, 40, 900)

        # Control 창 트랙바
        cv2.createTrackbar("SigmaMax (1..8)",   WINDOW_CTRL, 4, 8,    lambda v: None)
        cv2.createTrackbar("BlackRidges (0/1)", WINDOW_CTRL, 1, 1,    lambda v: None)

        cv2.createTrackbar("CLAHE on/off",      WINDOW_CTRL, 1, 1,    lambda v: None)
        cv2.createTrackbar("CLAHE clip x10",    WINDOW_CTRL, 20, 100, lambda v: None)  # 0.1..10.0
        cv2.createTrackbar("CLAHE tile k",      WINDOW_CTRL, 4, 20,   lambda v: None)  # 4,6,8,...

        cv2.createTrackbar("LowThr x1000",      WINDOW_CTRL, 100, 1000, lambda v: None)
        cv2.createTrackbar("HighThr x1000",     WINDOW_CTRL, 200, 1000, lambda v: None)

        cv2.createTrackbar("MinSize",           WINDOW_CTRL, 30, 5000,  lambda v: None)
        cv2.createTrackbar("UseClose (0/1)",    WINDOW_CTRL, 0, 1,       lambda v: None)
        cv2.createTrackbar("CloseK (odd 3..11)",WINDOW_CTRL, 1, 5,       lambda v: None)
        cv2.createTrackbar("Overlay alpha (0..100)", WINDOW_CTRL, 45, 100, lambda v: None)
        cv2.createTrackbar("Brush px",          WINDOW_CTRL, 14, 150,    lambda v: None)

        # 메인창 마우스
        cv2.setMouseCallback(WINDOW_MAIN, self.on_mouse_main)

    def get_params(self):
        sigma = max(1, cv2.getTrackbarPos("SigmaMax (1..8)", WINDOW_CTRL))
        black = bool(cv2.getTrackbarPos("BlackRidges (0/1)", WINDOW_CTRL))
        use_clahe = bool(cv2.getTrackbarPos("CLAHE on/off", WINDOW_CTRL))
        clip = max(1, cv2.getTrackbarPos("CLAHE clip x10", WINDOW_CTRL)) / 10.0
        tile = 2*max(1, cv2.getTrackbarPos("CLAHE tile k", WINDOW_CTRL)) + 2  # 4,6,8,...

        low  = cv2.getTrackbarPos("LowThr x1000", WINDOW_CTRL)  / 1000.0
        high = cv2.getTrackbarPos("HighThr x1000", WINDOW_CTRL) / 1000.0
        if high < low: high = min(1.0, low + 0.01)

        min_size  = cv2.getTrackbarPos("MinSize", WINDOW_CTRL)
        use_close = bool(cv2.getTrackbarPos("UseClose (0/1)", WINDOW_CTRL))
        close_k   = 2*max(1, cv2.getTrackbarPos("CloseK (odd 3..11)", WINDOW_CTRL)) + 1
        alpha     = cv2.getTrackbarPos("Overlay alpha (0..100)", WINDOW_CTRL) / 100.0
        brush     = max(1, cv2.getTrackbarPos("Brush px", WINDOW_CTRL))
        return dict(sigma=sigma, black=black, use_clahe=use_clahe, clip=clip, tile=tile,
                    low=low, high=high, min_size=min_size, use_close=use_close,
                    close_k=close_k, alpha=alpha, brush=brush)

    # ---------- 로드/세이브 ----------
    def load(self, path):
        img = io.imread(path)
        if img.ndim == 3: g = color.rgb2gray(img)
        else:             g = img_as_float(img)
        self.img = img
        self.gray = np.clip(g, 0, 1)
        self.H, self.W = self.gray.shape[:2]
        self.erase_mask = np.zeros((self.H, self.W), np.uint8)
        self.undo_stack.clear()
        self._stroke_mask_tmp = None
        self._erase_before = None

    def save(self):
        name = os.path.splitext(os.path.basename(self.files[self.idx]))[0] + ".png"
        out_path = os.path.join(self.dst, name)
        cv2.imwrite(out_path, self.final_mask())
        print(f"[✓] saved: {out_path}")

    # ---------- 처리 파이프라인 ----------
    def compute_gray_proc(self, p):
        if not p["use_clahe"]:
            return self.gray
        g8 = to_u8_gray(self.gray)
        clahe = cv2.createCLAHE(p["clip"], (p["tile"], p["tile"]))
        g8c = clahe.apply(g8)
        return g8c.astype(np.float32)/255.0

    def compute_base_mask(self, p, gray_src):
        sigmas = tuple(range(1, p["sigma"]+1)) or (1,)
        ridge = sato(gray_src, sigmas=sigmas, black_ridges=p["black"])
        rn = (ridge - ridge.min()) / (ridge.max() - ridge.min() + 1e-8)
        mask_bool = apply_hysteresis_threshold(rn, p["low"], p["high"])
        if p["min_size"] > 0:
            mask_bool = remove_small_objects(mask_bool, min_size=int(p["min_size"]))
        m8 = (mask_bool.astype(np.uint8) * 255)
        if p["use_close"]:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (p["close_k"], p["close_k"]))
            m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k, iterations=1)
        return rn, m8  # (정규화 응답, 기본 마스크)

    def final_mask(self):
        p = self.get_params()
        gray_src = self.compute_gray_proc(p)
        _, base = self.compute_base_mask(p, gray_src)
        # 최종 = base & (~erase)
        return cv2.bitwise_and(base, cv2.bitwise_not(self.erase_mask))

    # ---------- 그리기 ----------
    def draw(self):
        p = self.get_params()

        # 처리용 소스(마스크 생성용)
        gray_proc = self.compute_gray_proc(p)
        _, base = self.compute_base_mask(p, gray_proc)
        final = cv2.bitwise_and(base, cv2.bitwise_not(self.erase_mask))

        # 왼쪽: 항상 원본 (마스크/CLAHE 영향 없음)
        left = colorize_gray(self.gray, scale=VIEW_SCALE)
        left = put_label(left, "Left: Original (no overlay / no CLAHE)")

        # 오른쪽: CLAHE 토글 여부에 따라 Gray/CLAHE + Overlay
        right_base = gray_proc if p["use_clahe"] else self.gray
        right = overlay_mask(right_base, final, alpha=p["alpha"], scale=VIEW_SCALE)
        right = put_label(right, f"Right: {'CLAHE' if p['use_clahe'] else 'Gray'} + Overlay")

        panel = np.hstack([left, right])
        hint  = "[Mouse] L-drag: Erase  |  R-drag: Restore   [Keys] S:Save&Next  N:Next  P:Prev  Z:Undo  C:Clear  Q/Esc:Quit"
        panel = put_label(panel, hint)
        cv2.imshow(WINDOW_MAIN, panel)

    # ---------- 좌표/패널 ----------
    def panel_rects(self):
        disp_w = int(self.W * VIEW_SCALE)
        disp_h = int(self.H * VIEW_SCALE)
        left_rect  = (0, 0, disp_w, disp_h)
        right_rect = (disp_w, 0, disp_w, disp_h)
        return left_rect, right_rect

    def mouse_to_image_xy(self, x, y, rect):
        rx, ry, rw, rh = rect
        if not (rx <= x < rx+rw and ry <= y < ry+rh): return None
        lx = x - rx; ly = y - ry
        ix = int(lx / VIEW_SCALE); iy = int(ly / VIEW_SCALE)
        if 0 <= ix < self.W and 0 <= iy < self.H: return (ix, iy)
        return None

    # ---------- 지우개 편집 ----------
    def stroke_begin(self):
        self._stroke_mask_tmp = np.zeros((self.H, self.W), np.uint8)
        self._erase_before = self.erase_mask.copy()

    def apply_brush(self, ix, iy, radius, val):
        if ix is None: return
        cv2.circle(self._stroke_mask_tmp, (int(ix), int(iy)), int(radius), 255, -1, lineType=cv2.LINE_AA)
        if val == 255:
            cv2.circle(self.erase_mask, (int(ix), int(iy)), int(radius), 255, -1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(self.erase_mask, (int(ix), int(iy)), int(radius),   0, -1, lineType=cv2.LINE_AA)

    def stroke_end(self):
        if self._stroke_mask_tmp is None or self._erase_before is None: return
        changed = self._stroke_mask_tmp > 0
        if np.any(changed):
            prev_vals = self._erase_before[changed].copy()
            self.undo_stack.append((changed, prev_vals))
            if len(self.undo_stack) > 100: self.undo_stack.pop(0)
        self._stroke_mask_tmp = None
        self._erase_before = None

    def undo(self):
        if not self.undo_stack: return
        changed, prev = self.undo_stack.pop()
        self.erase_mask[changed] = prev

    # ---------- 마우스 콜백 ----------
    def on_mouse_main(self, event, mx, my, flags, param):
        p = self.get_params()
        brush = p["brush"]
        left_rect, right_rect = self.panel_rects()

        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.stroke_begin()

        if (event == cv2.EVENT_LBUTTONDOWN) or (event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON)):
            # LMB: erase (=255)
            ix = iy = None
            pos = self.mouse_to_image_xy(mx, my, left_rect)
            if pos is None:
                pos = self.mouse_to_image_xy(mx, my, right_rect)
            if pos is not None:
                ix, iy = pos
                self.apply_brush(ix, iy, brush, val=255)

        if (event == cv2.EVENT_RBUTTONDOWN) or (event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON)):
            # RMB: restore (=0)
            ix = iy = None
            pos = self.mouse_to_image_xy(mx, my, left_rect)
            if pos is None:
                pos = self.mouse_to_image_xy(mx, my, right_rect)
            if pos is not None:
                ix, iy = pos
                self.apply_brush(ix, iy, brush, val=0)

        if event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.stroke_end()

    # ---------- 루프 ----------
    def run(self):
        print("[*] Controls:")
        print("  Mouse: L-drag=Erase,  R-drag=Restore")
        print("  Keys : S=Save&Next  N=Next  P=Prev  Z=Undo  C=Clear  Q/Esc=Quit")
        while True:
            self.draw()
            k = cv2.waitKey(16) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('s'):
                self.save()
                if self.idx < len(self.files)-1:
                    self.idx += 1
                    self.load(self.files[self.idx])
                else:
                    print("[*] 마지막 이미지입니다.")
            elif k == ord('n'):
                if self.idx < len(self.files)-1:
                    self.idx += 1
                    self.load(self.files[self.idx])
                else:
                    print("[*] 마지막 이미지입니다.")
            elif k == ord('p'):
                if self.idx > 0:
                    self.idx -= 1
                    self.load(self.files[self.idx])
                else:
                    print("[*] 첫 이미지입니다.")
            elif k == ord('z'):
                self.undo()
            elif k == ord('c'):
                self.erase_mask[:] = 0
                self.undo_stack.clear()

        cv2.destroyAllWindows()

# ============================
# 실행
# ============================
if __name__ == "__main__":
    files = list_images(SRC_DIR)
    if not files:
        raise SystemExit("입력 폴더에 이미지가 없습니다.")
    app = App(files, DST_DIR)
    app.run()
