import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage import io, color, img_as_float
from skimage.filters import sato, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects
from albumentations import Normalize, Compose
from albumentations.pytorch import ToTensorV2

# ============================
# 사용자 설정
# ============================
SRC_DIR = r"C:\Users\dromii\Downloads\20250710_ori_migum\20250710_tiled_512\images"
DST_DIR = r"C:\Users\dromii\Downloads\20250710_ori_migum\20250710_tiled_512\crack_masks"

MODEL_PATH = r"C:\Users\dromii\Downloads\500_hc_unetpp_final_epoch_100.pth"
WINDOW_SIZE = (512, 512)
STRIDE = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2)

WINDOW_MAIN = "Sato+CLAHE + (Optional) Model Overlay  [L: Original Color, R: Color+Mask Overlay]"
WINDOW_CTRL = "Control (All Params Visible)"

VIEW_SCALE = 1.0  # 원본 100% 기본
SAVE_EMPTY_ON_SKIP = False
# ============================

# ----------------------------
# HC-Unet++ (추론용 경량 정의)
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class DPFFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        c = out_channels // 2
        self.upper = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
        self.lower = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return torch.cat([self.upper(x), self.lower(x)], dim=1)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x): return x * self.fc(self.pool(x))

class BlurPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])
        kernel = kernel / kernel.sum()
        self.register_buffer('filter', kernel[None,None,:,:].repeat(channels,1,1,1))
        self.channels = channels
    def forward(self, x): return F.conv2d(x, self.filter, stride=2, padding=1, groups=self.channels)

class HCUnetPlusPlus(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super().__init__()
        f = [64, 128, 256, 512, 1024]
        self.pool0 = BlurPool(f[0]); self.pool1 = BlurPool(f[1])
        self.pool2 = BlurPool(f[2]); self.pool3 = BlurPool(f[3])
        self.up1_0 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(f[2], f[2], 2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(f[3], f[3], 2, stride=2)
        self.up4_0 = nn.ConvTranspose2d(f[4], f[4], 2, stride=2)
        self.up1_1 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(f[2], f[2], 2, stride=2)
        self.up3_1 = nn.ConvTranspose2d(f[3], f[3], 2, stride=2)
        self.up1_2 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)
        self.up2_2 = nn.ConvTranspose2d(f[2], f[2], 2, stride=2)
        self.up1_3 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)

        self.conv0_0 = ConvBlock(input_channels, f[0], f[0])
        self.conv1_0 = ConvBlock(f[0], f[1], f[1])
        self.conv2_0 = ConvBlock(f[1], f[2], f[2])
        self.conv3_0 = ConvBlock(f[2], f[3], f[3])
        self.conv4_0 = DPFFB(f[3], f[4])

        self.conv0_1 = ConvBlock(f[0] + f[1], f[0], f[0])
        self.conv1_1 = ConvBlock(f[1] + f[2], f[1], f[1])
        self.conv2_1 = ConvBlock(f[2] + f[3], f[2], f[2])
        self.conv3_1 = ConvBlock(f[3] + f[4], f[3], f[3])

        self.conv0_2 = ConvBlock(f[0]*2 + f[1], f[0], f[0])
        self.conv1_2 = ConvBlock(f[1]*2 + f[2], f[1], f[1])
        self.conv2_2 = ConvBlock(f[2]*2 + f[3], f[2], f[2])

        self.conv0_3 = ConvBlock(f[0]*3 + f[1], f[0], f[0])
        self.conv1_3 = ConvBlock(f[1]*3 + f[2], f[1], f[1])
        self.conv0_4 = ConvBlock(f[0]*4 + f[1], f[0], f[0])

        self.se3_1 = SEBlock(f[3])
        self.output = nn.Conv2d(f[0], num_classes, 1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool0(x0_0))
        x2_0 = self.conv2_0(self.pool1(x1_0))
        x3_0 = self.conv3_0(self.pool2(x2_0))
        x4_0 = self.conv4_0(self.pool3(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))

        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], dim=1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))
        x3_1 = self.se3_1(x3_1)

        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], dim=1))

        return self.output(x0_4)

# ----------------------------
# 유틸
# ----------------------------
def list_images(folder):
    exts = ("*.jpg","*.jpeg","*.png","*.tif","*.tiff","*.bmp")
    files = []
    for e in exts: files += glob.glob(os.path.join(folder, e))
    return sorted(files)

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def to_u8_gray(f01): return (np.clip(f01, 0, 1) * 255).astype(np.uint8)

def colorize_gray(f01, scale=1.0):
    g8 = to_u8_gray(f01)
    if scale != 1.0:
        g8 = cv2.resize(g8, (int(g8.shape[1]*scale), int(g8.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)

def overlay_mask(gray_f01, mask_u8, alpha=0.45, color=(0,255,255), scale=1.0):
    base = colorize_gray(gray_f01, scale=scale)
    mu = cv2.resize(mask_u8, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST) if scale != 1.0 else mask_u8
    out = base.copy()
    sel = mu > 0
    out[sel] = (out[sel]*(1-alpha) + np.array(color)*alpha).astype(np.uint8)
    return out

def overlay_mask_on_bgr(bgr_img, mask_u8, alpha=0.45, color=(0,255,255), scale=1.0):
    base = bgr_img
    if scale != 1.0:
        base = cv2.resize(base, (int(base.shape[1]*scale), int(base.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    mu = cv2.resize(mask_u8, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST) if scale != 1.0 else mask_u8
    out = base.astype(np.float32)
    sel = mu > 0
    out[sel] = out[sel]*(1.0 - alpha) + np.array(color, dtype=np.float32)*alpha
    return np.clip(out, 0, 255).astype(np.uint8)

def bgr_scaled(bgr, scale=1.0):
    if scale == 1.0: return bgr
    return cv2.resize(bgr, (int(bgr.shape[1]*scale), int(bgr.shape[0]*scale)), interpolation=cv2.INTER_AREA)

def get_gaussian_kernel(window_size, sigma_scale=1./8):
    y, x = np.mgrid[0:window_size[0], 0:window_size[1]]
    cy, cx = (window_size[0]-1)/2., (window_size[1]-1)/2.
    sy, sx = window_size[0]*sigma_scale, window_size[1]*sigma_scale
    return np.exp(-((x-cx)**2/(2*sx**2) + (y-cy)**2/(2*sy**2)))

# ----------------------------
# 앱
# ----------------------------
class App:
    def __init__(self, files, dst_dir):
        self.files = files
        self.dst   = dst_dir
        ensure_dir(self.dst)

        self.done = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(self.dst, "*.png"))}
        self.base_to_index = {os.path.splitext(os.path.basename(p))[0]: i for i, p in enumerate(self.files)}

        self.idx = 0
        self.rgb = None
        self.bgr = None
        self.gray = None
        self.H = self.W = 0

        # 편집 마스크(최종 마스크용 지우개만)
        self.erase_mask = None
        self.undo_stack = []
        self._stroke_tmp = None
        self._erase_before = None

        # 모델 관련
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_loaded = False
        self.model_prob = None
        self.model_bin  = None
        self.model_cached_for = None
        self.model_thr_cached = None

        # 캐시
        self.gray_proc_cache = None; self.gray_proc_key = None
        self.sato_cache = None; self.sato_key = None

        # 보기 배율 상태
        self.view_scale = VIEW_SCALE

        # UI
        self.setup_windows()
        self.jump_to_first_unlabeled()  # ← 내부 로직을 '마지막 저장 이후'로 변경
        if 0 <= self.idx < len(self.files): self.load(self.files[self.idx])

    # ---------- 경로/상태 ----------
    def out_path_for(self, idx):
        name = os.path.splitext(os.path.basename(self.files[idx]))[0] + ".png"
        return os.path.join(self.dst, name)
    def out_path_current(self): return self.out_path_for(self.idx)
    def mask_exists_for(self, idx):
        base = os.path.splitext(os.path.basename(self.files[idx]))[0]
        return os.path.exists(self.out_path_for(idx)) or (base in self.done)

    def jump_to_first_unlabeled(self):
        """
        변경점:
        - DST_DIR 내 *.png 중 '수정시간이 가장 최근'인 마스크 파일을 찾아
          그 파일에 해당하는 원본의 '다음 인덱스'에서 시작.
        - 그 다음 인덱스가 이미 라벨링 되어 있다면, 최초 미라벨링 지점까지 전진.
        """
        mask_paths = glob.glob(os.path.join(self.dst, "*.png"))
        start = 0
        if mask_paths:
            last_mask = max(mask_paths, key=os.path.getmtime)  # 최신 저장본
            base = os.path.splitext(os.path.basename(last_mask))[0]
            last_idx = self.base_to_index.get(base, -1)
            start = last_idx + 1

        i = start
        while 0 <= i < len(self.files) and self.mask_exists_for(i):
            i += 1

        if 0 <= i < len(self.files):
            self.idx = i
            print(f"[*] Resuming AFTER last saved → #{self.idx+1}: {os.path.basename(self.files[self.idx])}")
            return

        self.idx = len(self.files)
        print("[*] 모든 이미지가 이미 라벨링되어 있습니다.]")

    def advance_to_next_unlabeled(self, direction=+1, include_current=False):
        if not self.files: return False
        i = self.idx if include_current else self.idx + direction
        while 0 <= i < len(self.files) and self.mask_exists_for(i): i += direction
        if 0 <= i < len(self.files): self.idx = i; self.load(self.files[self.idx]); return True
        print("[*] 더 이상 진행할 미라벨링 이미지가 없습니다."); return False

    # ---------- 상태바/오버레이 ----------
    def show_status(self, text, ms=2500):
        try:
            cv2.displayStatusBar(WINDOW_CTRL, text, ms)
        except Exception:
            try:
                cv2.displayOverlay(WINDOW_CTRL, text, ms)
            except Exception:
                print(text)

    def make_cb(self, name):
        def _cb(v, _n=name):
            self.show_status(f"{_n}: {v}")
        return _cb

    def show_help(self):
        help_text = (
            "Controls:\n"
            "  Mouse: L-drag=지움, R-drag=복원\n"
            "  Keys : S=Save&Next  N=Next  P=Prev  Z=Undo  C=Clear  M=Toggle Model  H=Help  Q/Esc=Quit\n"
            "Trackbars:\n"
            "  SigmaMax (1..8) / BlackRidges (0/1)\n"
            "  CLAHE enable (0/1), CLAHE clip*10 (0.1~10.0), CLAHE tile k (4,6,...)\n"
            "  Hysteresis Low/High *1000, MinSize, Morph Close, CloseK\n"
            "  Overlay alpha, Brush size, Use Model, Model Thr*1000\n"
            "  View scale % (25~200)\n"
        )
        self.show_status(help_text, 0)

    # ---------- 창/트랙바 ----------
    def setup_windows(self):
        cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_MAIN, 1680, 980)

        cv2.namedWindow(WINDOW_CTRL, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_CTRL, 380, 560)
        cv2.moveWindow(WINDOW_CTRL, 40, 900)

        self.show_help()

        cv2.createTrackbar("SigmaMax (1..8)",            WINDOW_CTRL, 4,   8,    self.make_cb("SigmaMax (1..8)"))
        cv2.createTrackbar("BlackRidges (0/1)",          WINDOW_CTRL, 1,   1,    self.make_cb("BlackRidges (0/1)"))
        cv2.createTrackbar("CLAHE enable (0/1)",         WINDOW_CTRL, 1,   1,    self.make_cb("CLAHE enable (0/1)"))
        cv2.createTrackbar("CLAHE clip *10 (0.1~10.0)",  WINDOW_CTRL, 20,  100,  self.make_cb("CLAHE clip*10"))
        cv2.createTrackbar("CLAHE tile k (4,6,...)",     WINDOW_CTRL, 4,   20,   self.make_cb("CLAHE tile k"))

        cv2.createTrackbar("Hysteresis Low *1000",       WINDOW_CTRL, 100, 1000, self.make_cb("Hysteresis Low*1000"))
        cv2.createTrackbar("Hysteresis High *1000",      WINDOW_CTRL, 200, 1000, self.make_cb("Hysteresis High*1000"))

        cv2.createTrackbar("MinSize (remove_small_objects)", WINDOW_CTRL, 30, 5000, self.make_cb("MinSize"))
        cv2.createTrackbar("Morph Close enable (0/1)",   WINDOW_CTRL, 0,   1,    self.make_cb("Morph Close enable"))
        cv2.createTrackbar("CloseK (odd 3..11)",         WINDOW_CTRL, 1,   5,    self.make_cb("CloseK (odd 3..11)"))
        cv2.createTrackbar("Overlay alpha (0..100)",     WINDOW_CTRL, 45,  100,  self.make_cb("Overlay alpha"))
        cv2.createTrackbar("Brush size px",              WINDOW_CTRL, 14,  150,  self.make_cb("Brush size px"))

        cv2.createTrackbar("Use Model (0/1)",            WINDOW_CTRL, 0,   1,    self.make_cb("Use Model (0/1)"))
        cv2.createTrackbar("Model Thr *1000",            WINDOW_CTRL, 500, 1000, self.make_cb("Model Thr *1000"))

        cv2.createTrackbar("View scale % (25~200)",      WINDOW_CTRL, int(VIEW_SCALE*100), 200, self.make_cb("View scale %"))

        cv2.setMouseCallback(WINDOW_MAIN, self.on_mouse_main)

    def get_params(self):
        sigma = max(1, cv2.getTrackbarPos("SigmaMax (1..8)", WINDOW_CTRL))
        black = bool(cv2.getTrackbarPos("BlackRidges (0/1)", WINDOW_CTRL))
        use_clahe = bool(cv2.getTrackbarPos("CLAHE enable (0/1)", WINDOW_CTRL))
        clip = max(1, cv2.getTrackbarPos("CLAHE clip *10 (0.1~10.0)", WINDOW_CTRL)) / 10.0
        tile = 2*max(1, cv2.getTrackbarPos("CLAHE tile k (4,6,...)", WINDOW_CTRL)) + 2

        low  = cv2.getTrackbarPos("Hysteresis Low *1000", WINDOW_CTRL)  / 1000.0
        high = cv2.getTrackbarPos("Hysteresis High *1000", WINDOW_CTRL) / 1000.0
        if high < low: high = min(1.0, low + 0.01)

        min_size  = cv2.getTrackbarPos("MinSize (remove_small_objects)", WINDOW_CTRL)
        use_close = bool(cv2.getTrackbarPos("Morph Close enable (0/1)", WINDOW_CTRL))
        close_k   = 2*max(1, cv2.getTrackbarPos("CloseK (odd 3..11)", WINDOW_CTRL)) + 1
        alpha     = cv2.getTrackbarPos("Overlay alpha (0..100)", WINDOW_CTRL) / 100.0
        brush     = max(1, cv2.getTrackbarPos("Brush size px", WINDOW_CTRL))

        use_model = bool(cv2.getTrackbarPos("Use Model (0/1)", WINDOW_CTRL))
        mthr      = cv2.getTrackbarPos("Model Thr *1000", WINDOW_CTRL) / 1000.0

        vs = max(25, cv2.getTrackbarPos("View scale % (25~200)", WINDOW_CTRL))
        view_scale = vs / 100.0

        return dict(
            sigma=sigma, black=black, use_clahe=use_clahe, clip=clip, tile=tile,
            low=low, high=high, min_size=min_size, use_close=use_close,
            close_k=close_k, alpha=alpha, brush=brush,
            use_model=use_model, mthr=mthr, view_scale=view_scale
        )

    # ---------- 로드 ----------
    def load(self, path):
        img = io.imread(path)
        if img.ndim == 3:
            self.rgb = img
            self.gray = np.clip(color.rgb2gray(img), 0, 1)
            self.bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            self.rgb = None
            self.gray = img_as_float(img)
            self.bgr = cv2.cvtColor(to_u8_gray(self.gray), cv2.COLOR_GRAY2BGR)

        self.H, self.W = self.gray.shape[:2]
        self.erase_mask = np.zeros((self.H, self.W), np.uint8)
        self.undo_stack.clear()
        self._stroke_tmp = None
        self._erase_before = None

        # 캐시 초기화
        self.gray_proc_cache = None; self.gray_proc_key = None
        self.sato_cache = None; self.sato_key = None
        self.model_prob = None; self.model_bin = None
        self.model_cached_for = None; self.model_thr_cached = None

        try:
            cv2.setWindowTitle(WINDOW_MAIN,
                f"{WINDOW_MAIN}   [{self.idx+1}/{len(self.files)}] "
                f"in: {os.path.basename(self.files[self.idx])}  →  out: {os.path.basename(self.out_path_current())}")
        except Exception:
            pass

    # ---------- 저장 ----------
    def save_final(self):
        p = self.get_params()
        gray_proc = self.compute_gray_proc_cached(p)
        _, base_sato = self.compute_sato_cached(p, gray_proc)
        base_combined = self.combine_with_model(p, base_sato)
        final = self.compose_final(base_combined)
        out_path = self.out_path_current()
        ok = cv2.imwrite(out_path, final)
        if ok:
            base = os.path.splitext(os.path.basename(self.files[self.idx]))[0]
            self.done.add(base); print(f"[✓] saved: {os.path.abspath(out_path)}")
        else:
            print(f"[!] save failed: {os.path.abspath(out_path)}")

    def save_empty(self):
        out_path = self.out_path_current()
        if not os.path.exists(out_path):
            empty = np.zeros((self.H, self.W), np.uint8)
            ok = cv2.imwrite(out_path, empty)
            if ok:
                base = os.path.splitext(os.path.basename(self.files[self.idx]))[0]
                self.done.add(base); print(f"[✓] saved empty: {os.path.abspath(out_path)}")
            else:
                print(f"[!] save empty failed: {os.path.abspath(out_path)}")

    # ---------- 캐시 기반 처리 ----------
    def compute_gray_proc_cached(self, p):
        key = (p["use_clahe"], round(p["clip"],3), p["tile"])
        if self.gray_proc_key != key:
            if not p["use_clahe"]:
                self.gray_proc_cache = self.gray
            else:
                g8 = to_u8_gray(self.gray)
                clahe = cv2.createCLAHE(p["clip"], (p["tile"], p["tile"]))
                g8c = clahe.apply(g8)
                self.gray_proc_cache = g8c.astype(np.float32)/255.0
            self.gray_proc_key = key
        return self.gray_proc_cache

    def compute_sato_cached(self, p, gray_src):
        key = (p["sigma"], p["black"], round(p["low"],3), round(p["high"],3),
               int(p["min_size"]), p["use_close"], p["close_k"], self.gray_proc_key)
        if self.sato_key != key:
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
            self.sato_cache = (rn, m8)
            self.sato_key = key
        return self.sato_cache

    # ---------- 모델 ----------
    def lazy_load_model(self):
        if self.model_loaded: return True
        if not os.path.exists(MODEL_PATH):
            print(f"[!] MODEL_PATH not found: {MODEL_PATH}")
            return False
        try:
            model = HCUnetPlusPlus(num_classes=1, input_channels=3).to(self.device)
            state = torch.load(MODEL_PATH, map_location=self.device)
            model.load_state_dict(state); model.eval()
            self.model = model; self.model_loaded = True
            print(f"[✓] Model loaded on {self.device}: {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"[!] Failed to load model: {e}"); return False

    def predict_prob_once(self):
        if self.model_cached_for == self.files[self.idx] and self.model_prob is not None:
            return self.model_prob
        transform = Compose([Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])
        image_rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape
        pad_h = max(WINDOW_SIZE[0]-h, 0); pad_w = max(WINDOW_SIZE[1]-w, 0)
        padded = np.pad(image_rgb, ((0,pad_h),(0,pad_w),(0,0)), mode='constant', constant_values=0)
        ph, pw, _ = padded.shape

        out = np.zeros((ph, pw), np.float32)
        cnt = np.zeros((ph, pw), np.float32)
        gkern = get_gaussian_kernel(WINDOW_SIZE)

        y_steps = list(range(0, ph - WINDOW_SIZE[0] + 1, STRIDE[0]))
        if y_steps[-1] != ph - WINDOW_SIZE[0]: y_steps.append(ph - WINDOW_SIZE[0])
        x_steps = list(range(0, pw - WINDOW_SIZE[1] + 1, STRIDE[1]))
        if x_steps[-1] != pw - WINDOW_SIZE[1]: x_steps.append(pw - WINDOW_SIZE[1])

        with torch.no_grad():
            for y in y_steps:
                for x in x_steps:
                    crop = padded[y:y+WINDOW_SIZE[0], x:x+WINDOW_SIZE[1]]
                    inp = transform(image=crop)['image'].unsqueeze(0).to(self.device)
                    prob = torch.sigmoid(self.model(inp)).squeeze().detach().cpu().numpy()
                    out[y:y+WINDOW_SIZE[0], x:x+WINDOW_SIZE[1]] += prob * gkern
                    cnt[y:y+WINDOW_SIZE[0], x:x+WINDOW_SIZE[1]] += gkern

        prob_map = (out / np.maximum(cnt, 1e-6))[:h, :w].astype(np.float32)
        self.model_prob = prob_map
        self.model_cached_for = self.files[self.idx]
        return prob_map

    def ensure_model_bin(self, p):
        if not p["use_model"]:
            self.model_bin = None; return None
        if not self.model_loaded and not self.lazy_load_model():
            self.model_bin = None; return None
        try:
            prob = self.predict_prob_once()
        except Exception as e:
            print(f"[!] Model inference failed: {e}")
            self.model_bin = None
            return None
        if self.model_thr_cached != p["mthr"] or self.model_bin is None:
            self.model_bin = (prob > float(np.clip(p["mthr"], 0.0, 1.0))).astype(np.uint8) * 255
            self.model_thr_cached = p["mthr"]
        return self.model_bin

    # ---------- 결합/최종 ----------
    def combine_with_model(self, p, sato_base):
        m = self.ensure_model_bin(p)
        if m is None: return sato_base
        return cv2.bitwise_or(sato_base, m)

    def compose_final(self, base_combined):
        return cv2.bitwise_and(base_combined, cv2.bitwise_not(self.erase_mask))

    # ---------- 그리기 ----------
    def draw(self):
        if not (0 <= self.idx < len(self.files)):
            print("[*] 작업을 종료합니다. (모든 이미지 완료)")
            cv2.destroyAllWindows(); raise SystemExit

        if self.mask_exists_for(self.idx):
            moved = self.advance_to_next_unlabeled(direction=+1, include_current=True)
            if not moved:
                print("[*] 작업을 종료합니다. (모든 이미지 완료)")
                cv2.destroyAllWindows(); raise SystemExit
            return

        p = self.get_params()
        self.view_scale = p["view_scale"]

        gray_proc = self.compute_gray_proc_cached(p)
        _, base_sato = self.compute_sato_cached(p, gray_proc)
        base_combined = self.combine_with_model(p, base_sato)
        final = self.compose_final(base_combined)

        left  = bgr_scaled(self.bgr, self.view_scale)
        right = overlay_mask_on_bgr(self.bgr, final, alpha=p["alpha"], scale=self.view_scale)

        panel = np.hstack([left, right])
        cv2.imshow(WINDOW_MAIN, panel)

    # ---------- 좌표/편집 ----------
    def mouse_to_image_xy(self, x, y, rect):
        rx, ry, rw, rh = rect
        if not (rx <= x < rx+rw and ry <= y < ry+rh): return None
        lx = x - rx; ly = y - ry
        ix = int(lx / self.view_scale); iy = int(ly / self.view_scale)
        if 0 <= ix < self.W and 0 <= iy < self.H: return (ix, iy)
        return None

    def on_mouse_main(self, event, mx, my, flags, param):
        p = self.get_params(); brush = p["brush"]
        disp_w = int(self.W * self.view_scale)
        left_rect  = (0, 0, disp_w, int(self.H * self.view_scale))
        right_rect = (disp_w, 0, disp_w, int(self.H * self.view_scale))

        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self._stroke_tmp = np.zeros((self.H, self.W), np.uint8)
            self._erase_before = self.erase_mask.copy()

        if (event == cv2.EVENT_LBUTTONDOWN) or (event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON)):
            pos = self.mouse_to_image_xy(mx, my, left_rect) or self.mouse_to_image_xy(mx, my, right_rect)
            if pos is not None:
                cv2.circle(self._stroke_tmp, pos, brush, 255, -1, lineType=cv2.LINE_AA)
                cv2.circle(self.erase_mask, pos, brush, 255, -1, lineType=cv2.LINE_AA)

        if (event == cv2.EVENT_RBUTTONDOWN) or (event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON)):
            pos = self.mouse_to_image_xy(mx, my, left_rect) or self.mouse_to_image_xy(mx, my, right_rect)
            if pos is not None:
                cv2.circle(self._stroke_tmp, pos, brush, 255, -1, lineType=cv2.LINE_AA)
                cv2.circle(self.erase_mask, pos, brush, 0,   -1, lineType=cv2.LINE_AA)

        if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            if self._stroke_tmp is not None and self._erase_before is not None:
                changed = self._stroke_tmp > 0
                if np.any(changed):
                    prev_vals = self._erase_before[changed].copy()
                    self.undo_stack.append(("erase", changed, prev_vals))
                    if len(self.undo_stack) > 200: self.undo_stack.pop(0)
            self._stroke_tmp = None; self._erase_before = None

    def undo(self):
        if not self.undo_stack: return
        target, changed, prev = self.undo_stack.pop()
        if target == "erase": self.erase_mask[changed] = prev

    # ---------- 루프 ----------
    def run(self):
        print("[*] Controls:")
        print("  Mouse: L-drag=지움(erase), R-drag=복원(restore)")
        print("  Keys : S=Save&Next  N=Next(unlabeled)  P=Prev(unlabeled)  Z=Undo  C=Clear(erase)  M=Toggle Model  H=Help  Q/Esc=Quit")
        while True:
            self.draw()
            k = cv2.waitKey(16) & 0xFF
            if k in (ord('q'), 27): break
            elif k == ord('s'):
                if self.mask_exists_for(self.idx):
                    print("[*] 이미 저장된 마스크가 있어 저장을 생략합니다.")
                    if not self.advance_to_next_unlabeled(+1): break
                else:
                    self.save_final()
                    if not self.advance_to_next_unlabeled(+1): break
            elif k == ord('n'):
                if SAVE_EMPTY_ON_SKIP and not self.mask_exists_for(self.idx): self.save_empty()
                if not self.advance_to_next_unlabeled(+1): break
            elif k == ord('p'):
                if SAVE_EMPTY_ON_SKIP and not self.mask_exists_for(self.idx): self.save_empty()
                if not self.advance_to_next_unlabeled(-1): break
            elif k == ord('z'):
                self.undo()
            elif k == ord('c'):
                self.erase_mask[:] = 0; self.undo_stack.clear(); print("[*] Erase mask cleared.")
            elif k == ord('m'):
                cur = cv2.getTrackbarPos("Use Model (0/1)", WINDOW_CTRL)
                cv2.setTrackbarPos("Use Model (0/1)", WINDOW_CTRL, 0 if cur==1 else 1)
            elif k == ord('h'):
                self.show_help()

        cv2.destroyAllWindows()

# ============================
# 실행
# ============================
if __name__ == "__main__":
    files = list_images(SRC_DIR)
    if not files: raise SystemExit("입력 폴더에 이미지가 없습니다.")
    app = App(files, DST_DIR)
    app.run()
