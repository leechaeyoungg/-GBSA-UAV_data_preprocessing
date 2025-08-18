import os
import cv2
import numpy as np

from skimage import io, color, img_as_float
from skimage.filters import sato, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects

# ========= 사용자 설정 =========
IMAGE_PATH = r"C:\Users\dromii\Downloads\20250710_ori_migum\crack_test_dataset\DJI_0192_tile_y0_x1943.jpg"
SAVE_DIR   = os.path.dirname(IMAGE_PATH)
SAVE_NAME  = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

WINDOW     = "Sato + Hysteresis (Realtime) + CLAHE"
VIEW_SCALE = 0.6    # 화면 표시만 축소(계산/저장은 원본)
# =================================

os.makedirs(SAVE_DIR, exist_ok=True)

# 1) 이미지 로드 → Gray(float 0~1)
img = io.imread(IMAGE_PATH)
if img.ndim == 3:
    img_gray = color.rgb2gray(img)        # float64, 0~1
else:
    img_gray = img_as_float(img)
H, W = img_gray.shape[:2]

# 화면 표시용 변환
def to_view_gray(gray_f01):
    g8 = (np.clip(gray_f01, 0, 1) * 255).astype(np.uint8)
    if VIEW_SCALE != 1.0:
        g8 = cv2.resize(g8, (int(W*VIEW_SCALE), int(H*VIEW_SCALE)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)

def colorize_gray(g):
    g8 = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    if VIEW_SCALE != 1.0:
        g8 = cv2.resize(g8, (int(W*VIEW_SCALE), int(H*VIEW_SCALE)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)

def overlay_mask_on_gray(gray_f01, mask_u8, alpha=0.45, color=(0,255,255)):
    gray_bgr = colorize_gray(gray_f01)
    if VIEW_SCALE != 1.0:
        mask_u8_view = cv2.resize(mask_u8, (gray_bgr.shape[1], gray_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask_u8_view = mask_u8
    out = gray_bgr.copy()
    m = mask_u8_view > 0
    out[m] = (out[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out

# 전역 캐시(속도)
_cached = {
    "sigma_max": None, "black": None,
    "use_clahe": None, "clip": None, "tile_w": None, "tile_h": None,
    "ridge_norm": None, "gray_src": None  # gray_src: 이번 Sato 계산에 사용된 Gray[0..1]
}

def get_gray_source(use_clahe, clip_limit, tile_w, tile_h):
    """현재 트랙바 설정에 따라 Gray 소스(원본/CLAHE)를 반환 (float[0..1])"""
    if not use_clahe:
        return img_gray
    # CLAHE 적용
    gray_u8 = (np.clip(img_gray, 0, 1) * 255).astype(np.uint8)
    # tile size는 최소 2 보장, clipLimit은 최소 0.1 정도 권장
    tile_w = max(2, int(tile_w))
    tile_h = max(2, int(tile_h))
    clip_limit = max(0.1, float(clip_limit))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_w, tile_h))
    gray_cla = clahe.apply(gray_u8)                 # uint8
    return gray_cla.astype(np.float32) / 255.0      # float [0..1]

def compute_sato_norm(sigma_max, black_ridges, use_clahe, clip_limit, tile_w, tile_h):
    """현재 설정으로 Sato 응답 정규화(0..1)와 사용 Gray를 계산/캐시"""
    same = (
        _cached["sigma_max"] == sigma_max and
        _cached["black"]     == black_ridges and
        _cached["use_clahe"] == use_clahe and
        _cached["clip"]      == clip_limit and
        _cached["tile_w"]    == tile_w and
        _cached["tile_h"]    == tile_h and
        _cached["ridge_norm"] is not None and
        _cached["gray_src"] is not None
    )
    if same:
        return _cached["ridge_norm"], _cached["gray_src"]

    # 소스 gray 선정
    gray_src = get_gray_source(use_clahe, clip_limit, tile_w, tile_h)

    # sigmas: 1..sigma_max
    sigmas = tuple(range(1, int(max(1, sigma_max)) + 1))
    ridge = sato(gray_src, sigmas=sigmas, black_ridges=bool(black_ridges))
    ridge_norm = (ridge - ridge.min()) / (ridge.max() - ridge.min() + 1e-8)

    _cached.update({
        "sigma_max": sigma_max, "black": black_ridges,
        "use_clahe": use_clahe, "clip": clip_limit, "tile_w": tile_w, "tile_h": tile_h,
        "ridge_norm": ridge_norm, "gray_src": gray_src
    })
    return ridge_norm, gray_src

def hysteresis_mask(ridge_norm, low_thr, high_thr, min_size, use_close=False, close_ksize=3):
    mask_bool = apply_hysteresis_threshold(ridge_norm, low_thr, high_thr)
    if min_size > 0:
        mask_bool = remove_small_objects(mask_bool, min_size=int(min_size))
    if use_close:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
        m8 = (mask_bool.astype(np.uint8) * 255)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k, iterations=1)
        mask_bool = m8 > 0
    return (mask_bool.astype(np.uint8) * 255)

# ---- UI ----
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 1500, 800)

# Sato 파라미터
cv2.createTrackbar("SigmaMax (1..8)",   WINDOW, 4, 8,   lambda v: None)
cv2.createTrackbar("BlackRidges 0/1",   WINDOW, 1, 1,   lambda v: None)
cv2.createTrackbar("LowThr x1000",      WINDOW, 100, 1000, lambda v: None)  # 0..1
cv2.createTrackbar("HighThr x1000",     WINDOW, 200, 1000, lambda v: None)  # 0..1
cv2.createTrackbar("MinSize",           WINDOW, 30, 2000, lambda v: None)
cv2.createTrackbar("UseClose 0/1",      WINDOW, 0, 1,   lambda v: None)
cv2.createTrackbar("CloseK (1..5)",     WINDOW, 1, 5,   lambda v: None)     # kernel=2*k+1

# CLAHE 파라미터
cv2.createTrackbar("UseCLAHE 0/1",      WINDOW, 0, 1,   lambda v: None)
cv2.createTrackbar("CLAHE clip x10",    WINDOW, 20, 200, lambda v: None)    # clip=val/10 => 0.1..20.0
cv2.createTrackbar("CLAHE tile W",      WINDOW, 8, 64,  lambda v: None)
cv2.createTrackbar("CLAHE tile H",      WINDOW, 8, 64,  lambda v: None)

print("[*] Controls:")
print("  Sato: SigmaMax, BlackRidges, Low/HighThr, MinSize, UseClose, CloseK")
print("  CLAHE: UseCLAHE, clip x10(=clipLimit/10), tile W/H (>=2)")
print("  S: 저장 (response/마스크),  Q/ESC: 종료")

while True:
    # Sato
    sigma_max = max(1, cv2.getTrackbarPos("SigmaMax (1..8)", WINDOW))
    black     = cv2.getTrackbarPos("BlackRidges 0/1", WINDOW)

    low_thr   = cv2.getTrackbarPos("LowThr x1000", WINDOW)  / 1000.0
    high_thr  = cv2.getTrackbarPos("HighThr x1000", WINDOW) / 1000.0
    if high_thr < low_thr:  # 안전장치
        high_thr = min(1.0, low_thr + 0.01)

    min_size  = cv2.getTrackbarPos("MinSize", WINDOW)
    use_close = bool(cv2.getTrackbarPos("UseClose 0/1", WINDOW))
    close_k   = 2 * max(1, cv2.getTrackbarPos("CloseK (1..5)", WINDOW)) + 1

    # CLAHE
    use_clahe = bool(cv2.getTrackbarPos("UseCLAHE 0/1", WINDOW))
    clip_lim  = cv2.getTrackbarPos("CLAHE clip x10", WINDOW) / 10.0
    tile_w    = max(2, cv2.getTrackbarPos("CLAHE tile W", WINDOW))
    tile_h    = max(2, cv2.getTrackbarPos("CLAHE tile H", WINDOW))

    # Sato 응답/Gray소스 계산(캐시)
    ridge_norm, gray_src = compute_sato_norm(
        sigma_max=sigma_max, black_ridges=black,
        use_clahe=use_clahe, clip_limit=clip_lim,
        tile_w=tile_w, tile_h=tile_h
    )

    # 히스테리시스 → 이진 마스크
    mask_u8 = hysteresis_mask(ridge_norm, low_thr, high_thr, min_size,
                              use_close=use_close, close_ksize=close_k)

    # 패널 구성: Gray(src) | Sato response | Overlay
    gray_panel  = to_view_gray(gray_src)
    resp_panel  = colorize_gray(ridge_norm)
    overlay_pan = overlay_mask_on_gray(gray_src, mask_u8, alpha=0.45)

    # 라벨
    def put_label(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        return out

    gray_name = f"Gray ({'CLAHE' if use_clahe else 'orig'})"
    gray_panel  = put_label(gray_panel,  gray_name + f" clip={clip_lim:.1f}, tile={tile_w}x{tile_h}" if use_clahe else gray_name)
    resp_panel  = put_label(resp_panel,  f"Sato resp (sig<= {sigma_max}, black={black})")
    overlay_pan = put_label(overlay_pan, f"Mask (low={low_thr:.2f}, high={high_thr:.2f}, min={min_size}, close={int(use_close)}/{close_k})")

    panel = np.hstack([gray_panel, resp_panel, overlay_pan])
    cv2.imshow(WINDOW, panel)

    key = cv2.waitKey(16) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key == ord('s'):
        # 참고용 응답(8bit) + CVAT용 이진 마스크 저장 (현재 소스 기준)
        resp_u8 = (np.clip(ridge_norm, 0, 1) * 255).astype(np.uint8)
        suffix  = f"{'clahe' if use_clahe else 'orig'}_sig{sigma_max}_b{black}_low{low_thr:.2f}_high{high_thr:.2f}"
        if use_clahe:
            suffix += f"_clip{clip_lim:.1f}_tile{tile_w}x{tile_h}"
        out_resp = os.path.join(SAVE_DIR, f"{SAVE_NAME}_sato_response_{suffix}.png")
        out_mask = os.path.join(SAVE_DIR, f"{SAVE_NAME}_sato_mask_hysteresis_{suffix}.png")
        cv2.imwrite(out_resp, resp_u8)
        cv2.imwrite(out_mask, mask_u8)
        print(f"[✓] saved:\n  - {out_resp}\n  - {out_mask}")

cv2.destroyAllWindows()
