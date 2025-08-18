import os
import cv2
import numpy as np

from skimage import io, color, img_as_float
from skimage.filters import sato, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects

# ========= 사용자 설정 =========
IMAGE_PATH = r"C:\Users\dromii\Downloads\20250710_ori_migum\crack_test_dataset\DJI_0193_tile_y0_x2867.jpg"
SAVE_DIR   = os.path.dirname(IMAGE_PATH)
SAVE_NAME  = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

WINDOW     = "Sato + Hysteresis (Realtime)"
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

# 화면 표시용 축소본
def to_view(img_gray):
    g8 = (np.clip(img_gray, 0, 1) * 255).astype(np.uint8)
    if VIEW_SCALE != 1.0:
        g8 = cv2.resize(g8, (int(W*VIEW_SCALE), int(H*VIEW_SCALE)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)

base_view = to_view(img_gray)

# 전역 캐시(속도): 현재 sigma/black 설정에서의 Sato 결과
_cached = {"sigma_max": None, "black": None, "ridge_norm": None}

def compute_sato_norm(sigma_max, black_ridges):
    """sigma_max, black_ridges가 바뀔 때만 재계산"""
    if _cached["sigma_max"] == sigma_max and _cached["black"] == black_ridges and _cached["ridge_norm"] is not None:
        return _cached["ridge_norm"]

    # sigmas: 1..sigma_max (정수)
    sigmas = tuple(range(1, int(sigma_max)+1))
    if len(sigmas) == 0:
        sigmas = (1,)

    ridge = sato(img_gray, sigmas=sigmas, black_ridges=bool(black_ridges))
    ridge_norm = (ridge - ridge.min()) / (ridge.max() - ridge.min() + 1e-8)

    _cached["sigma_max"] = sigma_max
    _cached["black"] = black_ridges
    _cached["ridge_norm"] = ridge_norm
    return ridge_norm

def hysteresis_mask(ridge_norm, low_thr, high_thr, min_size, use_close=False, close_ksize=3):
    # 히스테리시스 임계
    mask_bool = apply_hysteresis_threshold(ridge_norm, low_thr, high_thr)

    # 소형 영역 제거
    if min_size > 0:
        mask_bool = remove_small_objects(mask_bool, min_size=int(min_size))

    # (선택) CLOSE 1회
    if use_close:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
        m8 = (mask_bool.astype(np.uint8) * 255)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k, iterations=1)
        mask_bool = m8 > 0

    return (mask_bool.astype(np.uint8) * 255)

def colorize_gray(g):
    """시각화를 위해 GRAY(0~1) → BGR 8bit"""
    g8 = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    if VIEW_SCALE != 1.0:
        g8 = cv2.resize(g8, (int(W*VIEW_SCALE), int(H*VIEW_SCALE)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)

def overlay_mask_on_gray(gray_f01, mask_u8, alpha=0.45, color=(0,255,255)):
    """gray 위에 mask 오버레이"""
    gray_bgr = colorize_gray(gray_f01)
    if VIEW_SCALE != 1.0:
        mask_u8_view = cv2.resize(mask_u8, (gray_bgr.shape[1], gray_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask_u8_view = mask_u8
    out = gray_bgr.copy()
    m = mask_u8_view > 0
    out[m] = (out[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out

# ---- UI ----
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 1400, 700)

# 트랙바: 값 범위를 정수로 두고, 내부에서 0~1로 매핑
cv2.createTrackbar("SigmaMax (1..8)", WINDOW, 4, 8, lambda v: None)          # 1~8
cv2.createTrackbar("BlackRidges (0/1)", WINDOW, 1, 1, lambda v: None)        # 0 or 1

cv2.createTrackbar("LowThr x1000", WINDOW, 100, 1000, lambda v: None)        # 0..1 => /1000
cv2.createTrackbar("HighThr x1000", WINDOW, 200, 1000, lambda v: None)       # 0..1 => /1000

cv2.createTrackbar("MinSize", WINDOW, 30, 2000, lambda v: None)              # 픽셀 수
cv2.createTrackbar("UseClose (0/1)", WINDOW, 0, 1, lambda v: None)           # 0 or 1
cv2.createTrackbar("CloseK (odd 3..11)", WINDOW, 1, 5, lambda v: None)       # k=1..5 => kernel=2*k+1
cv2.createTrackbar("Overlay alpha (0..100)", WINDOW, 45, 100, lambda v: None)

print("[*] Controls:")
print("  - SigmaMax: Sato 스케일 상한(1~8)")
print("  - BlackRidges: 1이면 어두운 선을 ridge로")
print("  - Low/HighThr: 히스테리시스 임계(0~1). Low < High 권장")
print("  - MinSize: 작은 조각 제거(픽셀)")
print("  - UseClose: 1이면 CLOSE 1회(미세 연결)")
print("  - CloseK: CLOSE 커널(2*k+1 → 3,5,7,9,11)")
print("  - S: 저장,  Q/ESC: 종료")

while True:
    sigma_max = max(1, cv2.getTrackbarPos("SigmaMax (1..8)", WINDOW))
    black     = cv2.getTrackbarPos("BlackRidges (0/1)", WINDOW)

    low_thr   = cv2.getTrackbarPos("LowThr x1000", WINDOW) / 1000.0
    high_thr  = cv2.getTrackbarPos("HighThr x1000", WINDOW) / 1000.0
    # 안전장치: high < low면 약간 올려줌
    if high_thr < low_thr:
        high_thr = min(1.0, low_thr + 0.01)

    min_size  = cv2.getTrackbarPos("MinSize", WINDOW)
    use_close = bool(cv2.getTrackbarPos("UseClose (0/1)", WINDOW))
    close_k   = 2 * max(1, cv2.getTrackbarPos("CloseK (odd 3..11)", WINDOW)) + 1
    alpha     = cv2.getTrackbarPos("Overlay alpha (0..100)", WINDOW) / 100.0

    # Sato 응답(정규화) 계산/캐시
    ridge_norm = compute_sato_norm(sigma_max=sigma_max, black_ridges=black)

    # 히스테리시스 → 이진 마스크
    mask_u8 = hysteresis_mask(ridge_norm, low_thr, high_thr, min_size,
                              use_close=use_close, close_ksize=close_k)

    # 패널 구성: Gray | Sato response | Overlay
    gray_panel  = base_view
    resp_panel  = colorize_gray(ridge_norm)
    overlay_pan = overlay_mask_on_gray(img_gray, mask_u8, alpha=alpha)

    # 텍스트 상태표시
    def put_label(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        return out

    gray_panel  = put_label(gray_panel,  "Gray")
    resp_panel  = put_label(resp_panel,  f"Sato resp (sig<= {sigma_max}, black={black})")
    overlay_pan = put_label(overlay_pan, f"Mask (low={low_thr:.2f}, high={high_thr:.2f}, min={min_size}, close={int(use_close)}/{close_k})")

    panel = np.hstack([gray_panel, resp_panel, overlay_pan])
    cv2.imshow(WINDOW, panel)

    key = cv2.waitKey(16) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key == ord('s'):
        # 참고용 응답(8bit) + CVAT용 이진 마스크 저장
        resp_u8 = (np.clip(ridge_norm, 0, 1) * 255).astype(np.uint8)
        out_resp = os.path.join(SAVE_DIR, f"{SAVE_NAME}_sato_response.png")
        out_mask = os.path.join(SAVE_DIR, f"{SAVE_NAME}_sato_mask_hysteresis.png")
        cv2.imwrite(out_resp, resp_u8)
        cv2.imwrite(out_mask, mask_u8)
        print(f"[✓] saved:\n  - {out_resp}\n  - {out_mask}")

cv2.destroyAllWindows()
