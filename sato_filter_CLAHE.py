import os, cv2, numpy as np
from skimage import io, color, img_as_float
from skimage.filters import sato, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects

# ====== 경로/파라미터 ======
IMAGE_PATH   = r"C:\Users\dromii\Downloads\20250710_ori_migum\crack_test_dataset\DJI_0192_tile_y0_x1943.jpg"
SAVE_DIR     = os.path.dirname(IMAGE_PATH)
NAME         = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

SIGMAS       = (1,2,3,4)   # 크랙 두께 스케일
BLACK_RIDGES = True        # 크랙이 어두운 선이면 True
LOW_THR, HIGH_THR = 0.10, 0.20  # 히스테리시스 임계(0~1)
MIN_SIZE     = 30          # 소형영역 제거 픽셀 수
CLAHE_CLIP   = 2.0         # CLAHE 클리핑(과증폭 제한)
CLAHE_TILE   = (8,8)       # 타일(국소 영역) 크기
# ==========================

# 1) 입력 → Gray[0..1]
img = io.imread(IMAGE_PATH)
gray = color.rgb2gray(img) if img.ndim==3 else img_as_float(img)

# (선택) 간단한 그림자(저주파) 억제: 대형 블러를 빼서 배경 제거(홈모픽/Retinex 전단계 대용)
# blur_bg = cv2.GaussianBlur((gray*255).astype(np.uint8), (0,0), sigmaX=21, sigmaY=21)
# gray_deshadow = np.clip((gray*255 - blur_bg + 128)/255, 0, 1)

# 2) CLAHE (equalizeHist보다 그림자에 덜 민감)
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
gray_u8   = (np.clip(gray,0,1)*255).astype(np.uint8)
gray_cla  = clahe.apply(gray_u8)                 # uint8
gray_claF = gray_cla.astype(np.float32)/255.0    # float [0..1]

# 3) Sato: 원본/CLAHE 각각 적용
def sato_norm(g, sigmas, black):
    r = sato(g, sigmas=sigmas, black_ridges=black)
    r = (r - r.min()) / (r.max() - r.min() + 1e-8)
    return r

ridge_orig = sato_norm(gray,      SIGMAS, BLACK_RIDGES)
ridge_cla  = sato_norm(gray_claF, SIGMAS, BLACK_RIDGES)

# 4) 히스테리시스 + 소형영역 제거(형태 보존)
def to_mask(ridge_norm, low, high, min_size):
    m = apply_hysteresis_threshold(ridge_norm, low, high)
    if MIN_SIZE > 0:
        m = remove_small_objects(m, min_size=min_size)
    return (m.astype(np.uint8)*255)

mask_orig = to_mask(ridge_orig, LOW_THR, HIGH_THR, MIN_SIZE)
mask_cla  = to_mask(ridge_cla,  LOW_THR, HIGH_THR, MIN_SIZE)

# 5) 저장(응답/마스크)
cv2.imwrite(os.path.join(SAVE_DIR, f"{NAME}_sato_resp_orig.png"), (ridge_orig*255).astype(np.uint8))
cv2.imwrite(os.path.join(SAVE_DIR, f"{NAME}_sato_resp_clahe.png"), (ridge_cla *255).astype(np.uint8))
cv2.imwrite(os.path.join(SAVE_DIR, f"{NAME}_sato_mask_orig.png"),  mask_orig)
cv2.imwrite(os.path.join(SAVE_DIR, f"{NAME}_sato_mask_clahe.png"), mask_cla)

# 6) 빠른 비교 뷰(원본/CLAHE 응답/마스크 오버레이)
def overlay(grayF, mask, alpha=0.45, color=(0,255,255)):
    g = (np.clip(grayF,0,1)*255).astype(np.uint8)
    g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    m = mask>0
    g[m] = (g[m]*(1-alpha) + np.array(color)*alpha).astype(np.uint8)
    return g

ov_orig = overlay(gray,      mask_orig)
ov_cla  = overlay(gray_claF, mask_cla)

panel = np.hstack([
    cv2.cvtColor((gray*255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(gray_cla, cv2.COLOR_GRAY2BGR),
    cv2.cvtColor((ridge_orig*255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
    cv2.cvtColor((ridge_cla *255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
    ov_orig, ov_cla
])
cv2.putText(panel, "Gray | CLAHE | Sato(orig) | Sato(CLAHE) | Overlay(orig) | Overlay(CLAHE)",
            (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
cv2.putText(panel, "Gray | CLAHE | Sato(orig) | Sato(CLAHE) | Overlay(orig) | Overlay(CLAHE)",
            (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
cv2.namedWindow("Compare", cv2.WINDOW_NORMAL); cv2.imshow("Compare", panel); cv2.waitKey(0)
cv2.destroyAllWindows()
