import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color, img_as_float
from skimage.filters import sato  # ridge detector

# ====== 사용자 설정 ======
IMAGE_PATH = r"C:\Users\dromii\Downloads\20250710_ori_migum\crack_test_dataset\DJI_0193_tile_y0_x2867.jpg"
SAVE_DIR   = os.path.dirname(IMAGE_PATH)
SAVE_NAME  = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

# Sato 파라미터
SIGMAS = (1, 2, 3, 4)   # 멀티스케일(픽셀 단위). 얇은 크랙이면 (1,2,3) 정도부터 시작
BLACK_RIDGES = True     # 크랙이 어두운 선이면 True, 밝은 선이면 False
# ========================

# 1) 이미지 로드 → 그레이스케일(float[0,1])
img = io.imread(IMAGE_PATH)
if img.ndim == 3:
    img_gray = color.rgb2gray(img)        # float64, 0~1
else:
    # 단일 채널(8bit)일 경우 0~1 스케일로 변환
    img_gray = img_as_float(img)

# (선택) 살짝 블러로 노이즈 억제 — 너무 매끈하면 생략 가능
# from skimage.filters import gaussian
# img_gray = gaussian(img_gray, sigma=0.5, preserve_range=True)

# 2) Sato ridge filter 적용 (float 결과, 0~양수)
ridge = sato(img_gray, sigmas=SIGMAS, black_ridges=BLACK_RIDGES)

# 3) 시각화용 정규화 (0~1)
ridge_norm = (ridge - ridge.min()) / (ridge.max() - ridge.min() + 1e-8)

# 4) 간단 Threshold로 마스크 생성 (Otsu or 수동 임계)
#    수동값이 더 컨트롤하기 쉬움. 높일수록 더 얇고 강한 선만 남음.
thr_val = 0.08
mask = (ridge_norm >= thr_val).astype(np.uint8) * 255

# (선택) 모폴로지 정리
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# 5) 저장
out_ridge_path = os.path.join(SAVE_DIR, f"{SAVE_NAME}_sato_response.png")
out_mask_path  = os.path.join(SAVE_DIR, f"{SAVE_NAME}_sato_mask.png")
cv2.imwrite(out_ridge_path, (ridge_norm * 255).astype(np.uint8))
cv2.imwrite(out_mask_path, mask_clean)
print(f"[✓] saved:\n  - {out_ridge_path}\n  - {out_mask_path}")

# 6) 화면 표시
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.title("Gray"); plt.imshow(img_gray, cmap='gray'); plt.axis('off')
plt.subplot(1,3,2); plt.title(f"Sato response (sigmas={SIGMAS}, black={BLACK_RIDGES})")
plt.imshow(ridge_norm, cmap='gray'); plt.axis('off')
plt.subplot(1,3,3); plt.title(f"Mask (thr={thr_val})"); plt.imshow(mask_clean, cmap='gray'); plt.axis('off')
plt.tight_layout(); plt.show()
