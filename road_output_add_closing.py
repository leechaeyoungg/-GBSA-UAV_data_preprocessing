import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from contextlib import nullcontext
import os


# SAM2 프로젝트 경로
sys.path.append(r"C:\Users\dromii\segment-anything-2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 모델 가중치 및 설정 파일 경로
sam2_checkpoint = r"C:\Users\dromii\segment-anything-2\checkpoints\sam2_hiera_large.pt"
sam2_model_cfg = r"C:\Users\dromii\segment-anything-2\sam2_configs\sam2_hiera_l.yaml"

# 추출할 이미지 경로
image_path = r"C:\Users\dromii\Downloads\20250710_ori_migum\20250710_lcy\DJI_0177.JPG"

# ==============================================================================
# 장치 설정
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 모델 로드
# ==============================================================================
try:
    sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("SAM2 model loaded successfully.")
except Exception as e:
    print(f"Error loading SAM2 model. Ensure Hydra is configured if needed: {e}")
    import hydra
    config_dir = os.path.join(os.path.dirname(sam2_checkpoint), "..", "sam2_configs")
    if not os.path.exists(config_dir):
        # Fallback if the path is not correct
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "segment-anything-2", "sam2_configs"))
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=config_dir, version_base="1.1")
    sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("SAM2 model loaded successfully after Hydra initialization.")

# ==============================================================================
# 이미지 로드 및 OpenCV 변환
# ==============================================================================
image = Image.open(image_path)
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB) # Matplotlib 시각화용

# ==============================================================================
# 인터랙티브 포인트 선택 (Positive/Negative)
# ==============================================================================
input_points = []
input_labels = [] 
window_name = 'Select Points - L_Click(Include), R_Click(Exclude) | s: segment, r: reset, q: quit'
display_image = image_cv.copy() 
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name, display_image)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        input_points.append([x, y])
        input_labels.append(1)
        cv2.circle(display_image, (x, y), 10, (0, 255, 0), -1) 
        cv2.imshow(window_name, display_image)
    elif event == cv2.EVENT_RBUTTONDOWN: 
        input_points.append([x, y])
        input_labels.append(0)
        cv2.circle(display_image, (x, y), 10, (0, 0, 255), -1) 
        cv2.imshow(window_name, display_image)

cv2.setMouseCallback(window_name, mouse_callback)

print('도로 영역을 왼쪽 클릭, 제외할 영역을 오른쪽 클릭하세요.')
print('선택 완료: "s" | 초기화: "r" | 종료: "q"')

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        sys.exit("작업을 종료합니다.")
    elif key == ord('r'):
        input_points = []
        input_labels = []
        display_image = image_cv.copy()
        cv2.imshow(window_name, display_image)
        print("포인트를 초기화했습니다. 다시 선택하세요.")
    elif key == ord('s'):
        if not input_points:
            print("선택된 포인트가 없습니다. 하나 이상 선택해주세요.")
            continue
        cv2.destroyAllWindows()
        break

# ==============================================================================
# SAM2를 이용한 도로 영역 분할
# ==============================================================================
print(f"선택된 포인트 수: {len(input_points)}")
print("SAM2로 도로 영역 분할을 시작합니다...")

sam2_predictor.set_image(image_rgb)
points_np = np.array(input_points)
labels_np = np.array(input_labels)

autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()

with torch.inference_mode(), autocast_ctx:
    masks, scores, _ = sam2_predictor.predict(
        point_coords=points_np,
        point_labels=labels_np,
        multimask_output=False,
    )

if masks is None:
    print("분할된 영역을 찾지 못했습니다.")
    sys.exit()

# SAM2 원본 마스크 생성 (구멍이 있을 수 있음)
road_mask_raw = (masks[0] > 0.0)



# ==============================================================================
# 후처리: 마스크 내부의 구멍 채우기 (Closing 연산)
# ==============================================================================
print("마스크 후처리를 시작합니다: 내부 구멍 채우기...")

# OpenCV 연산을 위해 boolean 마스크를 0과 255의 값으로 변환
mask_for_closing = road_mask_raw.astype(np.uint8) * 255

# Closing 연산에 사용할 커널(kernel) 크기를 정의합니다.
# 이 값은 이미지 해상도와 메우고 싶은 구멍의 크기에 따라 조절해야 합니다.
kernel_size = 70  # 예시 값, 필요에 따라 30, 50, 100 등으로 조절
kernel = np.ones((kernel_size, kernel_size), np.uint8)

# Closing 연산 적용
closed_mask_uint8 = cv2.morphologyEx(mask_for_closing, cv2.MORPH_CLOSE, kernel)

# 최종 사용할 boolean 마스크로 다시 변환
road_mask_final = (closed_mask_uint8 > 0)

print("후처리가 완료되었습니다.")


# ==============================================================================
# 결과 시각화 및 저장 (이제 'road_mask_final'을 사용)
# ==============================================================================

# (선택 사항) 원본 마스크와 후처리 마스크 비교 시각화
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(road_mask_raw, cmap='gray')
plt.title('Original SAM2 Mask (with holes)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(road_mask_final, cmap='gray')
plt.title('Final Mask (Holes Filled)')
plt.axis('off')
plt.tight_layout()
plt.show()


# 최종 마스크를 사용하여 오버레이 이미지 생성
overlay_image = image_rgb.copy()
color_mask = np.array([255, 0, 0], dtype=np.uint8) # 빨간색으로 표시
alpha = 0.5
overlay_image[road_mask_final] = (overlay_image[road_mask_final] * (1 - alpha) + color_mask * alpha).astype(np.uint8)

# 클릭했던 포인트 표시
for i, point in enumerate(input_points):
    color = (0, 255, 0) if input_labels[i] == 1 else (0, 0, 255)
    cv2.circle(overlay_image, tuple(point), 15, color, -1)

# 최종 결과 시각화
plt.figure(figsize=(15, 15))
plt.imshow(overlay_image)
plt.title("Extracted Road Area (Post-processed)")
plt.axis('off')
plt.show()

# 최종 마스크 파일 저장
output_mask_path = r"C:\Users\dromii\road_mask_final.png"
road_mask_binary = (road_mask_final * 255).astype(np.uint8)
cv2.imwrite(output_mask_path, road_mask_binary)
print(f"최종 도로 영역 마스크가 '{output_mask_path}' 파일로 저장되었습니다.")

# 최종 마스크로 도로 영역만 추출하여 저장
output_road_path = r"C:\Users\dromii\road_only_final.png"
road_only_image = image_rgb.copy()
road_only_image[~road_mask_final] = 0 # 도로가 아닌 영역은 검은색으로 처리
cv2.imwrite(output_road_path, cv2.cvtColor(road_only_image, cv2.COLOR_RGB2BGR))
print(f"최종 추출된 도로 이미지가 '{output_road_path}' 파일로 저장되었습니다.")
