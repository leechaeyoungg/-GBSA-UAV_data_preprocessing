import os
import cv2
import numpy as np
import sys
import torch
from contextlib import nullcontext
from PIL import Image

# ==============================================================================
# 1. 설정 (사용자 수정 및 튜닝 영역)
# ==============================================================================

# --- 경로 설정 ---
INPUT_DIR = r"C:\Users\dromii\Downloads\20250710_ori_migum\20250710_lcy" # 테스트할 원본 타일이 있는 폴더
OUTPUT_DIR = r"C:\Users\dromii\Downloads\20250710_ori_migum\lcy_sam_masks" # 최종 마스크가 저장될 폴더

# --- SAM2 모델 설정 ---
SAM_PROJECT_PATH = r"C:\Users\dromii\segment-anything-2"
SAM2_CHECKPOINT = os.path.join(SAM_PROJECT_PATH, "checkpoints", "sam2_hiera_large.pt")
SAM2_MODEL_CFG = os.path.join(SAM_PROJECT_PATH, "sam2_configs", "sam2_hiera_l.yaml")

# --- Classic CV 파라미터 ---
GRAY_LOWER = np.array([0, 0, 40])
GRAY_UPPER = np.array([179, 50, 220])
OPEN_KERNEL_SIZE = 5
INITIAL_CLOSE_KERNEL_SIZE = 15

# --- 최종 후처리 파라미터 ---
FINAL_CLOSE_KERNEL_SIZE = 70

# ==============================================================================
# 2. 모델 로드 및 헬퍼 함수
# ==============================================================================

# --- SAM2 모델 로드 ---
sys.path.append(SAM_PROJECT_PATH)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
try:
    sam2_model = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("SAM2 model loaded successfully.")
except Exception as e:
    # (이전 코드와 동일한 Hydra 초기화 로직)
    print(f"Error loading SAM2 model: {e}. Trying with Hydra initialization...")
    import hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=os.path.join(SAM_PROJECT_PATH, "sam2_configs"), version_base="1.1")
    sam2_model = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("SAM2 model loaded successfully after Hydra initialization.")


def create_initial_road_mask(image):
    """Classic CV 기법으로 가장 유력한 도로 후보 마스크를 생성"""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray_mask = cv2.inRange(hsv_image, GRAY_LOWER, GRAY_UPPER)
    
    open_kernel = np.ones((OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE), np.uint8)
    initial_close_kernel = np.ones((INITIAL_CLOSE_KERNEL_SIZE, INITIAL_CLOSE_KERNEL_SIZE), np.uint8)
    opened_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, open_kernel)
    initial_closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, initial_close_kernel)
    
    contours, _ = cv2.findContours(initial_closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_rough_mask = np.zeros_like(initial_closed_mask)
    largest_contour = None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > (image.shape[0] * image.shape[1] * 0.1):
             cv2.drawContours(final_rough_mask, [largest_contour], -1, 255, -1)
    
    return final_rough_mask, largest_contour

def get_centroid_prompt(contour):
    """컨투어의 중심점을 찾아 SAM 프롬프트로 변환"""
    if contour is None:
        return np.array([]), np.array([])
        
    M = cv2.moments(contour)
    if M["m00"] == 0: # 면적이 0인 경우 (나누기 오류 방지)
        return np.array([]), np.array([])
        
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    
    point = np.array([[center_x, center_y]])
    label = np.array([1]) # 포지티브 프롬프트
    
    return point, label

# ==============================================================================
# 3. 메인 실행 루프
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.jpg')]
    print(f"총 {len(image_files)}개의 이미지를 처리합니다.")

    for i, filename in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] 처리 중: {filename}")
        image_path = os.path.join(INPUT_DIR, filename)
        image = cv2.imread(image_path)
        if image is None: continue
        
        # --- 1단계: Classic CV로 가장 큰 덩어리(컨투어) 찾기 ---
        rough_mask, largest_contour = create_initial_road_mask(image)
        if largest_contour is None:
            print("  -> 유효한 도로 후보 영역을 찾지 못했습니다. 건너뜁니다.")
            continue
            
        # --- 2단계: 중심점 프롬프트 생성 ---
        point_np, label_np = get_centroid_prompt(largest_contour)
        if len(point_np) == 0:
            print("  -> 중심점을 생성하지 못했습니다. 건너뜁니다.")
            continue
        print(f"  -> 중심점 {point_np[0]}을 생성하여 SAM에 전달합니다.")
        
        # --- 3단계: SAM2 실행 ---
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam2_predictor.set_image(image_rgb)
        
        with torch.inference_mode(), nullcontext():
            masks, _, _ = sam2_predictor.predict(
                point_coords=point_np,
                point_labels=label_np,
                multimask_output=False,
            )
        if masks is None: continue
        
        sam_mask_raw = (masks[0] > 0.0)
        
        # --- 4단계: 최종 후처리 (구멍 채우기) ---
        mask_for_closing = sam_mask_raw.astype(np.uint8) * 255
        close_kernel = np.ones((FINAL_CLOSE_KERNEL_SIZE, FINAL_CLOSE_KERNEL_SIZE), np.uint8)
        final_mask_uint8 = cv2.morphologyEx(mask_for_closing, cv2.MORPH_CLOSE, close_kernel)
        
        # --- 5단계: 결과 저장 ---
        output_filename = os.path.splitext(filename)[0] + "_mask.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, final_mask_uint8)
        print(f"  -> 최종 마스크를 저장했습니다: {output_path}")

    print("\n모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    main()
