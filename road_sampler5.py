import os
import cv2
import numpy as np
import sys
import torch
from contextlib import nullcontext

# ==============================================================================
# 1. 설정 (사용자 수정 및 튜닝 영역)
# ==============================================================================

# --- 경로 설정 ---
# 처리할 원본 고해상도 이미지들이 있는 '폴더' 경로
INPUT_DIR = r"C:\Users\dromii\Downloads\20250710_ori_migum\20250710_sjh"
# 최종 마스크가 저장될 폴더
OUTPUT_DIR = r"C:\Users\dromii\Downloads\20250710_ori_migum\sjh_masks"

# --- SAM2 모델 설정 ---
SAM_PROJECT_PATH = r"C:\Users\dromii\segment-anything-2"
SAM2_CHECKPOINT = os.path.join(SAM_PROJECT_PATH, "checkpoints", "sam2_hiera_large.pt")
SAM2_MODEL_CFG = os.path.join(SAM_PROJECT_PATH, "sam2_configs", "sam2_hiera_l.yaml")

# --- 1단계: Classic CV 파라미터 ---
GRAY_LOWER = np.array([0, 0, 50])
GRAY_UPPER = np.array([179, 50, 220])
OPEN_KERNEL_SIZE = 5
CLOSE_KERNEL_SIZE = 150

# --- 2단계: SAM 프롬프트 생성 파라미터 ---
NUM_RANDOM_POINTS = 20


# SAM 마스크의 작은 노이즈 제거용 커널
POST_SAM_OPEN_KERNEL_SIZE = 9
# SAM 마스크의 내부 구멍 채우기용 커널
POST_SAM_CLOSE_KERNEL_SIZE = 70

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
    print(f"Error loading SAM2 model: {e}. Trying with Hydra initialization...")
    import hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=os.path.join(SAM_PROJECT_PATH, "sam2_configs"), version_base="1.1")
    sam2_model = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("SAM2 model loaded successfully after Hydra initialization.")


def create_initial_mask_with_classic_cv(image):
    """OpenCV만을 사용하여 도로 후보 마스크를 생성합니다."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_mask = cv2.inRange(hsv_image, GRAY_LOWER, GRAY_UPPER)
    open_kernel = np.ones((OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE), np.uint8)
    opened_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, open_kernel)
    close_kernel = np.ones((CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE), np.uint8)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, close_kernel)
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    initial_mask = np.zeros_like(closed_mask)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(initial_mask, [largest_contour], -1, 255, -1)
    return initial_mask

def generate_points_from_mask(mask, num_points):
    """마스크 내부에서 지정된 개수만큼의 포인트를 무작위로 생성합니다."""
    positive_candidates = np.argwhere(mask > 0)
    if len(positive_candidates) < num_points:
        points_yx = positive_candidates
    else:
        indices = np.random.choice(len(positive_candidates), num_points, replace=False)
        points_yx = positive_candidates[indices]
    if len(points_yx) == 0:
        return np.array([]), np.array([])
    points_xy = points_yx[:, ::-1].copy()
    labels = np.ones(len(points_xy))
    return points_xy, labels

# ==============================================================================
# 3. 메인 실행 함수 (폴더 일괄 처리)
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print(f"오류: 입력 폴더에 지원하는 이미지 파일이 없습니다: {INPUT_DIR}")
            return
    except FileNotFoundError:
        print(f"오류: 입력 폴더를 찾을 수 없습니다: {INPUT_DIR}")
        return

    print(f"총 {len(image_files)}개의 이미지를 처리합니다.")

    for i, filename in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] 처리 중: {filename}")
        image_path = os.path.join(INPUT_DIR, filename)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"  -> 경고: 이미지를 로드할 수 없어 건너뜁니다.")
            continue

        # --- 1단계 & 2단계: 초기 마스크 생성 및 포인트 샘플링 ---
        initial_mask = create_initial_mask_with_classic_cv(image)
        if np.sum(initial_mask) == 0:
            print("  -> 1. Classic CV가 도로 후보 영역을 찾지 못했습니다. 건너뜁니다.")
            continue
        
        points_np, labels_np = generate_points_from_mask(initial_mask, NUM_RANDOM_POINTS)
        if len(points_np) == 0:
            print("  -> 2. 랜덤 프롬프트 포인트를 생성하지 못했습니다. 건너뜁니다.")
            continue
        print(f"  -> 1&2. 초기 마스크 및 {len(points_np)}개 포인트 생성 완료.")

        # --- 3단계: SAM2로 마스크 추출 ---
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam2_predictor.set_image(image_rgb)
        
        with torch.inference_mode(), nullcontext():
            masks, _, _ = sam2_predictor.predict(
                point_coords=points_np,
                point_labels=labels_np,
                multimask_output=False,
            )
        if masks is None:
            print("  -> 3. SAM2가 마스크를 생성하지 못했습니다. 건너뜁니다.")
            continue
        
        sam_mask_raw = (masks[0] > 0.0).astype(np.uint8) * 255
        print("  -> 3. SAM2 마스크 생성 완료.")

        # ★★★★★★★★★★★★ 이 부분이 요청하신 핵심 로직입니다 ★★★★★★★★★★★★
        # --- 4단계: SAM2 결과물에 대한 강화된 후처리 ---
        print("  -> 4. SAM2 마스크에 대한 강화된 후처리 시작...")
        # 4-a. Opening 연산으로 작은 노이즈 제거
        open_kernel = np.ones((POST_SAM_OPEN_KERNEL_SIZE, POST_SAM_OPEN_KERNEL_SIZE), np.uint8)
        opened_sam_mask = cv2.morphologyEx(sam_mask_raw, cv2.MORPH_OPEN, open_kernel)

        # 4-b. Closing 연산으로 내부 구멍 채우기
        close_kernel = np.ones((POST_SAM_CLOSE_KERNEL_SIZE, POST_SAM_CLOSE_KERNEL_SIZE), np.uint8)
        closed_sam_mask = cv2.morphologyEx(opened_sam_mask, cv2.MORPH_CLOSE, close_kernel)
        
        # 4-c. 처리된 마스크에서 가장 큰 영역만 남기기
        contours, _ = cv2.findContours(closed_sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(closed_sam_mask)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
        print("  -> 4. 강화된 후처리 완료.")
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        
        # --- 5단계: 최종 마스크 저장 ---
        if np.sum(final_mask) > 0:
            base_name = os.path.splitext(filename)[0]
            output_mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_final_mask.png")
            cv2.imwrite(output_mask_path, final_mask)
            print(f"  -> 5. 최종 마스크를 저장했습니다: {output_mask_path}")
        else:
            print("  -> 5. 후처리 결과 유효한 마스크 영역이 없어 저장하지 않습니다.")

    print("\n===================================")
    print("모든 작업이 성공적으로 완료되었습니다!")
    print("===================================")

if __name__ == "__main__":
    main()