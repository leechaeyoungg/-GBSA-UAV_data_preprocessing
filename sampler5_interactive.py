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
INPUT_DIR = r"C:\Users\dromii\Downloads\20250710_ori_migum\sjh_wrong_road"
# 최종 마스크가 저장될 폴더
OUTPUT_DIR = r"C:\Users\dromii\Downloads\20250710_ori_migum\sjh_wrong_road_masks"

# --- SAM2 모델 설정 ---
SAM_PROJECT_PATH = r"C:\Users\dromii\segment-anything-2"
SAM2_CHECKPOINT = os.path.join(SAM_PROJECT_PATH, "checkpoints", "sam2_hiera_large.pt")
SAM2_MODEL_CFG = os.path.join(SAM_PROJECT_PATH, "sam2_configs", "sam2_hiera_l.yaml")

# --- 강화된 후처리 파라미터 ---
# SAM 마스크의 작은 노이즈(흰 점) 제거용 커널 크기
POST_SAM_OPEN_KERNEL_SIZE = 9
# SAM 마스크의 내부 구멍(검은 점) 채우기용 커널 크기
POST_SAM_CLOSE_KERNEL_SIZE = 70

# ==============================================================================
# 2. 모델 로드
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

sys.path.append(SAM_PROJECT_PATH)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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

# ==============================================================================
# 3. 메인 실행 함수 (폴더 일괄 처리 및 인터랙티브 루프)
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    except FileNotFoundError:
        print(f"오류: 입력 폴더를 찾을 수 없습니다: {INPUT_DIR}")
        return

    print(f"총 {len(image_files)}개의 이미지를 처리합니다.")

    # --- 외부 루프: 폴더의 모든 이미지를 순회 ---
    for i, filename in enumerate(image_files):
        image_path = os.path.join(INPUT_DIR, filename)
        image_cv = cv2.imread(image_path)
        if image_cv is None:
            print(f"경고: {filename}을(를) 로드할 수 없습니다. 건너뜁니다.")
            continue
        
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        input_points = []
        input_labels = []
        final_mask = None
        segmentation_complete = False
        
        # --- 내부 루프: 한 이미지에 대한 작업 (결과 승인 전까지 반복) ---
        while True:
            window_name = f"[{i+1}/{len(image_files)}] {filename}"
            
            # 현재 상태에 따라 화면에 표시할 이미지 결정
            if not segmentation_complete:
                display_image = image_cv.copy()
                for pt, label in zip(input_points, input_labels):
                    color = (0, 255, 0) if label == 1 else (0, 0, 255)
                    cv2.circle(display_image, tuple(pt), 15, color, -1)
                instructions = "Points: L-Click(IN), R-Click(EX) | Keys: (s)egment, (q)uit"
            else:
                display_image = image_cv.copy()
                overlay_color = np.array([255, 0, 0], dtype=np.uint8) # 파란색
                display_image[final_mask > 0] = (display_image[final_mask > 0] * 0.6 + overlay_color * 0.4).astype(np.uint8)
                instructions = "Keys: (n)ext, (r)etry, (q)uit"

            cv2.putText(display_image, instructions, (20, display_image.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            def mouse_callback(event, x, y, flags, param):
                if segmentation_complete: return
                if event == cv2.EVENT_LBUTTONDOWN:
                    input_points.append([x, y]); input_labels.append(1)
                elif event == cv2.EVENT_RBUTTONDOWN:
                    input_points.append([x, y]); input_labels.append(0)
            
            cv2.setMouseCallback(window_name, mouse_callback)
            cv2.imshow(window_name, display_image)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows(); print("작업을 종료합니다."); return
            
            elif key == ord('r'):
                print(" -> 포인트를 초기화하고 현재 이미지 작업을 재시도합니다.")
                input_points, input_labels, final_mask, segmentation_complete = [], [], None, False
                continue
            
            elif key == ord('s') and not segmentation_complete:
                if not input_points:
                    print(" -> 경고: 포인트를 먼저 선택해야 합니다."); continue
                
                print(" -> SAM2로 분할 및 후처리를 시작합니다...")
                sam2_predictor.set_image(image_rgb)
                points_np = np.array(input_points)
                labels_np = np.array(input_labels)

                with torch.inference_mode(), nullcontext():
                    masks, _, _ = sam2_predictor.predict(
                        point_coords=points_np, point_labels=labels_np, multimask_output=False
                    )
                
                if masks is None:
                    print(" -> SAM2가 마스크를 생성하지 못했습니다. 다시 시도해주세요."); continue
                
                # --- 강화된 후처리 로직 ---
                sam_mask_raw = (masks[0] > 0.0).astype(np.uint8) * 255
                open_kernel = np.ones((POST_SAM_OPEN_KERNEL_SIZE, POST_SAM_OPEN_KERNEL_SIZE), np.uint8)
                opened_sam_mask = cv2.morphologyEx(sam_mask_raw, cv2.MORPH_OPEN, open_kernel)
                close_kernel = np.ones((POST_SAM_CLOSE_KERNEL_SIZE, POST_SAM_CLOSE_KERNEL_SIZE), np.uint8)
                closed_sam_mask = cv2.morphologyEx(opened_sam_mask, cv2.MORPH_CLOSE, close_kernel)
                contours, _ = cv2.findContours(closed_sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                temp_final_mask = np.zeros_like(closed_sam_mask)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    cv2.drawContours(temp_final_mask, [largest_contour], -1, 255, -1)
                final_mask = temp_final_mask
                # --- 후처리 끝 ---
                
                print(" -> 분할 및 후처리 완료. 결과를 확인하세요.")
                segmentation_complete = True
            
            elif key == ord('n') and segmentation_complete:
                print(" -> 결과를 승인하고 마스크를 저장합니다.")
                base_name = os.path.splitext(filename)[0]
                output_mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_mask.png")
                cv2.imwrite(output_mask_path, final_mask)
                print(f" -> 저장 완료: {output_mask_path}")
                break

    cv2.destroyAllWindows()
    print("\n===================================")
    print("모든 이미지 처리가 완료되었습니다!")
    print("===================================")

if __name__ == "__main__":
    main()