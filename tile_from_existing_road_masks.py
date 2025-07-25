import os
import re
import cv2
import numpy as np
from pathlib import Path

# ─── 0. 경로·파라미터 ────────────────────────────────────────────────────────────
INPUT_IMAGE_DIR   = r"C:\Users\dromii\Downloads\20250710_ori_migum\20250710\images"
INPUT_MASK_DIR    = r"C:\Users\dromii\Downloads\20250710_ori_migum\20250710\masks"
OUTPUT_DATASET_DIR= r"C:\Users\dromii\Downloads\20250710_ori_migum\20250710_tiled_512"

TILE_SIZE  = 512
OVERLAP    = 128
MIN_RATIO  = 0.20                       # 타일 내 도로(흰색) 최소 비율

# ─── 1. 마스크 사전 구축:  DJI_####  →  [마스크파일1, 마스크파일2 …] ─────────────
mask_lookup = {}
for fname in os.listdir(INPUT_MASK_DIR):
    if not fname.lower().endswith(".png"): 
        continue
    m = re.match(r"(DJI_\d+)", fname)     # DJI_0517_mask.png, DJI_0517_mask_lcy.png 등
    if not m:
        continue
    dji_id = m.group(1)                   # 'DJI_0517'
    mask_lookup.setdefault(dji_id, []).append(fname)

# ─── 2. 출력 폴더 준비 ───────────────────────────────────────────────────────────
images_out = Path(OUTPUT_DATASET_DIR) / "images"
masks_out  = Path(OUTPUT_DATASET_DIR) / "masks"
images_out.mkdir(parents=True, exist_ok=True)
masks_out.mkdir(parents=True, exist_ok=True)

# ─── 3. 이미지 루프 ──────────────────────────────────────────────────────────────
image_files = [f for f in os.listdir(INPUT_IMAGE_DIR)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"총 {len(image_files)}개 이미지 처리")

stride = TILE_SIZE - OVERLAP
total_tiles = 0

for idx, img_name in enumerate(image_files, 1):
    print(f"\n[{idx}/{len(image_files)}]  {img_name}")
    base  = Path(img_name).stem                 # 'DJI_0569' or 'DJI_0569_lcy'
    
    # ── 3‑1. DJI 아이디·폴더키 추출 ────────────────────────────────────────────
    m_id = re.match(r"(DJI_\d+)", base)
    if not m_id:
        print("  └─ DJI 아이디 인식 실패 → 건너뜀")
        continue
    dji_id = m_id.group(1)                     # 'DJI_0569'
    
    # 이미지 이름에 추가 키(_lcy/_sjh 등) 존재 여부
    extra_part = base[len(dji_id):]            # ''  or '_lcy' or '_lcy_1'
    folder_key = extra_part.lstrip('_').split('_')[0] if extra_part.startswith('_') else None
    
    # ── 3‑2. 대응 마스크 후보 찾기 ────────────────────────────────────────────
    candidates = mask_lookup.get(dji_id, [])
    selected_mask = None
    
    if folder_key:                             # 키가 있으면 우선 매칭
        for f in candidates:
            stem = Path(f).stem
            if f"_{folder_key}" in stem:       # '_lcy' 포함 확인
                selected_mask = f
                break
    
    if not selected_mask and candidates:       # 그래도 없으면 첫 번째 아무거나
        selected_mask = candidates[0]
    
    if not selected_mask:
        print("  └─ ⚠️  마스크 없음 → 건너뜀")
        continue
    
    # ── 3‑3. 파일 로드 ──────────────────────────────────────────────────────
    img_path  = Path(INPUT_IMAGE_DIR) / img_name
    mask_path = Path(INPUT_MASK_DIR)  / selected_mask
    image = cv2.imread(str(img_path))
    mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None or image.shape[:2] != mask.shape:
        print("  └─ ⚠️  파일 로드/크기 불일치 → 건너뜀")
        continue
    
    # ── 3‑4. 도로 BBox 산출(최대 컨투어) ────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("  └─ 도로 없음 → 건너뜀")
        continue
    bx, by, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    
    # ── 3‑5. BBox 내부 슬라이딩 타일링 ───────────────────────────────────────
    tiles_here = 0
    for y in range(by, by + bh, stride):
        for x in range(bx, bx + bw, stride):
            y2, x2 = y + TILE_SIZE, x + TILE_SIZE
            if y2 > mask.shape[0] or x2 > mask.shape[1]:
                continue
            
            mask_tile = mask[y:y2, x:x2]
            if (mask_tile > 127).sum() < TILE_SIZE * TILE_SIZE * MIN_RATIO:
                continue
            
            img_tile = image[y:y2, x:x2]
            tile_stem = f"{base}_tile_y{y}_x{x}"
            cv2.imwrite(str(images_out / f"{tile_stem}.jpg"), img_tile)
            cv2.imwrite(str(masks_out  / f"{tile_stem}.png"), mask_tile)
            tiles_here += 1
    
    print(f"  └─ 유효 타일 {tiles_here}개")
    total_tiles += tiles_here

# ─── 4. 요약 ────────────────────────────────────────────────────────────────────
print("\n==============================")
print(f"총 생성 타일: {total_tiles}")
print(f"저장 위치  : {OUTPUT_DATASET_DIR}")
print("==============================")
