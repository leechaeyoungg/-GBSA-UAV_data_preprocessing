import cv2
import numpy as np
import os

# ========= 사용자 설정 =========
IMAGE_PATH = r"C:\Users\dromii\Downloads\20250710_ori_migum\crack_test_dataset\DJI_0377_tile_y1152_x1920.jpg"
SAVE_DIR   = os.path.dirname(IMAGE_PATH)
WINDOW     = "Color-Pick Segmentation"
# =================================

samples = []
color_space = "HSV"
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(IMAGE_PATH)
h0, w0 = img_bgr.shape[:2]

def to_space(bgr, space):
    if space == "HSV":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    elif space == "Lab":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    else:
        raise ValueError(space)

img_hsv = to_space(img_bgr, "HSV")
img_lab = to_space(img_bgr, "Lab")

def get_space_image(space):
    return img_hsv if space == "HSV" else img_lab

def on_mouse(event, x, y, flags, param):
    global samples
    if event == cv2.EVENT_LBUTTONDOWN:
        space_img = get_space_image(color_space)
        color_vec = space_img[y, x].astype(int)
        samples.append((color_vec, color_space))
        print(f"[+] sample added ({color_space}) @({x},{y}): {color_vec.tolist()}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if samples:
            removed = samples.pop()
            print(f"[-] sample removed: {removed[0].tolist()} ({removed[1]})")

def make_mask():
    if not samples:
        return np.zeros((h0, w0), np.uint8)

    h_tol = cv2.getTrackbarPos("H/L  tol", WINDOW)
    s_tol = cv2.getTrackbarPos("S/a  tol", WINDOW)
    v_tol = cv2.getTrackbarPos("V/b  tol", WINDOW)
    ksz   = cv2.getTrackbarPos("morph k", WINDOW) * 2 + 1

    mask_total = np.zeros((h0, w0), np.uint8)
    img_space = get_space_image(color_space)

    for color_vec, space in samples:
        c0, c1, c2 = int(color_vec[0]), int(color_vec[1]), int(color_vec[2])
        if color_space == "HSV":
            h_low  = (c0 - h_tol) % 180
            h_high = (c0 + h_tol) % 180
            s_low, s_high = max(0, c1 - s_tol), min(255, c1 + s_tol)
            v_low, v_high = max(0, c2 - v_tol), min(255, c2 + v_tol)
            if h_low <= h_high:
                local_mask = cv2.inRange(img_space, (h_low, s_low, v_low),
                                                    (h_high, s_high, v_high))
            else:
                r1 = cv2.inRange(img_space, (0,     s_low, v_low),
                                           (h_high, s_high, v_high))
                r2 = cv2.inRange(img_space, (h_low, s_low, v_low),
                                           (179,   s_high, v_high))
                local_mask = cv2.bitwise_or(r1, r2)
        else:
            l_low, l_high = max(0, c0 - h_tol), min(255, c0 + h_tol)
            a_low, a_high = max(0, c1 - s_tol), min(255, c1 + s_tol)
            b_low, b_high = max(0, c2 - v_tol), min(255, c2 + v_tol)
            local_mask = cv2.inRange(img_space, (l_low, a_low, b_low),
                                                (l_high, a_high, b_high))
        mask_total = cv2.bitwise_or(mask_total, local_mask)

    if ksz >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, k, iterations=1)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, k, iterations=1)

    return mask_total

def overlay_mask(bgr, mask, alpha=0.5):
    color = (0, 255, 255)
    overlay = bgr.copy()
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return overlay

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 1200, 800)
cv2.setMouseCallback(WINDOW, on_mouse)

cv2.createTrackbar("H/L  tol", WINDOW, 10, 40,  lambda v: None)
cv2.createTrackbar("S/a  tol", WINDOW, 30, 127, lambda v: None)
cv2.createTrackbar("V/b  tol", WINDOW, 30, 127, lambda v: None)
cv2.createTrackbar("morph k", WINDOW, 1, 5,    lambda v: None)

help_text = (
    "[Left Click] add sample  |  [Right Click] remove last  |  [C] clear samples\n"
    "[Space] toggle HSV/Lab   |  [S] save mask & overlay   |  [Q] quit"
)

while True:
    mask = make_mask()
    view = overlay_mask(img_bgr, mask, alpha=0.45)
    panel = view.copy()
    y0 = 24
    for line in help_text.split("\n"):
        cv2.putText(panel, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(panel, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        y0 += 26
    cv2.putText(panel, f"Space: {color_space}  |  Samples: {len(samples)}",
                (12, y0+6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,0,255), 2, cv2.LINE_AA)

    cv2.imshow(WINDOW, panel)
    key = cv2.waitKey(16) & 0xFF

    if key in (ord('q'), 27):
        break
    elif key == ord('c'):
        samples.clear()
        print("[*] samples cleared")
    elif key == ord(' '):
        color_space = "Lab" if color_space == "HSV" else "HSV"
        print(f"[*] color space -> {color_space}")
    elif key == ord('s'):
        base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
        mask_path = os.path.join(SAVE_DIR, f"{base}_colorpick_mask.png")
        overlay_path = os.path.join(SAVE_DIR, f"{base}_colorpick_overlay.png")
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(overlay_path, overlay_mask(img_bgr, mask, alpha=0.45))
        print(f"[✓] saved:\n  - {mask_path}\n  - {overlay_path}")

cv2.destroyAllWindows()
