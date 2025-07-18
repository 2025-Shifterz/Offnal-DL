# ─────────────────────────────
# 0. 패키지 임포트 및 설정
# ─────────────────────────────
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow as tf

# 사용자 설정
IMAGE_PATH   = "/content/camera1.jpg"       # 근무표 이미지 경로
TFLITE_MODEL = "dne_classifier.tflite"      # TFLite 분류기 모델
WHITE_THRESH = 0.92                         # 빈 셀 판단 기준
CLASS_LABELS = ['D', 'N', 'E']              # 분류 클래스
HEADER_SKIP_Y = 1                           # 위쪽 헤더 행 수
HEADER_SKIP_X = 1                           # 왼쪽 헤더 열 수

# ─────────────────────────────
# 1. 표 검출 및 워핑(정렬)
# ─────────────────────────────
def detect_and_warp_table(img_path):
    orig_img = cv2.imread(img_path)
    assert orig_img is not None, f"Image not found at {img_path}"
    max_width = 1000
    if orig_img.shape[1] > max_width:
        scale = max_width / orig_img.shape[1]
        orig_img = cv2.resize(orig_img, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)
    plt.imshow(binary, cmap='gray')
    plt.title("이진화된 이미지")
    plt.axis('off')
    plt.show()

    # 외곽 검출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    table_contour = None
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            table_contour = approx
            break

    if table_contour is not None:
        def sort_points(pts):
            pts = pts.reshape(4, 2)
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            return np.array([
                pts[np.argmin(s)], pts[np.argmin(diff)],
                pts[np.argmax(s)], pts[np.argmax(diff)]
            ], dtype='float32')
        rect = sort_points(table_contour)
        width = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))
        height = int(max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2])))
        dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype='float32')
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(orig_img, M, (width, height))
        # 시각화
        temp = orig_img.copy()
        cv2.drawContours(temp, [table_contour], -1, (0, 255, 0), 3)
    else:
        print("⚠️ 표 외곽 인식 실패 — 원본 사용")
        warped = orig_img
        temp = orig_img.copy()

    temp_rgb = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    plt.imshow(temp_rgb)
    plt.title("표 외곽 검출 결과")
    plt.axis('off')
    plt.show()
    return warped

# ─────────────────────────────
# 2. robust grid 기반 셀 경계 검출
# ─────────────────────────────
def get_table_cells(warped, skip_rows=HEADER_SKIP_Y, skip_cols=HEADER_SKIP_X):
    img = warped
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    h, w = bw.shape
    hor_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//30), 1))
    ver_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//30)))
    hor = cv2.erode(bw, hor_k); hor = cv2.dilate(hor, hor_k)
    ver = cv2.erode(bw, ver_k); ver = cv2.dilate(ver, ver_k)
    grid = cv2.bitwise_and(hor, ver)
    pts = cv2.findNonZero(grid)
    ys = sorted({p[0][1] for p in pts})
    xs = sorted({p[0][0] for p in pts})

    def group(coords, tol=10):
        groups, cur = [], [coords[0]]
        for c in coords[1:]:
            if abs(c-cur[-1]) <= tol:
                cur.append(c)
            else:
                groups.append(cur)
                cur = [c]
        groups.append(cur)
        return [int(sum(g)/len(g)) for g in groups]

    y_lines = group(ys)
    x_lines = group(xs)

    # --- gap 보정 x_lines ---
    x_gaps = np.diff(x_lines)
    mean_gap = np.median(x_gaps)
    new_x = [x_lines[0]]
    for a, b in zip(x_lines, x_lines[1:]):
        gap = b - a
        if gap > mean_gap * 1.5:
            new_x.append(int((a + b) // 2))
        new_x.append(b)
    x_lines = sorted(set(new_x))

    # --- gap 보정 y_lines ---
    y_gaps = np.diff(y_lines)
    mean_gap_y = np.median(y_gaps)
    new_y = [y_lines[0]]
    for a, b in zip(y_lines, y_lines[1:]):
        gap = b - a
        if gap > mean_gap_y * 1.5:
            new_y.append(int((a + b) // 2))
        new_y.append(b)
    y_lines = sorted(set(new_y))

    data_ys = y_lines[skip_rows:]
    data_xs = x_lines[skip_cols:]
    row_ranges = [(data_ys[i],   data_ys[i+1]) for i in range(len(data_ys)-1)]
    col_ranges = [(data_xs[j],   data_xs[j+1]) for j in range(len(data_xs)-1)]

    # grid 시각화 (옵션)
    grid_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for y in y_lines:
        cv2.line(grid_vis, (x_lines[0], y), (x_lines[-1], y), (0,0,255), 1)
    for x in x_lines:
        cv2.line(grid_vis, (x, y_lines[0]), (x, y_lines[-1]), (0,0,255), 1)
    plt.imshow(grid_vis)
    plt.title("grid 검출 결과 (gap 보정 포함)")
    plt.axis('off')
    plt.show()

    return gray, row_ranges, col_ranges


# ─────────────────────────────
# 3. tight crop + OCR 분류 함수(기존과 동일)
# ─────────────────────────────
def tight_crop_and_resize(cell_img, out_size=(32,32), extra_crop=3):
    if cell_img is None or cell_img.size == 0:
        return np.full(out_size, 255, dtype=np.uint8)
    _, threshed = cv2.threshold(cell_img, 200, 255, cv2.THRESH_BINARY)
    inv = 255 - threshed
    coords = cv2.findNonZero(inv)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        x1 = max(x + extra_crop, 0)
        y1 = max(y + extra_crop, 0)
        x2 = min(x + w - extra_crop, cell_img.shape[1])
        y2 = min(y + h - extra_crop, cell_img.shape[0])
        if x2 > x1 and y2 > y1:
            cropped = cell_img[y1:y2, x1:x2]
        else:
            cropped = cell_img[y:y+h, x:x+w]
    else:
        cropped = cell_img
    resized = cv2.resize(cropped, out_size)
    return resized

def classify_dne(cell_img, interpreter, input_details, output_details, debug=False, r=0, c=0):
    tight_img = tight_crop_and_resize(cell_img)
    white_ratio = np.mean(tight_img > 180)
    if debug:
        cv2.imwrite(f'debug_cell_r{r}_c{c}.png', tight_img)
    if white_ratio > WHITE_THRESH:
        if debug: print("Blank cell detected by white_ratio:", white_ratio)
        return '-'
    h, w = input_details[0]['shape'][1:3]
    inp = tight_img.astype(np.float32) / 255.0
    sample = inp.reshape(1, h, w, 1)
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])[0]
    max_prob = np.max(out)
    if debug:
        print(f"Probabilities: D={out[0]:.3f}, N={out[1]:.3f}, E={out[2]:.3f} (max={max_prob:.3f})")
    if max_prob < 0.8:
        if debug: print("Blank cell detected by prob:", max_prob)
        return '-'
    return CLASS_LABELS[np.argmax(out)]

# ─────────────────────────────
# 4. 메인 실행 (셀 분할+OCR+시각화)
# ─────────────────────────────
def main():
    # (1) 표 워핑
    warped = detect_and_warp_table(IMAGE_PATH)
    # (2) robust grid기반 셀 좌표 추출
    gray, data_rows, data_cols = get_table_cells(warped, skip_rows=HEADER_SKIP_Y, skip_cols=HEADER_SKIP_X)
    # (3) 모델 준비
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # (4) 셀별 분류 및 저장
    result = {}
    debug_count = 0
    n_rows, n_cols = len(data_rows), len(data_cols)
    plt.figure(figsize=(n_cols*1.5, n_rows*1.5))
    for r, (y1, y2) in enumerate(data_rows):
        row_dict = {}
        for c, (x1, x2) in enumerate(data_cols):
            cell_img = gray[y1:y2, x1:x2]
            debug = (debug_count < 20)
            val = classify_dne(cell_img, interpreter, input_details, output_details,
                               debug=debug, r=r, c=c)
            row_dict[str(c+1)] = val
            if debug: debug_count += 1
            # 셀 시각화
            ax = plt.subplot(n_rows, n_cols, r*n_cols + c + 1)
            plt.imshow(cell_img, cmap='gray')
            plt.title(val)
            plt.axis('off')
        result[str(r+1)] = row_dict
    plt.tight_layout()
    plt.show()

    # (5) 결과 저장
    with open("schedule_inferred4.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("✅ schedule_inferred4.json 저장 완료")
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ─────────────────────────────
# 5. 실행
# ─────────────────────────────
if __name__ == '__main__':
    main()
