# ─────────────────────────────
# 0. 패키지 임포트 및 설정
# ─────────────────────────────
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow as tf

# 사용자 설정
IMAGE_PATH   = "/content/camera3.jpg"       # 업로드한 근무표 이미지 경로
TFLITE_MODEL = "dne_classifier.tflite"         # TFLite 분류기 모델
WHITE_THRESH = 0.94                           # 빈 셀 판단 기준
CLASS_LABELS = ['D', 'N', 'E']                 # 분류 클래스
HEADER_SKIP_Y = 2                              # 위쪽 헤더 행 수
HEADER_SKIP_X = 1                              # 왼쪽 헤더 열 수

# ─────────────────────────────
# 1. 셀 좌표 검출 및 표 정렬
# ─────────────────────────────
def get_table_cells(img_path, header_skip_y=2, header_skip_x=1, vis_path="table_detected.png"):
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
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    else:
        print("⚠️ 표 외곽 인식 실패 — 원본 사용")
        warped = orig_img
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    def find_seps(mask, axis, ratio=0.5):
        proj = mask.sum(axis=0) if axis == 'x' else mask.sum(axis=1)
        thresh = proj.max() * ratio
        idx = np.where(proj > thresh)[0]
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        return sorted(int(np.median(g)) for g in groups if len(g) > 0)

    x_seps = find_seps(v_lines, 'x', 0.5)
    y_seps = find_seps(h_lines, 'y', 0.5)

    if vis_path:
        img_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for y in y_seps:
            cv2.line(img_vis, (x_seps[0], y), (x_seps[-1], y), (0,0,255), 2)
        for x in x_seps:
            cv2.line(img_vis, (x, y_seps[0]), (x, y_seps[-1]), (0,0,255), 2)
        cv2.imwrite(vis_path, img_vis)
        print(f"✅ 셀 시각화 저장 완료: {vis_path}")

    data_rows = list(zip(y_seps[header_skip_y-1:-1], y_seps[header_skip_y:]))
    data_cols = list(zip(x_seps[header_skip_x:-1], x_seps[header_skip_x+1:]))
    return gray, data_rows, data_cols

# ─────────────────────────────
# 2. 셀 crop 및 리사이즈
# ─────────────────────────────
def tight_crop_and_resize(cell_img, out_size=(32,32), extra_crop=3):
    if cell_img is None or cell_img.size == 0:
        return np.full(out_size, 255, dtype=np.uint8)
    _, threshed = cv2.threshold(cell_img, 200, 255, cv2.THRESH_BINARY)
    inv = 255 - threshed
    coords = cv2.findNonZero(inv)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # 더 타이트하게 자르기 (상하좌우 extra_crop 만큼 추가로 크롭)
        x1 = max(x + extra_crop, 0)
        y1 = max(y + extra_crop, 0)
        x2 = min(x + w - extra_crop, cell_img.shape[1])
        y2 = min(y + h - extra_crop, cell_img.shape[0])
        if x2 > x1 and y2 > y1:
            cropped = cell_img[y1:y2, x1:x2]
        else:
            cropped = cell_img[y:y+h, x:x+w]  # 만약 잘못 잘라지면 fallback
    else:
        cropped = cell_img
    resized = cv2.resize(cropped, out_size)
    return resized


# ─────────────────────────────
# 3. 셀 분류 함수
# ─────────────────────────────
def classify_dne(cell_img, interpreter, input_details, output_details, debug=False, r=0, c=0):
    tight_img = tight_crop_and_resize(cell_img)
    white_ratio = np.mean(tight_img > 180)
    if debug:
        cv2.imwrite(f'debug_cell_r{r}_c{c}.png', tight_img)
    # 1. white_ratio로 먼저 빈칸 필터링
    if white_ratio > WHITE_THRESH:
        if debug: print("Blank cell detected by white_ratio:", white_ratio)
        return '-'
    # 2. 모델 예측
    h, w = input_details[0]['shape'][1:3]
    inp = tight_img.astype(np.float32) / 255.0
    sample = inp.reshape(1, h, w, 1)
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])[0]
    max_prob = np.max(out)
    if debug:
        print(f"Probabilities: D={out[0]:.3f}, N={out[1]:.3f}, E={out[2]:.3f} (max={max_prob:.3f})")
    # 3. max 확률 0.7 미만이면 빈칸으로 처리
    if max_prob < 0.7:
        if debug: print("Blank cell detected by prob:", max_prob)
        return '-'
    # 4. 확률이 충분히 높으면 클래스 리턴
    return CLASS_LABELS[np.argmax(out)]


# ─────────────────────────────
# 4. 메인 실행 및 JSON 저장
# ─────────────────────────────
def main():
    img, data_rows, data_cols = get_table_cells(
        IMAGE_PATH, header_skip_y=HEADER_SKIP_Y, header_skip_x=HEADER_SKIP_X,
        vis_path="table_detected4.png"
    )

    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    result = {}
    debug_count = 0
    for r, (y1, y2) in enumerate(data_rows):
        row_dict = {}
        for c, (x1, x2) in enumerate(data_cols):
            cell_img = img[y1:y2, x1:x2]
            debug = (debug_count < 30)
            val = classify_dne(cell_img, interpreter, input_details, output_details,
                               debug=debug, r=r, c=c)
            row_dict[str(c+1)] = val
            if debug: debug_count += 1
        result[str(r+1)] = row_dict

    with open("schedule_inferred4.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("✅ schedule_inferred4.json 저장 완료")
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ─────────────────────────────
# 5. 실행
# ─────────────────────────────
if __name__ == '__main__':
    main()
