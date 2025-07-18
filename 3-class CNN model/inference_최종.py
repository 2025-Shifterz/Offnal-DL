import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from google.colab import files

img_path = "/content/schedule9.jpeg"
image = cv2.imread(img_path)

if image is None:
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {img_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.title("이미지 확인")
plt.axis('off')
plt.show()

# 크기 제한 (선택사항)
max_width = 1000
if image.shape[1] > max_width:
    scale = max_width / image.shape[1]
    image = cv2.resize(image, None, fx=scale, fy=scale)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8
)

plt.imshow(thresh, cmap='gray')
plt.title("이진화된 이미지 (Adaptive Threshold)")
plt.axis('off')
plt.show()

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

table_contour = None
for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        table_contour = approx
        break

if table_contour is None:
    print("❌ 표를 찾지 못했습니다.")
else:
    temp = image.copy()
    cv2.drawContours(temp, [table_contour], -1, (0, 255, 0), 3)
    plt.imshow(temp)
    plt.title("표 외곽 검출 결과")
    plt.axis('off')
    plt.show()

def sort_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

# 정렬 좌표 계산
rect = sort_points(table_contour)
(tl, tr, br, bl) = rect
widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
maxWidth = max(int(widthA), int(widthB))
heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
maxHeight = max(int(heightA), int(heightB))

# Warp
dst = np.array([[0, 0], [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

plt.imshow(warped)
plt.title("Warped Image (정렬 완료)")
plt.axis('off')
plt.show()

# ─────────────────────────────
# 셀 crop + 리사이즈 함수
# ─────────────────────────────
def tight_crop_and_resize(cell_img, out_size=(32,32)):
    if cell_img is None or cell_img.size == 0:
        return np.full(out_size, 255, dtype=np.uint8)
    _, threshed = cv2.threshold(cell_img, 200, 255, cv2.THRESH_BINARY)
    inv = 255 - threshed
    coords = cv2.findNonZero(inv)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = cell_img[y:y+h, x:x+w]
    else:
        cropped = cell_img
    resized = cv2.resize(cropped, out_size)
    return resized


# ─────────────────────────────
# 셀 단일 분류 함수 (TFLite 기반)
# ─────────────────────────────
def classify_dne(cell_img, interpreter, input_details, output_details, debug=False, r=0, c=0):
    tight_img = tight_crop_and_resize(cell_img)
    white_ratio = np.mean(tight_img > 220)
    if debug:
        cv2.imwrite(f'debug_cell_r{r}_c{c}.png', tight_img)
    if white_ratio > WHITE_THRESH:
        if debug: print("Blank cell detected: white_ratio=", white_ratio)
        return '-'
    h, w = input_details[0]['shape'][1:3]
    inp = tight_img.astype(np.float32) / 255.0
    sample = inp.reshape(1, h, w, 1)
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])[0]
    if debug: print(f"Probabilities: D={out[0]:.3f}, N={out[1]:.3f}, E={out[2]:.3f}")
    idx = np.argmax(out)
    label = CLASS_LABELS[idx]
    if debug: print("Predicted label:", label)
    return label


# ─────────────────────────────
# 메인 실행 함수 (전체 파이프라인)
# ─────────────────────────────
def main():
    # (1) 셀 경계 자동 추출 및 왜곡 보정
    img, data_rows, data_cols = get_table_cells(
        IMAGE_PATH, header_skip_y=HEADER_SKIP_Y, header_skip_x=HEADER_SKIP_X,
        vis_path="table_detected4.png"
    )

    # (2) TFLite 모델 준비
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # (3) 셀 순회하며 분류
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

    # (4) JSON 저장
    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    print(json_str)
    with open('schedule_inferred4.json', 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("✅ schedule_inferred4.json 생성 완료")


# ─────────────────────────────
# 실행
# ─────────────────────────────
if __name__ == '__main__':
    main()
