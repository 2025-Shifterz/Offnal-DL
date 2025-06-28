import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1) Load image as grayscale
img_path = "/content/sample_data/schedule3.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 2) Binarize (invert + Otsu)
_, binary = cv2.threshold(img, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 3) Morphology to extract thick border lines
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

# 4) Separator detection
def find_seps(mask, axis, ratio=0.5):
    proj = mask.sum(axis=0) if axis=='x' else mask.sum(axis=1)
    thresh = proj.max() * ratio
    idx = np.where(proj > thresh)[0]
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return sorted(int(np.median(g)) for g in groups if len(g) > 0)

x_seps = find_seps(v_lines, 'x', 0.5)  # 세로 분할선 좌표
y_seps = find_seps(h_lines, 'y', 0.5)  # 가로 분할선 좌표

# 5) Separator 기반 동적 split
cols = list(zip(x_seps[:-1], x_seps[1:]))  # [(x0,x1), (x1,x2), ...]
rows = list(zip(y_seps[:-1], y_seps[1:]))  # [(y0,y1), (y1,y2), ...]

header_skip = 2               # "날짜", "요일" 같은 헤더 행 개수
data_rows   = rows[header_skip:]  # 실제 근무표 데이터 행만 선택

# 6) Overlay detected separators
plt.figure(figsize=(6,6))
plt.imshow(img, cmap='gray')
for x in x_seps:
    plt.axvline(x, color='r', linewidth=0.5)
for y in y_seps:
    plt.axhline(y, color='r', linewidth=0.5)
plt.title("Dynamic Grid Overlay")
plt.axis('off')
plt.show()

# 7) Crop and visualize a few sample cells
samples = [(0, 0), (0, 1), (1, 0)]  # (row_index, col_index)
fig, axes = plt.subplots(1, len(samples), figsize=(12,4))
for ax, (r, c) in zip(axes, samples):
    y1, y2 = data_rows[r]
    x1, x2 = cols[c]
    cell = img[y1:y2, x1:x2]
    ax.imshow(cell, cmap='gray')
    ax.set_title(f"Cell ({r},{c})")
    ax.axis('off')
plt.tight_layout()
plt.show()
