import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from collections import deque

# 參數設置
GAUSSIAN_BLUR_KERNEL = (11, 11)
SOBEL_KERNEL_SIZE = 3
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

def preprocess_image(image_path):
    # 讀取圖像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)
    
    # Sobel 邊緣檢測
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=SOBEL_KERNEL_SIZE)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.convertScaleAbs(sobel)
    
    # LBP 特徵提取
    lbp = local_binary_pattern(sobel, LBP_POINTS, LBP_RADIUS, method="uniform")
    lbp = (lbp / lbp.max() * 255).astype("uint8")
    
    return image, lbp

def bfs_road_detection(lbp_image):
    # 假設馬路區域較大，從圖像底部中心開始搜索
    h, w = lbp_image.shape
    start_point = (h - 1, w // 2)
    queue = deque([start_point])
    visited = np.zeros_like(lbp_image, dtype=bool)
    road_mask = np.zeros_like(lbp_image, dtype=np.uint8)

    while queue:
        x, y = queue.popleft()
        if not visited[x, y] and lbp_image[x, y] < 170:  # 假設馬路像素值較低
            visited[x, y] = True
            road_mask[x, y] = 255
            
            # BFS 四個鄰居
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                    queue.append((nx, ny))

    return road_mask

def overlay_road(image, road_mask):
    # 創建綠色透明圖層
    overlay = image.copy()
    overlay[road_mask == 255] = (0, 255, 0)
    output = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    return output

def main(image_path):
    # 圖像預處理
    image, lbp_image = preprocess_image(image_path)
    
    # 使用BFS檢測馬路區域
    road_mask = bfs_road_detection(lbp_image)
    
    # 疊加綠色透明圖層
    result_image = overlay_road(image, road_mask)
    
    # 顯示結果
    cv2.imshow("Result", road_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 跑程式
image_path = 'road.jpg'
main(image_path)