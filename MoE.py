import cv2
import numpy as np
import os


def draw_dashed_rectangle(img, pt1, pt2, color, thickness=2, dash_length=8):
    x1, y1 = pt1
    x2, y2 = pt2
    # 上下边
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    # 左右边
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)


def create_separate_academic_feature_maps(image_path, roi_x, roi_y, roi_w, roi_h, output_dir):
    if not os.path.exists(image_path):
        print(f"❌ 找不到文件: {image_path}")
        return

    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("❌ 图像解码失败")
        return

    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
    height, width = img.shape[:2]
    inset_size = 180


    os.makedirs(output_dir, exist_ok=True)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ==========================================

    # ==========================================
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 80, 180)

    edge_bg = np.zeros_like(img_rgb)
    edge_bg[:, :] = [15, 20, 40]

    edges_dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    glow_color = np.zeros_like(img_rgb)
    glow_color[edges_dilated > 0] = [0, 255, 255]
    glow_color[edges > 0] = [255, 255, 255]

    glow = cv2.GaussianBlur(glow_color, (5, 5), 0)
    img_A = cv2.addWeighted(edge_bg, 1.0, glow, 0.8, 0)
    img_A = cv2.add(img_A, glow_color)

    # ==========================================

    # ==========================================
    edges_for_cam = cv2.Canny(gray, 40, 120)
    edges_dilated_cam = cv2.dilate(edges_for_cam, np.ones((7, 7), np.uint8), iterations=2)

    base_attention = cv2.GaussianBlur(edges_dilated_cam, (101, 101), 0).astype(np.float32)
    if base_attention.max() > 0:

        base_attention = (base_attention / base_attention.max()) * 0.7  # 从 0.45 增加到 0.7

    cx, cy = roi_x + roi_w // 2, roi_y + roi_h // 2
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    sigma = max(roi_w, roi_h) * 0.8
    gaussian_focus = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)

    final_attention = base_attention + gaussian_focus * 0.2  # 从 0.8 降低到 0.2
    final_attention = np.clip(final_attention, 0, 1)

    heatmap_gray = (final_attention * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    gray_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    img_B = cv2.addWeighted(gray_3c, 0.4, heatmap_rgb, 0.6, 0)


    img_B = cv2.GaussianBlur(img_B, (5, 5), 0)

    def render_and_save(image_rgb, box_color_rgb, filename):
        disp_img = image_rgb.copy()


        crop = disp_img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        crop_res = cv2.resize(crop, (inset_size, inset_size), interpolation=cv2.INTER_CUBIC)


        disp_img[-(inset_size + 10):-10, -(inset_size + 10):-10] = crop_res


        cv2.rectangle(disp_img, (width - inset_size - 10, height - inset_size - 10),
                      (width - 10, height - 10), box_color_rgb, 3)


        draw_dashed_rectangle(disp_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), box_color_rgb, thickness=2)


        disp_img_bgr = cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR)


        save_path = os.path.join(output_dir, filename)
        cv2.imencode('.png', disp_img_bgr)[1].tofile(save_path)
        print(f"✅ 成功生成 640x640 图像并保存至: {save_path}")

    # 图A  / 图B 
    render_and_save(img_A, (0, 255, 255), '')
    render_and_save(img_B, (255, 0, 0), '')


if __name__ == '__main__':
    IMAGE_PATH = r""
    OUTPUT_DIR = r""

    # 你指定的 ROI 坐标 (保持不变)
    X, Y, W, H = 342, 422, 46, 68

    create_separate_academic_feature_maps(IMAGE_PATH, X, Y, W, H, OUTPUT_DIR)
