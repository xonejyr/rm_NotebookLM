import fitz  # PyMuPDF
import numpy as np
import cv2
import io

def four_way_gradient_replace(pdf_path, output_path, start_x, start_y):
    doc = fitz.open(pdf_path)
    width, height = 105, 15  # 你的目标尺寸

    for page in doc:
        # 1. 采样：取一个比目标大 1 像素的框，为了抓取 4 条边的原生颜色
        sample_rect = fitz.Rect(start_x - 1, start_y - 1, start_x + width + 1, start_y + height + 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=sample_rect)
        
        # 转换为 numpy (RGB 格式，确保不偏色)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        
        # 提取 4 条边界的真实像素值 (在高分辨率 2x 下坐标翻倍)
        h_img, w_img = img.shape[:2]
        top_line = img[0, 1:-1].astype(float)
        bot_line = img[-1, 1:-1].astype(float)
        lef_line = img[1:-1, 0].astype(float)
        rig_line = img[1:-1, -1].astype(float)

        # 2. 构造 4 向渐变场
        # 创建一个空补丁 (内缩回 100x15 的原始比例)
        rows, cols = h_img - 2, w_img - 2
        patch = np.zeros((rows, cols, 3), dtype=float)

        # 使用双线性融合公式：每一像素都是 4 条边的权重贡献
        for r in range(rows):
            for c in range(cols):
                # 计算权重 (0 到 1)
                u = c / (cols - 1) if cols > 1 else 0.5 # 水平进度
                v = r / (rows - 1) if rows > 1 else 0.5 # 垂直进度
                
                # 水平方向插值结果
                color_h = (1 - u) * lef_line[r] + u * rig_line[r]
                # 垂直方向插值结果
                color_v = (1 - v) * top_line[c] + v * bot_line[c]
                
                # 核心：将水平和垂直结果再次融合
                # 这样消灭了单一方向的“撞车”，让颜色在二维平面平滑过渡
                patch[r, c] = (color_h + color_v) / 2

        # 3. 贴回 PDF
        final_patch = patch.astype(np.uint8)
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(final_patch, cv2.COLOR_RGB2BGR))
        if is_success:
            page.insert_image(fitz.Rect(start_x, start_y, start_x + width, start_y + height), 
                             stream=io.BytesIO(buffer).getvalue())

        # 4. 写入文字 (靖宇)
        text = "靖宇乱说"
        f_size, f_name = 15.0, "china-s"
        tw = fitz.get_text_length(text, fontname=f_name, fontsize=f_size)
        page.insert_text((start_x + (width - tw)/2, start_y + 13), 
                         text, fontsize=f_size, fontname=f_name, color=(0, 0, 0))

    doc.save(output_path)
    print(f"4向梯度填充完成。保存在: final_no_seam.pdf")

# 执行
four_way_gradient_replace("Family_Pride_Algorithm.pdf", "final_no_seam.pdf", 1265, 745)