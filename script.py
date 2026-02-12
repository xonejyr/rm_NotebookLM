import fitz  # PyMuPDF
import numpy as np
import cv2
import io

def generate_feathered_alpha(h, w, feather_radius=5):
    """
    创建一个边缘柔和羽化的 Alpha 通道蒙版。
    """
    # 初始全白（不透明）
    mask = np.ones((h, w), dtype=np.float32) * 255
    
    # 在边缘绘制黑色矩形，用于生成渐变
    # 这里的关键是：我们让最外圈彻底透明
    mask = cv2.copyMakeBorder(mask[feather_radius:-feather_radius, feather_radius:-feather_radius], 
                               feather_radius, feather_radius, feather_radius, feather_radius, 
                               cv2.BORDER_CONSTANT, value=0)
    
    # 高斯模糊，将黑白边界变成柔和的渐变
    mask = cv2.GaussianBlur(mask, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
    return mask.astype(np.uint8)

def ultimate_feathered_replace(pdf_path, start_x, start_y):
    doc = fitz.open(pdf_path)
    width, height = 105, 15
    zoom = 3 # 保持高清采样
    mat = fitz.Matrix(zoom, zoom)
    
    # 羽化半径：控制边缘融合的宽度（3-5像素效果最好）
    feather_r = 3 

    for i, page in enumerate(doc):
        # 扩展采样区：为了实现羽化，我们需要采一个比 100x15 稍微大一点的区域
        # 这样羽化带就能完美衔接外部背景
        context_rect = fitz.Rect(start_x - feather_r, start_y - feather_r, 
                                 start_x + width + feather_r, start_y + height + feather_r)
        
        # 1. 采集图像
        pix = page.get_pixmap(matrix=mat, clip=context_rect)
        img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        
        # 2. 生成内部逻辑（4向梯度 + 纹理注入）
        # 这里直接在 img_rgb 的基础上生成（由于采样了大一圈，边缘更自然）
        from_v3 = generate_4way_gradient(img_rgb) # 这里复用上一版的函数
        
        # 提取纹理（假设取左侧邻域作为纹理源，避免污染）
        tex_src_rect = fitz.Rect(start_x - width - 5, start_y, start_x - 5, start_y + height)
        pix_tex = page.get_pixmap(matrix=mat, clip=tex_src_rect)
        tex_rgb = np.frombuffer(pix_tex.samples, dtype=np.uint8).reshape(pix_tex.height, pix_tex.width, 3)
        
        # 融合底色与纹理
        combined_rgb = apply_texture(from_v3, tex_rgb) # 这里复用上一版的函数

        # 3. 核心：添加 Alpha 羽化通道
        h_p, w_p = combined_rgb.shape[:2]
        alpha = generate_feathered_alpha(h_p, w_p, feather_radius=feather_r * zoom)
        
        # 合并为 RGBA
        r, g, b = cv2.split(combined_rgb)
        rgba_patch = cv2.merge([r, g, b, alpha])

        # 4. 贴回 PDF (使用 context_rect 确保羽化边缘覆盖到正确位置)
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(rgba_patch, cv2.COLOR_RGBA2BGRA))
        if is_success:
            page.insert_image(context_rect, stream=io.BytesIO(buffer).getvalue())

        # 5. 写入文字
        text = "靖宇"
        tw = fitz.get_text_length(text, fontname="china-s", fontsize=15.0)
        page.insert_text((start_x + (width - tw)/2, start_y + 13), text, 
                         fontsize=15.0, fontname="china-s", color=(0, 0, 0))
        
    name = pdf_path.rsplit(".pdf", 1)[0]
    output_path = f"{name}_{text}.pdf"
    doc.save(output_path, garbage=4, deflate=True)
    print(f"羽化融合完成，边界感已消除: {output_path}")

# --- 辅助函数：确保代码完整可运行 ---
def generate_4way_gradient(img_rgb):
    h, w = img_rgb.shape[:2]
    img = img_rgb.astype(float)
    top, bot, lef, rig = img[0,:], img[-1,:], img[:,0], img[:,-1]
    base = np.zeros((h, w, 3), dtype=float)
    v_coords, u_coords = np.linspace(0, 1, h), np.linspace(0, 1, w)
    uu, vv = np.meshgrid(u_coords, v_coords)
    for c in range(3):
        ch = (1-uu)*lef[:,c][:,np.newaxis] + uu*rig[:,c][:,np.newaxis]
        cv = (1-vv)*top[:,c][np.newaxis,:] + vv*bot[:,c][np.newaxis,:]
        base[:,:,c] = (ch + cv) / 2
    return base

def apply_texture(smooth_base, texture_src):
    h, w = smooth_base.shape[:2]
    src_res = cv2.resize(texture_src, (w, h), interpolation=cv2.INTER_CUBIC).astype(float)
    low_freq = cv2.GaussianBlur(src_res, (15, 15), 0)
    texture_residual = src_res - low_freq
    k = 0.4 # noise
    return np.clip(smooth_base + k * texture_residual, 0, 255).astype(np.uint8)

# 执行
ultimate_feathered_replace("GeoFlow_Precision_Visualization.pdf", 1265, 746)