import fitz  # PyMuPDF
import numpy as np
import cv2
import io

# --- 1. 核心算法函数 ---

def generate_4way_gradient(img_rgb):
    """4向插值底色层 (解决渐变分界)"""
    h, w = img_rgb.shape[:2]
    img = img_rgb.astype(float)
    top, bot, lef, rig = img[0,:], img[-1,:], img[:,0], img[:,-1]
    base = np.zeros((h, w, 3), dtype=float)
    uu, vv = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    for c in range(3):
        ch = (1-uu)*lef[:,c][:,np.newaxis] + uu*rig[:,c][:,np.newaxis]
        cv = (1-vv)*top[:,c][np.newaxis,:] + vv*bot[:,c][np.newaxis,:]
        base[:,:,c] = (ch + cv) / 2
    return base

def find_cleanest_texture_src(page, target_rect, zoom=3):
    """
    自适应寻源：在上、下、左、右四个方向扫描，寻找方差最小（最干净）的背景
    """
    w, h = target_rect.width, target_rect.height
    mat = fitz.Matrix(zoom, zoom)
    gap = 1  # 采样间隙，避开边缘
    
    # 候选偏移 (dx, dy)
    candidates = [
        (0, -h-gap), # Top
        (0, h+gap),  # Bottom
        (-w-gap, 0), # Left
        (w+gap, 0)   # Right
    ]
    
    best_texture = None
    min_var = float('inf')
    
    for dx, dy in candidates:
        cand_rect = target_rect + (dx, dy, dx, dy)
        # 边界检查
        if not (0 <= cand_rect.x0 and cand_rect.x1 <= page.rect.width and 
                0 <= cand_rect.y0 and cand_rect.y1 <= page.rect.height):
            continue
            
        pix = page.get_pixmap(matrix=mat, clip=cand_rect)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 计算局部方差：方差越小，区域越“白净”
        current_var = np.var(gray)
        if current_var < min_var:
            min_var = current_var
            best_texture = img
            
    return best_texture

def apply_texture(smooth_base, texture_src, k=0.4):
    """高频纹理传递层 (解决颗粒感)"""
    h, w = smooth_base.shape[:2]
    src_res = cv2.resize(texture_src, (w, h), interpolation=cv2.INTER_CUBIC).astype(float)
    low_freq = cv2.GaussianBlur(src_res, (15, 15), 0)
    texture_residual = src_res - low_freq
    return np.clip(smooth_base + k * texture_residual, 0, 255).astype(np.uint8)

def generate_feathered_alpha(h, w, feather_radius=9):
    """Alpha羽化层 (解决边界感)"""
    mask = np.ones((h, w), dtype=np.float32) * 255
    r = min(feather_radius, h // 2 - 1, w // 2 - 1)
    mask = cv2.copyMakeBorder(mask[r:-r, r:-r], r, r, r, r, cv2.BORDER_CONSTANT, value=0)
    k_size = r * 2 + 1
    mask = cv2.GaussianBlur(mask, (k_size, k_size), 0)
    return mask.astype(np.uint8)

# --- 2. 主流程 ---

def ultimate_adaptive_replace(pdf_path, start_x, start_y):
    doc = fitz.open(pdf_path)
    width, height = 105, 15
    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
    feather_r = 3 
    
    replace_text = "靖宇"

    for i, page in enumerate(doc):
        # 扩展采样区（为了羽化衔接）
        context_rect = fitz.Rect(start_x - feather_r, start_y - feather_r, 
                                 start_x + width + feather_r, start_y + height + feather_r)
        
        # 1. 采集上下文底色梯度
        pix_ctx = page.get_pixmap(matrix=mat, clip=context_rect)
        img_rgb = np.frombuffer(pix_ctx.samples, dtype=np.uint8).reshape(pix_ctx.height, pix_ctx.width, 3)
        
        # 2. 【高明点】自适应寻找周围最干净的背景作为纹理源
        # 自动避开文字、线条、表格边框
        target_rect = fitz.Rect(start_x, start_y, start_x + width, start_y + height)
        texture_src = find_cleanest_texture_src(page, target_rect, zoom)
        
        # 3. 混合生成
        smooth_base = generate_4way_gradient(img_rgb)
        if texture_src is not None:
            # 使用找到的最干净纹理进行注入
            combined_rgb = apply_texture(smooth_base, texture_src, k=0.4)
        else:
            combined_rgb = smooth_base.astype(np.uint8)

        # 4. 生成 Alpha 通道融合
        h_p, w_p = combined_rgb.shape[:2]
        alpha = generate_feathered_alpha(h_p, w_p, feather_radius=feather_r * zoom)
        rgba_patch = cv2.merge([cv2.split(combined_rgb)[0], cv2.split(combined_rgb)[1], cv2.split(combined_rgb)[2], alpha])

        # 5. 贴回与刻字
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(rgba_patch, cv2.COLOR_RGBA2BGRA))
        if is_success:
            page.insert_image(context_rect, stream=io.BytesIO(buffer).getvalue())

        tw = fitz.get_text_length(replace_text, fontname="china-s", fontsize=15.0)
        page.insert_text((start_x + (width - tw)/2, start_y + 13), replace_text, 
                         fontsize=15.0, fontname="china-s", color=(0, 0, 0))

    name_base = pdf_path.rsplit(".pdf", 1)[0]
    output_path = f"{name_base}_{replace_text[:2]}.pdf"
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    print(f"自适应处理完成: {output_path}")

# --- 执行 ---
if __name__ == "__main__":
    ultimate_adaptive_replace("GeoFlow_Precision_Visualization.pdf", 1265, 746)