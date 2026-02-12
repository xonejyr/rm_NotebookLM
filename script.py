import fitz  # PyMuPDF
import numpy as np
import cv2
import io

# --- 核心算法函数：底色重建与纹理注入 ---

def generate_4way_gradient(target_rgb):
    """
    基于目标区域边缘，生成 4 向插值的光滑色彩底色。
    """
    h, w = target_rgb.shape[:2]
    img = target_rgb.astype(float)
    
    # 提取四条边界的像素行/列
    top, bot = img[0, :], img[-1, :]
    lef, rig = img[:, 0], img[:, -1]

    base = np.zeros((h, w, 3), dtype=float)
    # 创建坐标网格
    v_coords = np.linspace(0, 1, h)
    u_coords = np.linspace(0, 1, w)
    uu, vv = np.meshgrid(u_coords, v_coords)

    for c in range(3): # 处理 R, G, B 三个通道
        # 水平插值分量
        color_h = (1 - uu) * lef[:, c][:, np.newaxis] + uu * rig[:, c][:, np.newaxis]
        # 垂直插值分量
        color_v = (1 - vv) * top[:, c][np.newaxis, :] + vv * bot[:, c][np.newaxis, :]
        # 融合
        base[:, :, c] = (color_h + color_v) / 2
    return base

def get_best_texture_source(page, target_rect, zoom=3):
    """
    8邻域方差扫描：寻找周围最干净的背景块。
    """
    w, h = target_rect.width, target_rect.height
    mat = fitz.Matrix(zoom, zoom)
    gap = 4 # 避开原水印边缘的采样间隙
    
    # 候选偏移：上、下、左、右及四个斜角
    candidates = [
        (0, -h-gap), (0, h+gap), (-w-gap, 0), (w+gap, 0),
        (-w-gap, -h-gap), (w+gap, -h-gap), (-w-gap, h+gap), (w+gap, h+gap)
    ]
    
    best_img = None
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
        
        # 计算方差：方差越小说明越“白净”，没有文字干扰
        current_var = np.var(gray)
        if current_var < min_var:
            min_var = current_var
            best_img = img
            
    return best_img

def apply_texture(smooth_base, texture_src):
    """
    高频纹理传递：将参考区的噪点注入到光滑基底。
    """
    h, w = smooth_base.shape[:2]
    # 调整纹理源大小以匹配补丁
    src_res = cv2.resize(texture_src, (w, h), interpolation=cv2.INTER_CUBIC).astype(float)
    
    # 提取高频细节：原图 - 低通滤波图
    low_freq = cv2.GaussianBlur(src_res, (15, 15), 0)
    texture_residual = src_res - low_freq
    
    # 融合并限幅
    k = 0.5 # 颗粒感
    combined = np.clip(smooth_base + k * texture_residual, 0, 255).astype(np.uint8)
    return combined

# --- 主逻辑：PDF 处理流程 ---

def run_ultimate_replacement(input_pdf, output_pdf, start_x, start_y):
    doc = fitz.open(input_pdf)
    width, height = 100, 15
    zoom = 3 # 纹理采样精度
    
    print(f"开始处理 PDF，坐标点: ({start_x}, {start_y})")

    for i, page in enumerate(doc):
        # 坐标预检：如果坐标超过页面宽高，发出警告
        if start_x > page.rect.width or start_y > page.rect.height:
            print(f"警告：第 {i+1} 页坐标 ({start_x}, {start_y}) 超出页面范围 {page.rect.width}x{page.rect.height}")

        target_rect = fitz.Rect(start_x, start_y, start_x + width, start_y + height)

        # 1. 采集目标位置的边缘梯度信息
        pix_target = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=target_rect)
        target_img = np.frombuffer(pix_target.samples, dtype=np.uint8).reshape(pix_target.height, pix_target.width, 3)

        # 2. 自动寻找最佳纹理参考区
        texture_src = get_best_texture_source(page, target_rect, zoom)
        
        # 3. 生成底色基底
        smooth_base = generate_4way_gradient(target_img)
        
        # 4. 融合纹理（如果找到了纹理源）
        if texture_src is not None:
            final_patch_rgb = apply_texture(smooth_base, texture_src)
        else:
            final_patch_rgb = smooth_base.astype(np.uint8)

        # 5. 贴回 PDF
        # 注意：OpenCV 使用 BGR，PyMuPDF/PNG 使用 RGB
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(final_patch_rgb, cv2.COLOR_RGB2BGR))
        if is_success:
            page.insert_image(target_rect, stream=io.BytesIO(buffer).getvalue())

        # 6. 插入文字 (靖宇)
        text_to_add = "靖宇"
        font_size = 15.0
        # 使用内置宋体风格
        tw = fitz.get_text_length(text_to_add, fontname="china-s", fontsize=font_size)
        
        # 精确居中计算
        text_x = start_x + (width - tw) / 2
        text_y = start_y + 13 # 15高度下的经验视觉基线
        
        page.insert_text((text_x, text_y), text_to_add, 
                         fontsize=font_size, fontname="china-s", color=(0, 0, 0))

    doc.save(output_pdf, garbage=4, deflate=True)
    doc.close()
    print(f"处理成功！输出文件: {output_pdf}")

# --- 执行入口 ---
if __name__ == "__main__":
    # 请确保文件名正确
    run_ultimate_replacement("Family_Pride_Algorithm.pdf", "final_output_perfect.pdf", 1268, 745)