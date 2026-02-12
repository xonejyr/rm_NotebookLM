import fitz  # PyMuPDF
import cv2   # OpenCV
import numpy as np
import io

def generate_feathered_mask(shape, padding_x, padding_y, blur_radius):
    """
    生成一个边缘柔和羽化的 Alpha 遮罩。
    中心区域白色（不透明），边缘逐渐过渡到黑色（透明）。
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    # 绘制中心白色矩形，留出羽化边距
    cv2.rectangle(mask, (padding_x, padding_y), (w - padding_x, h - padding_y), (255), -1)
    # 使用高斯模糊制造柔和边界
    # blur_radius 越大，边界越模糊，融合越好
    feathered_mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
    return feathered_mask

def ultimate_seamless_replace(pdf_path, output_path, start_x, start_y):
    doc = fitz.open(pdf_path)
    
    # --- 精确的目标数据 ---
    target_w = 100
    target_h = 15
    
    # --- 关键参数调优 ---
    # 1. 上下文范围：向外扩展多少像素用于采样背景和融合
    # 范围越大，渐变过渡越自然，建议 10-20
    context_margin = 15 
    
    # 2. 羽化半径：控制边界的模糊程度。必须是奇数。
    # 越大边界越不可见，但太大会导致中心遮盖力下降。建议 11-21。
    feather_blur = 15 
    
    # 定义包含周围环境的大区域
    context_rect = fitz.Rect(start_x - context_margin, start_y - context_margin, 
                             start_x + target_w + context_margin, start_y + target_h + context_margin)

    print(f"启动终极融合引擎。处理文档共 {len(doc)} 页...")

    for page_num, page in enumerate(doc):
        # --- 步骤 1: 高清采集上下文环境 ---
        # 使用 2 倍缩放采集，保证纹理细节
        zoom = 2
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=context_rect)
        img_bgra = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
        
        # --- 步骤 2: 制造完美的“干净基底” (Inpainting) ---
        # 创建修复蒙版
        inpaint_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        # 计算需要擦除的水印区域在高清图中的坐标
        mx1 = int((context_margin) * zoom)
        my1 = int((context_margin) * zoom)
        mx2 = int((context_margin + target_w) * zoom)
        my2 = int((context_margin + target_h) * zoom)
        # 稍微内缩一点点，确保不误伤外部纹理
        cv2.rectangle(inpaint_mask, (mx1+2, my1+2), (mx2-2, my2-2), (255), -1)
        
        # 使用 Navier-Stokes 算法进行修复 (比 Telea 通常更平滑，适合渐变背景)
        clean_base_bgr = cv2.inpaint(img_bgr, inpaint_mask, 3, cv2.INPAINT_NS)
        clean_base_rgb = cv2.cvtColor(clean_base_bgr, cv2.COLOR_BGR2RGB)

        # --- 步骤 3: 核心魔法 - 创建柔焦 Alpha 通道 ---
        # 我们不直接贴这个干净图，而是把它做成一个边缘透明的贴纸
        
        # 计算羽化遮罩的内缩边距
        pad_x = int(context_margin * zoom * 0.5) # 让羽化区只发生在 margin 区域内
        pad_y = int(context_margin * zoom * 0.5)
        
        # 生成柔和的 alpha mask
        alpha_channel = generate_feathered_mask(clean_base_rgb.shape, pad_x, pad_y, feather_blur)
        
        # 合并 RGB 和 Alpha 通道，生成 RGBA 图像
        r, g, b = cv2.split(clean_base_rgb)
        rgba_patch = cv2.merge([r, g, b, alpha_channel])

        # --- 步骤 4: 将柔和贴片融合回 PDF ---
        # 编码为 PNG 以保留 Alpha 通道
        is_success, png_buffer = cv2.imencode(".png", rgba_patch)
        if not is_success: continue
        png_stream = io.BytesIO(png_buffer)

        # 插入这个带有柔和边缘的图像。
        # PDF渲染器会自动处理 Alpha 混合，使边界消失。
        page.insert_image(context_rect, stream=png_stream.getvalue())

        # --- 步骤 5: 写入高质量矢量文字 ---
        my_text = "靖宇"
        f_size, f_name = 15.0, "china-s"
        text_w = fitz.get_text_length(my_text, fontname=f_name, fontsize=f_size)
        # 精确居中计算
        text_x = start_x + (target_w - text_w) / 2
        text_y = start_y + 13
        # 确保文字在最上层
        page.insert_text((text_x, text_y), my_text, fontsize=f_size, fontname=f_name, color=(0, 0, 0), overlay=True)

    # 使用高压缩保存，并清理冗余数据
    doc.save(output_path, garbage=4, deflate=True)
    print(f"终极融合完成。输出文件: {output_path}")

# --- 执行 ---
# 请替换你的文件名
ultimate_seamless_replace("Family_Pride_Algorithm.pdf", "final_ultimate_output.pdf", 1268, 745)