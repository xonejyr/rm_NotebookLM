import fitz  # PyMuPDF
import cv2   # OpenCV
import numpy as np
import io

def seamless_inpainting_replace(pdf_path, output_path, start_x, start_y):
    doc = fitz.open(pdf_path)
    
    # 目标区域定义 (你的精确数据)
    width = 100
    height = 15
    # 定义一个稍微大一点的“上下文区域”，用于给算法提供周围环境参考
    # 我们向外扩展 5 个像素
    margin = 5
    context_rect = fitz.Rect(start_x - margin, start_y - margin, 
                             start_x + width + margin, start_y + height + margin)

    print(f"开始处理 {len(doc)} 页文档，使用 OpenCV 结构化修复技术...")

    for page_num, page in enumerate(doc):
        # --- 步骤 1: 高分辨率渲染上下文区域 ---
        # 使用 3 倍缩放 (matrix=3) 以确保纹理细节足够清晰供算法分析
        zoom_matrix = fitz.Matrix(3, 3)
        pix = page.get_pixmap(matrix=zoom_matrix, clip=context_rect)
        
        # 将 pixmap 转换为 OpenCV 可处理的 numpy 数组 (BGRA 格式)
        img_bgra = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        # 转换为 BGR (OpenCV 默认格式) 丢弃 Alpha 通道
        img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

        # --- 步骤 2: 创建修复遮罩 (Inpainting Mask) ---
        # 遮罩是一个纯黑图像，只有需要修复的区域是纯白
        mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        
        # 计算目标区域在放大后的图像中的坐标
        # 注意：图像坐标系是从 context_rect 的左上角开始的
        mask_x1 = int(margin * zoom_matrix.a)
        mask_y1 = int(margin * zoom_matrix.d)
        mask_x2 = int((margin + width) * zoom_matrix.a)
        mask_y2 = int((margin + height) * zoom_matrix.d)
        
        # 绘制白色矩形作为需要“擦除”的区域
        # 稍微向内收缩 1 像素 (padding=-1)，利用外部纹理向内生长
        cv2.rectangle(mask, (mask_x1+1, mask_y1+1), (mask_x2-1, mask_y2-1), (255), -1)

        # --- 步骤 3: 执行 OpenCV 修复算法 ---
        # cv2.INPAINT_TELEA: 基于快速行进法(FMM)的修复算法，适合小区域纹理重建
        # inpaintRadius=3: 参考周围 3 像素的邻域范围
        cleaned_patch_bgr = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)

        # --- 步骤 4: 将修复后的完美补丁贴回 PDF ---
        # 将 BGR 转回 RGB
        cleaned_patch_rgb = cv2.cvtColor(cleaned_patch_bgr, cv2.COLOR_BGR2RGB)
        # 编码为 PNG 格式在内存中
        is_success, png_buffer = cv2.imencode(".png", cleaned_patch_rgb)
        if not is_success: continue
        png_bytes = io.BytesIO(png_buffer)

        # 插入图像补丁：它现在是一个完美的、带有周围渐变和噪点的背景块
        # 注意：这里插入的是包含 margin 的 context_rect 区域
        page.insert_image(context_rect, stream=png_bytes.getvalue())

        # --- 步骤 5: 写入矢量文字 (保持不变) ---
        my_text = "靖宇"
        f_size, f_name = 15.0, "china-s"
        text_w = fitz.get_text_length(my_text, fontname=f_name, fontsize=f_size)
        text_x = start_x + (width - text_w) / 2
        text_y = start_y + 13
        
        # 使用叠加模式写入文字，确保文字清晰浮于背景之上
        page.insert_text((text_x, text_y), my_text, fontsize=f_size, fontname=f_name, color=(0, 0, 0), overlay=True)
        
        print(f"Page {page_num+1}: 无痕修复完成。")

    # 保存时进行压缩优化
    doc.save(output_path, garbage=4, deflate=True)
    print(f"所有页面处理完毕。输出文件: {output_path}")

# 执行
# 替换为你的实际文件名和坐标
seamless_inpainting_replace("Family_Pride_Algorithm.pdf", "final_seamless_output.pdf", 1268, 745)