import fitz  # PyMuPDF

def clone_texture_replace(pdf_path, output_path, start_x, start_y):
    doc = fitz.open(pdf_path)
    
    # 你指定的精确数据
    width = 100
    height = 15
    target_rect = fitz.Rect(start_x, start_y, start_x + width, start_y + height)

    for page in doc:
        # --- 1. 寻找“干净”的纹理源 (Texture Source) ---
        # 我们取水印正上方 20 像素位置的一块同等大小的区域
        # 这里的偏移量 (-20) 是关键，它能保证取到的是这一页最真实的“空白背景”
        # src_rect = fitz.Rect(start_x, start_y - 20, start_x + width, start_y - 20 + height)
        
        # 如果上方可能也有文字，可以尝试取左侧：
        src_rect = fitz.Rect(start_x - width - 5, start_y, start_x - 5, start_y + height)

        # --- 2. 采样并克隆纹理 ---
        # 将源区域渲染为高分辨率像素图 (dpi=300 确保纹理细腻)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=src_rect)
        
        # 将这块“真实的背景图”覆盖到水印位置
        # 它会完美继承原文件的底色、噪点和任何细微纹理
        page.insert_image(target_rect, pixmap=pix)

        # --- 3. 写入新文字 (保持居中) ---
        my_text = "靖宇"
        f_size = 15.0
        f_name = "china-s"
        
        # 水平居中计算
        text_w = fitz.get_text_length(my_text, fontname=f_name, fontsize=f_size)
        text_x = start_x + (width - text_w) / 2
        
        # 垂直基线微调
        text_y = start_y + 13 
        
        page.insert_text(
            (text_x, text_y),
            my_text,
            fontsize=f_size,
            fontname=f_name,
            color=(0, 0, 0)
        )

    doc.save(output_path)
    print(f"纹理克隆完成！已消除生硬感。输出：{output_path}")

# 执行
clone_texture_replace("Family_Pride_Algorithm.pdf", "final_output_replace.pdf", 1268, 745)