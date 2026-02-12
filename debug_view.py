import fitz  # PyMuPDF

def debug_every_page(pdf_path, x, y):
    doc = fitz.open(pdf_path)
    print(f"文档共 {len(doc)} 页，开始生成每页的 Debug 视图...")

    # 设定遮盖框的大小（宽 150，高 40），如果太小盖不住，请改大这里的 150 和 40
    rect_width = 100
    rect_height = 15
    
    # 构造矩形：左上角 (x, y) -> 右下角 (x+w, y+h)
    target_rect = fitz.Rect(x, y, x + rect_width, y + rect_height)

    for i, page in enumerate(doc):
        # 画红框
        page.draw_rect(target_rect, color=(1, 0, 0), width=2)
        # 标页码方便检查
        page.insert_text((x, y-5), f"Page {i+1} Check", color=(1,0,0), fontsize=10)

    output_name = "debug_all_pages.pdf"
    doc.save(output_name)
    print(f"Debug 完成！请打开 {output_name} 检查每一页的红框是否都盖住了水印。")

# 运行：传入你测出的坐标 1261, 743
debug_every_page("Family_Pride_Algorithm.pdf", 1268, 745)