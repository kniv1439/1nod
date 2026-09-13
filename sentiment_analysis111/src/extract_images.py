import os
import fitz  

def extract_images_from_pdf(pdf_path, output_folder):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ 文件不存在: {pdf_path}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder) 

    doc = fitz.open(pdf_path)  
    image_count = 0

    for page_num, page in enumerate(doc, start=1):
        images = page.get_images(full=True)  
        for img_index, img in enumerate(images):
            xref = img[0]  
            base_image = doc.extract_image(xref)  
            image_bytes = base_image["image"]  
            image_ext = base_image["ext"]  

            image_filename = os.path.join(output_folder, f"page{page_num}_img{img_index}.{image_ext}")
            
            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)

            print(f"✅ 提取图片: {image_filename}")
            image_count += 1
    if image_count == 0:
        print("⚠️ 未在 PDF 中找到图片")
    else:
        print(f"🎉 共提取 {image_count} 张图片，保存到 {output_folder}")