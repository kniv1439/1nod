import os
import fitz  

def extract_text_from_pdf(pdf_path, outdir):
    """提取PDF中的文本"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ 文件不存在: {pdf_path}")
    print("📂 处理文件:", pdf_path)
    doc = fitz.open(pdf_path)

    output_dir = "output/text"
    os.makedirs(output_dir, exist_ok=True)  
    text = "\n".join([page.get_text("text") for page in doc])  
    text_output_file = os.path.join(outdir, f"{os.path.basename(pdf_path).replace('.pdf', '.txt')}")
    with open(text_output_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ 文本已提取并保存到 {text_output_file}")
    return text