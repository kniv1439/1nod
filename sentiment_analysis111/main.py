import os
import glob
from src.extract_text import extract_text_from_pdf
from src.extract_images import extract_images_from_pdf  
from src.cleaning import clean_text
from src.sentiment import load_sentiment_dict, analyze_folder_sentiment
from src.deepseek_api import token_extraction
import pandas as pd

def process_pdf_files(pdf_folder, text_output_folder, images_output_folder):
    for folder in [text_output_folder, images_output_folder]:
        os.makedirs(folder, exist_ok=True)

    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

    if not pdf_files:
        print("❌ 未找到任何 PDF 文件")
        return

    for pdf_file in pdf_files:
        print(f"📄 处理文件：{os.path.basename(pdf_file)}")
        try:
            extract_text_from_pdf(pdf_file, text_output_folder)
        except Exception as e:
            print(f"❌ 提取文本时发生错误（{os.path.basename(pdf_file)}）：{e}")
        try:
            extract_images_from_pdf(pdf_file, images_output_folder)
        except Exception as e:
            print(f"❌ 提取图片时发生错误（{os.path.basename(pdf_file)}）：{e}")
if __name__ == "__main__":
    # pdf_folder = "data"                           # 存放 PDF 文件的文件夹
    # text_output_folder = "output/text"            # 提取的文本输出路径
    # images_output_folder = "output/images"        # 提取的图片输出路径
    # cleaned_text_folder = "output/cleaned"        # 清洗文本存放路径
    # #clean_text(text_output_folder, cleaned_text_folder)

    # dic_path = "dic/dic.txt"                     # 词典路径
    # input_folder = "output/cleaned"               # 清洗后的txt文件夹
    # output_csv = "output/sentiment_results.csv"   # 情感结果保存路径

    # fin_sentiment_dict = load_sentiment_dict(dic_path)
    folder_path = r"output\text"
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')] 
    for file_name in txt_files:
        file_path = os.path.join(folder_path,file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        summary = token_extraction(text)
        output_file_path = os.path.join(r"output\sentiment_score_of_each_year", f"{file_name}_sentiment.txt")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(summary)
        
        print(f"Processed {file_name} and saved sentiment score.")