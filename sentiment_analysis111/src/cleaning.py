import os
import re
import spacy
from spacy.lang.zh.stop_words import STOP_WORDS

# 加载中文模型
nlp = spacy.load("zh_core_web_sm")

def clean_text(text):
    # 去除文本中的所有数字
    text = re.sub(r'\d+', '', text)

    # 去除非中文字符
    text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)

    # 使用spaCy处理文本
    doc = nlp(text)

    # 去除停用词并返回清洗后的文本
    cleaned_text = " ".join([token.text for token in doc if token.text not in STOP_WORDS])

    # 去除多余的换行符（合并多余的换行符为一个，并去掉开头和结尾的换行）
    cleaned_text = re.sub(r'\n+', ' ', cleaned_text).strip()  # 将多余换行符替换为空格，并去除首尾空白

    return cleaned_text

def process_files_in_folder(input_folder, output_folder):
    """批量处理文件夹中的所有txt文件，清洗内容并保存到新的文件夹"""
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"❌ 文件夹不存在: {input_folder}")

    # 创建输出文件夹（如果不存在的话）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历文件夹中的所有txt文件
    for filename in os.listdir(input_folder):
        # 只处理txt文件
        if filename.endswith(".txt"):
            input_file = os.path.join(input_folder, filename)

            # 读取文件内容
            with open(input_file, 'r', encoding='utf-8') as file:
                text = file.read()

            # 清洗文本
            cleaned_text = clean_text(text)

            # 生成输出文件路径
            output_file = os.path.join(output_folder, filename)

            # 将清洗后的文本写入新文件
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)

            print(f"✅ 清洗后的文本已保存到 {output_file}")