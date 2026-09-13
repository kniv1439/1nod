import os
import csv

# 加载情感词典
def load_sentiment_dict(dic_path):
    sentiment_dict = {}
    with open(dic_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            parts = line.strip().split()  # 按空格分割
            if len(parts) == 2:          # 确保每行有两个部分
                word, score = parts
                sentiment_dict[word] = int(score)  # 将得分转为整数
    return sentiment_dict

# 情感分析函数
def analyze_sentiment(text, sentiment_dict):
    words = text.split()  # 假设已分词，每个词以空格分隔
    sentiment_score = sum(sentiment_dict.get(word, 0) for word in words)

    if sentiment_score > 0:
        sentiment = "正面"
    elif sentiment_score < 0:
        sentiment = "负面"
    else:
        sentiment = "中立"

    return sentiment_score, sentiment

# 处理文件夹中的所有txt文件并保存结果到csv
def analyze_folder_sentiment(input_folder, sentiment_dict, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['文件名', '情感得分', '情感倾向']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filename in os.listdir(input_folder):
            if filename.endswith(".txt"):
                input_file = os.path.join(input_folder, filename)
                with open(input_file, 'r', encoding='utf-8') as file:
                    text = file.read()

                score, sentiment = analyze_sentiment(text, sentiment_dict)
                writer.writerow({'文件名': filename, '情感得分': score, '情感倾向': sentiment})

                print(f"✅ 文件 {filename} 的情感分析已保存到 CSV 文件")