import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('output/sentiment_results.csv', encoding='utf-8-sig')

files = data['文件名']  
positive_scores = data['情感得分']  

plt.figure(figsize=(10, 6))
plt.plot(files, positive_scores, marker='o', linestyle='-', color='b', label='scores')

plt.title('Changes of sentimental value')
plt.xlabel('files')
plt.ylabel('sentiment scores')

step = 4  # 每四个显示一个标签
xticks_positions = range(0, len(files), step)  # 每四个文件选择一个位置
xticks_labels = [files[i][:4] for i in xticks_positions]  # 获取文件名的前四个字符


plt.xticks(xticks_positions, xticks_labels, rotation=45)  

plt.legend()  
plt.tight_layout() 
plt.show()  
