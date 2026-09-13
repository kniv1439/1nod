from openai import OpenAI

def token_extraction(text):
    client = OpenAI(
        api_key="sk-cd29c98ca3a54917b3bbe951bf30f3b7",  
        base_url="https://api.deepseek.com"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的文本处理助手，专门提取中文token并分析情绪值"},
                {"role": "user", "content": f"""
                 请严格按以下要求处理文本：
                 1. 输出200个token即可
                 2. 清洗文本，保留最能体现情绪的2字或4字的中文token，为每个token赋予1-10的情绪系数，输出格式为以'begin'开头，'end'结尾，每行格式为'token,k'
                 3. 对重复token只保留第一次出现的
                 待处理文本：
                 {text}
                 """},
            ],
            stream=False
        ) 
        result = response.choices[0].message.content
        return result 
    
    except Exception as e:
        return f"Error: {str(e)}"