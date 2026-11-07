import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not configured.")

client = OpenAI(api_key=api_key)


def analyze_article(text: str) -> dict:
    prompt = f"""
請分析以下文章，並以標準 JSON 格式回傳：
1. "title": 擬定一個標題
2. "summary_points": 3個核心重點（陣列格式）
3. "sentiment": 情感（"正向", "中立", 或 "負向"）

文章內容：
{text}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一個資料分析師，只輸出標準 JSON。"},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("The model returned an empty response.")
    return json.loads(content)


if __name__ == "__main__":
    sample_article = (
        "臺灣開放銀行宣佈導入生成式 AI 審查系統，將原本數週的合規審查時間"
        "縮短至數小時，大幅降低新創合作門檻。"
    )
    print("🤖 正在分析文章...")
    result = analyze_article(sample_article)
    print(json.dumps(result, ensure_ascii=False, indent=2))