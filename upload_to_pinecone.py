import os
import csv
import uuid
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

def normalize_text(text):
    replacements = {
        "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
        "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
        "1": "一", "2": "二", "3": "三", "4": "四", "5": "五",
        "6": "六", "7": "七", "8": "八", "9": "九", "10": "十"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def embed_text(text):
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",  # ⚠️ 請與你 Pinecone index 的 model 一致
        input=[text]
    )
    return embedding.data[0].embedding

def upload_textfile(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                cleaned = normalize_text(line)
                vector = embed_text(cleaned)
                index.upsert([{
                    "id": f"text-{uuid.uuid4()}",
                    "values": vector,
                    "metadata": {"text": cleaned}
                }])
    print("✅ .txt 上傳完成")

def upload_csv(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            text = " ".join(row).strip()
            if text:
                cleaned = normalize_text(text)
                vector = embed_text(cleaned)
                index.upsert([{
                    "id": f"csv-{uuid.uuid4()}",
                    "values": vector,
                    "metadata": {"text": cleaned}
                }])
    print("✅ .csv 上傳完成")

if __name__ == "__main__":
    path = input("請輸入上傳檔案路徑 (.txt 或 .csv)：").strip()
    if path.endswith(".txt"):
        upload_textfile(path)
    elif path.endswith(".csv"):
        upload_csv(path)
    else:
        print("❌ 不支援的檔案格式，請上傳 .txt 或 .csv")
