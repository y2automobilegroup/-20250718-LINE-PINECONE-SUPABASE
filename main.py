import os
import uuid
from flask import Flask, request
from linebot.v3.messaging import MessagingApi, Configuration, ApiClient, ReplyMessageRequest, TextMessage
from linebot.v3.webhook import WebhookParser
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
from supabase import create_client, Client
from collections import defaultdict, deque

# ✅ 載入 .env 環境變數
load_dotenv()

# ✅ 初始化 OpenAI、Pinecone、Supabase、LINE
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))
configuration = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
parser = WebhookParser(os.getenv("LINE_CHANNEL_SECRET"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ Flask 應用與對話記憶
app = Flask(__name__)
user_memory = defaultdict(lambda: deque(maxlen=10))
manual_mode = set()

# ✅ 查詢 Supabase cars 表
def query_supabase_cars(query: str):
    try:
        print(f"[查詢 Supabase] 使用關鍵字：{query}")
        response = supabase.table("cars") \
            .select("物件編號, 廠牌, 車款, 車型, 年式, 車輛售價, 車輛賣點, 車輛副標題, 賣家保證, 特色說明") \
            .ilike("特色說明", f"%{query}%") \
            .limit(3).execute()

        print(f"[Supabase 結果] {response.data}")
        if response.data:
            blocks = []
            for row in response.data:
                text = f"""【{row.get("廠牌", "")} {row.get("車款", "")}】
副標：{row.get("車輛副標題", "")}
賣點：{row.get("車輛賣點", "")}
特色說明：{row.get("特色說明", "")}
保固內容：{row.get("賣家保證", "")}
價格：{row.get("車輛售價", "")} 元"""
                blocks.append(text)
            return "\n\n".join(blocks)
        return None
    except Exception as e:
        print(f"[Supabase 查詢錯誤] {e}")
        return None

# ✅ 向量化文字
def embed_text(text):
    try:
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return embedding.data[0].embedding
    except Exception as e:
        print(f"[Embedding 錯誤] {e}")
        return []

# ✅ 處理 LINE Webhook
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("x-line-signature")
    body = request.get_data(as_text=True)

    try:
        events = parser.parse(body, signature)
    except:
        return "Invalid signature", 400

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        for event in events:
            if isinstance(event, MessageEvent) and isinstance(event.message, TextMessageContent):
                user_id = event.source.user_id
                query = event.message.text.strip()

                # 記憶對話
                user_memory[user_id].append({"role": "user", "content": query})

                # ✅ 人工客服開啟
                if query == "人工客服您好":
                    manual_mode.add(user_id)
                    print(f"[人工模式 ON] {user_id}")
                    line_bot_api.reply_message(ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text="✅ 已為您切換至人工客服模式，請稍候專人回覆。")]
                    ))
                    return "OK", 200

                # ✅ 人工客服結束
                if query == "人工客服結束":
                    manual_mode.discard(user_id)
                    print(f"[人工模式 OFF] {user_id}")
                    line_bot_api.reply_message(ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text="✅ 已離開人工客服模式，亞鈺智能客服為您繼續服務！😄")]
                    ))
                    return "OK", 200

                if user_id in manual_mode:
                    print(f"[靜默中] {user_id} 為人工客服中，跳過 GPT 回覆")
                    return "OK", 200

                # ✅ 查詢 Pinecone
                vector = embed_text(query)
                res = index.query(vector=vector, top_k=5, include_metadata=True)
                matches = res.get("matches", [])
                matches = [m for m in matches if m.get("score", 0) >= 0.2]

                memory_messages = list(user_memory[user_id])
                memory_messages.append({"role": "user", "content": query})

                system_prompt = {
                    "role": "system",
                    "content": (
                        "你是亞鈺汽車的50年資深客服專員，擅長解決問題且擅長思考拆解問題，"
                        "請先透過參考資料判斷並解析問題點，只詢問參考資料需要的問題，"
                        "不要問不相關參考資料的問題，如果詢問內容不在參考資料內，請先判斷這句話是什麼類型的問題，"
                        "然後針對參考資料內的資料做反問問題，最後問到需要的答案，請用最積極與充滿溫度的方式回答，"
                        "若參考資料與問題無關，比如他是來聊天的，請回覆罐頭訊息：\"感謝您的詢問，請詢問亞鈺汽車相關問題，我們很高興為您服務！😄\""
                    )
                }

                user_prompt = {
                    "role": "user",
                    "content": ""
                }

                # ✅ 如果有 Pinecone 匹配
                if matches:
                    context = "\n".join([m["metadata"]["text"] for m in matches])
                    user_prompt["content"] = f"參考資料：{context}\n\n問題：{query}"
                else:
                    # ✅ 查詢 Supabase 資料表
                    supabase_context = query_supabase_cars(query)
                    if supabase_context:
                        system_prompt["content"] = (
                            "你是亞鈺汽車的50年資深客服專員，擅長解決問題且擅長思考拆解問題，"
                            "目前你參考的不是Pinecone的資料，而是Supabase中 `cars` 資料表的描述欄位，"
                            "請嘗試從中理解客戶問題，並以溫暖、有耐心的語氣協助回答。"
                        )
                        user_prompt["content"] = f"資料來源：{supabase_context}\n\n問題：{query}"
                    else:
                        fallback = "亞鈺智能客服您好：感謝您的詢問，目前您的問題需要專人回覆您，請稍後馬上有人為您服務！😄"
                        user_memory[user_id].append({"role": "assistant", "content": fallback})
                        line_bot_api.reply_message(ReplyMessageRequest(
                            reply_token=event.reply_token,
                            messages=[TextMessage(text=fallback)]
                        ))
                        return "OK", 200

                # ✅ 呼叫 GPT 回覆
                chat_completion = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[system_prompt] + memory_messages + [user_prompt]
                )
                answer = chat_completion.choices[0].message.content.strip()
                if not answer.startswith("亞鈺智能客服您好："):
                    answer = "亞鈺智能客服您好：" + answer

                user_memory[user_id].append({"role": "assistant", "content": answer})
                line_bot_api.reply_message(ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=answer)]
                ))

    return "OK", 200

# ✅ 根目錄測試
@app.route("/")
def home():
    return "LINE GPT Bot Ready"

# ✅ 上傳文字到 Pinecone
@app.route("/upload", methods=["POST"])
def upload_text():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return {"error": "Missing text"}, 400

    embedding = embed_text(text)
    id = "web-" + str(uuid.uuid4())[:8]

    index.upsert([
        {
            "id": id,
            "values": embedding,
            "metadata": {"text": text}
        }
    ])
    return {"message": "✅ 上傳成功", "id": id}

# ✅ 啟動服務
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
