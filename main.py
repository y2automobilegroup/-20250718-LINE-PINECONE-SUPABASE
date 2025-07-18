import os
import uuid
from flask import Flask, request
from linebot.v3.messaging import MessagingApi, Configuration, ApiClient, ReplyMessageRequest, TextMessage
from linebot.v3.webhook import WebhookParser
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone

# 記憶對話 & 人工客服模式
from collections import defaultdict, deque
user_memory = defaultdict(lambda: deque(maxlen=10))
manual_mode = set()  # 存放進入人工客服模式的 user_id

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))
configuration = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
parser = WebhookParser(os.getenv("LINE_CHANNEL_SECRET"))
app = Flask(__name__)

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

                # ✅ 對話記憶（不論是否人工）
                user_memory[user_id].append({"role": "user", "content": query})

                # ✅ 切換人工客服啟用
                if query == "人工客服您好":
                    manual_mode.add(user_id)
                    print(f"[人工模式 ON] {user_id}")
                    return "OK", 200

                # ✅ 結束人工客服
                if query == "人工客服結束":
                    manual_mode.discard(user_id)
                    print(f"[人工模式 OFF] {user_id}")
                    return "OK", 200

                # ✅ 如果是人工客服模式，不回覆但記憶
                if user_id in manual_mode:
                    print(f"[靜默中] {user_id} 為人工客服中，跳過 GPT 回覆")
                    return "OK", 200

                # ✅ 查詢 Pinecone
                vector = embed_text(query)
                res = index.query(vector=vector, top_k=5, include_metadata=True)
                matches = [m for m in res["matches"] if m["score"] >= 0.2]

                if not matches:
                    fallback = "亞鈺智能客服您好：感謝您的詢問，目前您的問題需要專人回覆您，請稍後馬上有人為您服務！😄"
                    user_memory[user_id].append({"role": "assistant", "content": fallback})
                    line_bot_api.reply_message(ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=fallback)]
                    ))
                    return "OK", 200

                context = "\n".join([m["metadata"]["text"] for m in matches])

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
                    "content": f"參考資料：{context}\n\n問題：{query}"
                }
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

@app.route("/")
def home():
    return "LINE GPT Bot Ready"

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

def embed_text(text):
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return embedding.data[0].embedding

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
