import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# Загрузка данных
df = pd.read_csv("products.csv")
texts = df["name"].astype(str).tolist()
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(texts, show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings).astype("float32"))

import os
TOKEN = os.environ.get("TOKEN")

# Обработка сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), k=5)

    results = []
    for idx in I[0]:
        name = df.iloc[idx]["name"]
        price = df.iloc[idx]["price"]
        supplier = df.iloc[idx]["supplier"]
        results.append(f"📦 {name}\n💰 {price} ₽\n🏬 {supplier}")

    reply = "\n\n".join(results)
    await update.message.reply_text(reply if results else "Ничего не найдено.")

# Запуск бота
def run_bot():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Бот запущен. Ждёт запросов...")
    app.run_polling()

if __name__ == "__main__":
    run_bot()