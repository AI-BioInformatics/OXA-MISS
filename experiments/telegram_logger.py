# conda install conda-forge::python-telegram-bot
import asyncio, re
from telegram import Bot

# import token e CHANNEL_ID da file .env
from dotenv import load_dotenv
import os
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", None)
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", None)

async def send_telegram_message(message:str):
    if TOKEN is None or CHANNEL_ID is None:
        print("Telegram token or channel ID not set in environment variables.")
        return
    bot = Bot(token=TOKEN)
    message = re.sub(r"\.(?=\d)", "\\.", message)
    message = message.replace("-", "\-")
    message = message.replace(">", "\>")
    message = message.replace("_", "\_")
    print(message)
    msg = await bot.send_message(chat_id=CHANNEL_ID, text=message, parse_mode='MarkdownV2')
    print("Chat ID:", msg.chat_id)

if __name__ == "__main__":
    asyncio.run(send_telegram_message("*Test message*"))
