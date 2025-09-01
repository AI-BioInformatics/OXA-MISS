# conda install conda-forge::python-telegram-bot
import asyncio, re
from telegram import Bot

TOKEN = "7514286919:AAFb5AhURo10uYIrXx5cOfF0R-unSNaKlPY"
CHANNEL_ID = "-1002519311384"

async def send_telegram_message(message:str):
    bot = Bot(token=TOKEN)
    message = re.sub(r"\.(?=\d)", "\\.", message)
    message = message.replace("-", "\-")
    message = message.replace(">", "\>")
    message = message.replace("_", "\_")
    print(message)
    msg = await bot.send_message(chat_id=CHANNEL_ID, text=message, parse_mode='MarkdownV2')
    print("Chat ID:", msg.chat_id)

if __name__ == "__main__":
    asyncio.run(send_telegram_message("*Prova 0.1*"))
