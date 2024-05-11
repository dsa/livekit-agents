import asyncio
from livekit import rtc
from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
)
from chatgpt import (
    ChatGPTMessage,
    ChatGPTMessageRole,
    ChatGPTPlugin,
)

from dotenv import load_dotenv

load_dotenv()

CHATGPT_MODEL = "gpt-4-1106-preview"


async def entrypoint(job: JobContext):
    tasks = []
    chatgpt_plugin = ChatGPTPlugin(prompt="", message_capacity=20, model=CHATGPT_MODEL)

    async def process_chatgpt_result(text_stream):
        first_message = True
        message: rtc.ChatMessage = None
        async for text in text_stream:
            if first_message:
                message = await chat.send_message(text)
                first_message = False
            else:
                message.message = message.message + text
                await chat.update_message(message)

    def on_chat_received(message: rtc.ChatMessage):
        if message.deleted:
            return
        msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=message.message)
        chatgpt_result = chatgpt_plugin.add_message(msg)
        tasks.append(asyncio.create_task(process_chatgpt_result(chatgpt_result)))

    chat = rtc.ChatManager(job.room)
    chat.on("message_received", on_chat_received)


async def request_fnc(req: JobRequest) -> None:
    await req.accept(entrypoint, auto_subscribe=None)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc=request_fnc))
