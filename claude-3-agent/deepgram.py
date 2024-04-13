import asyncio

import json
import logging
import os
from urllib.parse import urlencode
from dataclasses import dataclass
from typing import Optional

import aiohttp
from livekit import rtc

from enum import Enum


# taken from deepgram-sdk
class LiveTranscriptionEvents(str, Enum):
    Open: str = "Open"
    Close: str = "Close"
    Transcript: str = "Results"
    Metadata: str = "Metadata"
    UtteranceEnd: str = "UtteranceEnd"
    SpeechStarted: str = "SpeechStarted"
    Error: str = "Error"
    Warning: str = "Warning"


STREAM_KEEPALIVE_MSG: str = json.dumps({"type": "KeepAlive"})
STREAM_CLOSE_MSG: str = json.dumps({"type": "CloseStream"})


class STTStream:
    @dataclass
    class StartedEvent:
        type: str = "started"

    @dataclass
    class InterimEvent:
        text: str
        type: str = "interim"

    @dataclass
    class FinishedEvent:
        text: str
        type: str = "finished"

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._api_key = os.environ["DEEPGRAM_API_KEY"]

        self._queue = asyncio.Queue()
        self._event_queue = asyncio.Queue[
            STTStream.StartedEvent | STTStream.InterimEvent | STTStream.FinishedEvent
        ]()
        self._closed = False
        self._main_task = asyncio.create_task(self._run(max_retry=32))

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(f"deepgram task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")

        self._queue.put_nowait(frame.remix_and_resample(16000, 1))

    async def aclose(self) -> None:
        await self._queue.put(STREAM_CLOSE_MSG)
        await self._main_task

    async def _run(self, max_retry: int) -> None:
        """Try to connect to Deepgram with exponential backoff and forward frames"""
        async with aiohttp.ClientSession() as session:
            retry_count = 0
            ws: Optional[aiohttp.ClientWebSocketResponse] = None
            listen_task: Optional[asyncio.Task] = None
            keepalive_task: Optional[asyncio.Task] = None
            while True:
                try:
                    ws = await self._try_connect(session)
                    listen_task = asyncio.create_task(self._listen_loop(ws))
                    keepalive_task = asyncio.create_task(self._keepalive_loop(ws))
                    # break out of the retry loop if we are done
                    if await self._send_loop(ws):
                        keepalive_task.cancel()
                        await asyncio.wait_for(listen_task, timeout=5)
                        break
                except Exception as e:
                    if retry_count > max_retry and max_retry > 0:
                        logging.error(f"failed to connect to Deepgram: {e}")
                        break

                    retry_delay = min(retry_count * 5, 5)  # max 5s
                    retry_count += 1
                    logging.warning(
                        f"failed to connect to Deepgram: {e} - retrying in {retry_delay}s"
                    )
                    await asyncio.sleep(retry_delay)

        self._closed = True

    async def _send_loop(self, ws: aiohttp.ClientWebSocketResponse) -> bool:
        while not ws.closed:
            data = await self._queue.get()
            # fire and forget, we don't care if we miss frames in the error case
            self._queue.task_done()

            if ws.closed:
                raise Exception("websocket closed")

            if isinstance(data, rtc.AudioFrame):
                await ws.send_bytes(data.data.tobytes())
            else:
                if data == STREAM_CLOSE_MSG:
                    await ws.send_str(data)
                    return True
        return False

    async def _keepalive_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while not ws.closed:
            await ws.send_str(STREAM_KEEPALIVE_MSG)
            await asyncio.sleep(5)

    async def _listen_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        speaking = False
        last_transcript = ""

        while not ws.closed:
            msg = await ws.receive()
            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                break

            try:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    type = data.get("type")
                    if not type:
                        continue

                    if not speaking:
                        if type == LiveTranscriptionEvents.SpeechStarted:
                            speaking = True
                            event = self.StartedEvent()
                            await self._event_queue.put(event)
                    else:
                        if type == LiveTranscriptionEvents.UtteranceEnd:
                            if last_transcript != "":
                                speaking = False
                                event = self.FinishedEvent(text=last_transcript)
                                last_transcript = ""
                                await self._event_queue.put(event)
                        elif type == LiveTranscriptionEvents.Transcript:
                            is_final_transcript = data["is_final"]
                            is_endpoint = data["speech_final"]

                            if is_final_transcript:
                                transcript = data["channel"]["alternatives"][0][
                                    "transcript"
                                ]
                                if transcript != "":
                                    last_transcript += transcript
                                    if not is_endpoint:
                                        last_transcript += " "

                            if is_endpoint and last_transcript != "":
                                speaking = False
                                event = self.FinishedEvent(text=last_transcript)
                                last_transcript = ""
                                await self._event_queue.put(event)

            except Exception as e:
                logging.error("Error handling message %s: %s", msg, e)
                continue

    async def _try_connect(
        self, session: aiohttp.ClientSession
    ) -> aiohttp.ClientWebSocketResponse:
        live_config = {
            "model": "nova-2",
            "language": "en-US",
            "filler_words": True,
            "punctuate": True,
            "smart_format": True,
            "interim_results": True,
            "encoding": "linear16",
            "sample_rate": 16000,
            "channels": 1,
            "endpointing": 200,
            "vad_events": True,
            "utterance_end_ms": 1000,
        }

        query_params = urlencode(live_config).lower()

        url = f"wss://api.deepgram.com/v1/listen?{query_params}"
        ws = await session.ws_connect(
            url, headers={"Authorization": f"Token {self._api_key}"}
        )

        return ws

    def __aiter__(self) -> "STTStream":
        return self

    async def __anext__(
        self,
    ):
        if self._closed and self._event_queue.empty():
            raise StopAsyncIteration

        return await self._event_queue.get()
