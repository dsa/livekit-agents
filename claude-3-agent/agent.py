# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import time
import threading
import asyncio
from datetime import datetime
from enum import Enum
import json
import logging
from typing import AsyncIterable

from livekit import rtc, agents
from livekit.agents.tts import SynthesisEvent, SynthesisEventType

from claude import ClaudeMessage, ClaudeMessageRole, ClaudePlugin, ClaudeModels
from deepgram import STTStream
from livekit.plugins.elevenlabs import TTS

from dotenv import load_dotenv

load_dotenv()


PROMPT = "You are KITT, a friendly voice assistant powered by LiveKit.  \
          Conversation should be personable, and be sure to ask follow up questions. \
          If your response is a question, please append a question mark symbol to the end of it.\
          Don't respond with more than a few sentences."
INTRO = "Hello, I am KITT, a friendly voice assistant powered by LiveKit Agents. \
        You can find my source code in the top right of this screen if you're curious how I work. \
        Feel free to ask me anything — I'm here to help! Just start talking or type in the chat."
SIP_INTRO = "Hello, I am KITT, a friendly voice assistant powered by LiveKit Agents. \
             Feel free to ask me anything — I'm here to help! Just start talking."


# convert intro response to a stream
async def intro_text_stream(sip: bool):
    if sip:
        yield SIP_INTRO
        return

    yield INTRO


AgentState = Enum("AgentState", "IDLE, LISTENING, THINKING, SPEAKING")

ELEVEN_TTS_SAMPLE_RATE = 24000
ELEVEN_TTS_CHANNELS = 1


class WorkerLifecycle:
    def __init__(self):
        self._accepting_jobs = True
        # Stop accepting jobs after a random time between 20 and 30 minutes
        self._stop_thread = threading.Thread(
            target=self._stop_accepting_jobs_after,
            args=(random.randrange(60 * 60, 60 * 120),),
        )
        self._stop_thread.start()

    def _stop_accepting_jobs_after(self, after: int):
        time.sleep(after)
        self._accepting_jobs = False
        self._kill_after(
            random.randrange(2 * 60, 4 * 60)
        )  # kill 10-15 minutes after stopping accepting jobs

    def _kill_after(self, after: int):
        time.sleep(after)
        self._kill()

    def should_accept_job(self):
        return self._accepting_jobs

    def _kill(self):
        # kill the worker
        os._exit(0)


class KITT:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        kitt = KITT(ctx)
        await kitt.start()

    def __init__(self, ctx: agents.JobContext):
        # plugins
        self.claude_plugin = ClaudePlugin(
            prompt=PROMPT, message_capacity=20, model=ClaudeModels.Claude3Opus.value
        )
        self.tts_plugin = TTS(
            model_id="eleven_turbo_v2", sample_rate=ELEVEN_TTS_SAMPLE_RATE
        )

        self.ctx: agents.JobContext = ctx
        self.chat = rtc.ChatManager(ctx.room)
        self.audio_out = rtc.AudioSource(ELEVEN_TTS_SAMPLE_RATE, ELEVEN_TTS_CHANNELS)

        self._sending_audio = False
        self._processing = False
        self._agent_state: AgentState = AgentState.IDLE

        self.chat.on("message_received", self.on_chat_received)
        self.ctx.room.on("track_subscribed", self.on_track_subscribed)

    async def start(self):
        # if you have to perform teardown cleanup, you can listen to the disconnected event
        # self.ctx.room.on("disconnected", your_cleanup_function)

        # publish audio track
        track = rtc.LocalAudioTrack.create_audio_track("agent-mic", self.audio_out)
        await self.ctx.room.local_participant.publish_track(track)

        # allow the participant to fully subscribe to the agent's audio track, so it doesn't miss
        # anything in the beginning
        await asyncio.sleep(1)

        sip = self.ctx.room.name.startswith("sip")
        await self.process_claude_result(intro_text_stream(sip))
        self.update_state()

    def on_chat_received(self, message: rtc.ChatMessage):
        # TODO: handle deleted and updated messages in message context
        if message.deleted:
            return

        msg = ClaudeMessage(role=ClaudeMessageRole.user, content=message.message)
        claude_result = self.claude_plugin.add_message(msg)
        self.ctx.create_task(self.process_chatgpt_result(claude_result))

    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        self.ctx.create_task(self.process_track(track))

    async def process_track(self, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        stream = STTStream()
        self.ctx.create_task(self.process_stt_stream(stream))
        async for audio_frame_event in audio_stream:
            if self._agent_state != AgentState.LISTENING:
                continue
            stream.push_frame(audio_frame_event.frame)
        await stream.flush()

    async def process_stt_stream(self, stream: STTStream):
        buffered_text = ""
        async for event in stream:
            if event.type == "started":
                pass
            elif event.type == "interim":
                pass
            elif event.type == "finished":
                if event.text == "":
                    continue
                buffered_text = " ".join([buffered_text, event.text])
                await self.ctx.room.local_participant.publish_data(
                    json.dumps(
                        {
                            "text": buffered_text,
                            "timestamp": int(datetime.now().timestamp() * 1000),
                        }
                    ),
                    topic="transcription",
                )

                msg = ClaudeMessage(role=ClaudeMessageRole.user, content=buffered_text)
                claude_stream = self.claude_plugin.add_message(msg)
                self.ctx.create_task(self.process_claude_result(claude_stream))
                buffered_text = ""

    async def process_claude_result(self, text_stream):
        self.update_state(processing=True)

        stream = self.tts_plugin.stream()
        # send audio to TTS in parallel
        self.ctx.create_task(self.send_audio_stream(stream))
        all_text = ""
        async for text in text_stream:
            stream.push_text(text)
            all_text += text

        self.update_state(processing=False)
        # buffer up the entire response from Groq before sending a chat message
        await self.chat.send_message(all_text)
        await stream.flush()

    async def send_audio_stream(self, tts_stream: AsyncIterable[SynthesisEvent]):
        async for e in tts_stream:
            if e.type == SynthesisEventType.STARTED:
                self.update_state(sending_audio=True)
            elif e.type == SynthesisEventType.FINISHED:
                self.update_state(sending_audio=False)
            elif e.type == SynthesisEventType.AUDIO:
                await self.audio_out.capture_frame(e.audio.data)
        await tts_stream.aclose()

    def update_state(self, sending_audio: bool = None, processing: bool = None):
        if sending_audio is not None:
            self._sending_audio = sending_audio
        if processing is not None:
            self._processing = processing

        state = AgentState.LISTENING
        if self._sending_audio:
            state = AgentState.SPEAKING
        elif self._processing:
            state = AgentState.THINKING

        self._agent_state = state
        metadata = json.dumps(
            {
                "agent_state": state.name.lower(),
            }
        )
        self.ctx.create_task(self.ctx.room.local_participant.update_metadata(metadata))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    worker_lifecycle = WorkerLifecycle()

    async def job_request_cb(job_request: agents.JobRequest):
        logging.info("Accepting job for KITT")
        if not worker_lifecycle.should_accept_job():
            await job_request.reject()
            return

        await job_request.accept(
            KITT.create,
            identity="claude_agent",
            name="Claude",
            auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY,
            auto_disconnect=agents.AutoDisconnect.DEFAULT,
        )

    worker = agents.Worker(request_handler=job_request_cb)
    agents.run_app(worker)
