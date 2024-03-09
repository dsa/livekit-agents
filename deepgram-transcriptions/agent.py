import json
import logging
from livekit import agents, rtc
from livekit.plugins.deepgram import STT, SpeechStream

from dotenv import load_dotenv

load_dotenv()


class TranscriptionAgent:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        agent = TranscriptionAgent(ctx)
        await agent.start()

    def __init__(self, ctx: agents.JobContext):
        self.ctx = ctx
        self.chat = rtc.ChatManager(ctx.room)
        self.stt = STT(
            min_silence_duration=100,
        )

    async def start(self):
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            self.ctx.create_task(self.process_track(track, participant))

        self.ctx.room.on("track_subscribed", on_track_subscribed)

        self.update_agent_state("listening")

    async def process_track(
        self, track: rtc.AudioTrack, participant: rtc.RemoteParticipant
    ):
        audio_stream = rtc.AudioStream(track)
        stream = self.stt.stream()
        self.ctx.create_task(self.process_stt(stream, participant))
        async for audio_frame_event in audio_stream:
            stream.push_frame(audio_frame_event.frame)
        await stream.flush()

    async def process_stt(
        self, stream: SpeechStream, participant: rtc.RemoteParticipant
    ):
        buffered_text = ""
        async for event in stream:
            if event.alternatives[0].text == "":
                continue
            if event.is_final:
                buffered_text = " ".join([buffered_text, event.alternatives[0].text])

            if not event.end_of_speech:
                continue

            self.ctx.create_task(
                self.chat.send_message(f"{participant.identity} said: {buffered_text}")
            )
            buffered_text = ""

    def update_agent_state(self, state: str):
        metadata = json.dumps(
            {
                "agent_state": state,
            }
        )
        self.ctx.create_task(self.ctx.room.local_participant.update_metadata(metadata))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def job_request_cb(job_request: agents.JobRequest):
        await job_request.accept(
            TranscriptionAgent.create,
            identity="deepgram-transcriber",
            name="Transcriber",
            # subscribe to all video tracks automatically
            auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY,
            # disconnect when the last participant leaves
            auto_disconnect=agents.AutoDisconnect.DEFAULT,
        )

    worker = agents.Worker(request_handler=job_request_cb)
    agents.run_app(worker)
