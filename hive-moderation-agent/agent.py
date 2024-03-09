import aiohttp
import json
import logging
import time
from livekit import agents, rtc
from PIL import Image
import os
from io import BytesIO
from hive_data_classes import HiveResponse, from_dict


from dotenv import load_dotenv

load_dotenv()

hive_headers = {
    "Authorization": f"Token {os.getenv('HIVE_API_KEY')}",
    "accept": "application/json",
}


class ModeratorAgent:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        agent = ModeratorAgent(ctx)
        await agent.start()

    def __init__(self, ctx: agents.JobContext):
        self.ctx = ctx
        self.chat = rtc.ChatManager(ctx.room)

    async def start(self):
        self.ctx.create_task(
            self.chat.send_message(
                "I'm a moderation agent, I will detect and notify you of all inappropriate material you transmit in your video stream"
            )
        )

        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            self.ctx.create_task(self.process_track(track))

        self.ctx.room.on("track_subscribed", on_track_subscribed)

        self.update_agent_state("monitoring")

    async def process_track(self, track: rtc.VideoTrack):
        video_stream = rtc.VideoStream(track)
        last_processed_time = 0
        frame_interval = 5.0  # 1 frame every 5 seconds
        async for frame in video_stream:
            current_time = time.time()
            if (current_time - last_processed_time) >= frame_interval:
                last_processed_time = current_time
                self.ctx.create_task(self.detect(frame))

    async def detect(self, frame: rtc.VideoFrame):
        argb_frame = frame.frame.convert(rtc.VideoBufferType.RGBA)
        image = Image.frombytes(
            "RGBA", (argb_frame.width, argb_frame.height), argb_frame.data
        )
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)  # reset buffer position to beginning after writing

        data = aiohttp.FormData()
        data.add_field("image", buffer, filename="image.png", content_type="image/png")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.thehive.ai/api/v2/task/sync",
                headers=hive_headers,
                data=data,
            ) as response:
                response_dict = await response.json()
                hive_response: HiveResponse = from_dict(HiveResponse, response_dict)
                if (
                    hive_response.code == 200
                    and len(hive_response.status) > 0
                    and len(hive_response.status[0].response.output) > 0
                ):
                    results = hive_response.status[0].response.output[0].classes
                    if len(results) > 0:
                        sorted_results = sorted(
                            results, key=lambda r: r.score, reverse=True
                        )[:10]
                        results_str = "Results:\n"
                        for result in sorted_results:
                            results_str += f"{result.class_}: {result.score}"
                        self.ctx.create_task(self.chat.send_message(results_str))

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
            ModeratorAgent.create,
            identity="hive-moderator",
            name="Moderator",
            # subscribe to all video tracks automatically
            auto_subscribe=agents.AutoSubscribe.VIDEO_ONLY,
            # disconnect when the last participant leaves
            auto_disconnect=agents.AutoDisconnect.DEFAULT,
        )

    worker = agents.Worker(request_handler=job_request_cb)
    agents.run_app(worker)
