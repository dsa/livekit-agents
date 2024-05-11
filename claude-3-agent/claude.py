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
import logging
import asyncio
from anthropic import AsyncAnthropic
from dataclasses import dataclass
from typing import AsyncIterable, List, Optional
from enum import Enum

ClaudeMessageRole = Enum("MessageRole", ["system", "user", "assistant", "function"])


class ClaudeModels(Enum):
    Claude3Opus = "claude-3-opus-20240229"
    Claude3Sonnet = "claude-3-sonnet-20240229"
    Claude3Haiku = "claude-3-haiku-20240307"


@dataclass
class ClaudeMessage:
    role: ClaudeMessageRole
    content: str

    def to_api(self):
        return {"role": self.role.name, "content": self.content}


class ClaudePlugin:
    """Claude Plugin"""

    def __init__(self, prompt: str, message_capacity: int, model: str):
        """
        Args:
            prompt (str): First 'system' message sent to the chat that prompts the assistant
            message_capacity (int): Maximum number of messages to send to the chat
            model (str): Which model to use (i.e. 'gpt-3.5-turbo')
        """
        self._model = model
        self._client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._prompt = prompt
        self._message_capacity = message_capacity
        self._messages: List[ClaudeMessage] = []
        self._producing_response = False
        self._needs_interrupt = False

    def interrupt(self):
        """Interrupt a currently streaming response (if there is one)"""
        if self._producing_response:
            self._needs_interrupt = True

    async def aclose(self):
        pass

    async def send_system_prompt(self) -> AsyncIterable[str]:
        """Send the system prompt to the chat and generate a streamed response

        Returns:
            AsyncIterable[str]: Streamed ChatGPT response
        """
        async for text in self.add_message(None):
            yield text

    async def add_message(self, message: Optional[ClaudeMessage]) -> AsyncIterable[str]:
        """Add a message to the chat and generate a streamed response

        Args:
            message (ChatGPTMessage): The message to add

        Returns:
            AsyncIterable[str]: Streamed ChatGPT response
        """

        if message is not None:
            self._messages.append(message)
        if len(self._messages) > self._message_capacity:
            self._messages.pop(0)

        async for text in self._generate_text_streamed(self._model):
            yield text

    async def _generate_text_streamed(self, model: str) -> AsyncIterable[str]:
        prompt_message = ClaudeMessage(
            role=ClaudeMessageRole.system, content=self._prompt
        )
        try:
            chat_messages = [m.to_api() for m in self._messages]
            chat_stream = await self._client.messages.create(
                model=model,
                stream=True,
                max_tokens=1024,
                messages=[prompt_message.to_api()] + chat_messages,
            )
        except TimeoutError:
            yield "Sorry, I'm taking too long to respond. Please try again later."
            return

        self._producing_response = True
        complete_response = ""

        async def anext_util(aiter):
            async for item in aiter:
                return item

            return None

        while True:
            try:
                chunk = await asyncio.wait_for(anext_util(chat_stream), 5)
            except TimeoutError:
                break
            except asyncio.CancelledError:
                self._producing_response = False
                self._needs_interrupt = False
                break

            if chunk is None:
                break
            content = chunk.choices[0].delta.content

            if self._needs_interrupt:
                self._needs_interrupt = False
                logging.info("ChatGPT interrupted")
                break

            if content is not None:
                complete_response += content
                yield content

        self._messages.append(
            ClaudeMessage(role=ClaudeMessageRole.assistant, content=complete_response)
        )
        self._producing_response = False
