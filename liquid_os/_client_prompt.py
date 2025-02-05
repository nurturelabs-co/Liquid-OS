from enum import Enum
from typing import List, Optional, Tuple

from openai.types import chat
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel

from ._schema import ClientMessage
from .messages import (
    ArgsDict,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from .tools import RunContext


class ToolInvocationState(str, Enum):
    CALL = "call"
    PARTIAL_CALL = "partial-call"
    RESULT = "result"


class ToolCall(BaseModel):
    toolCallId: str
    toolName: str
    args: dict
    state: Optional[str] = None
    result: Optional[str] = None


async def add_context(ctx: RunContext) -> str:
    """Add context to the conversation."""
    return """You are a helpful AI assistant that can help users with various tasks.
    You can get weather information and ask for user confirmation when needed.
    Always ask for confirmation before accessing user's location."""


def convert_messages(
    messages: List[ClientMessage],
) -> List[chat.ChatCompletionMessageParam]:
    """Convert client messages to a format suitable for the agent."""
    openai_messages = []

    for msg in messages:
        message: chat.ChatCompletionMessageParam = {
            "role": msg.role,
            "content": msg.content,
        }

        if msg.toolInvocations:
            for tool in msg.toolInvocations:
                tool_message: chat.ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "tool_call_id": tool.get("id", ""),
                    "name": tool.get("toolName", ""),
                    "content": str(tool.get("result", "")),
                }
                openai_messages.append(tool_message)

        openai_messages.append(message)

    return openai_messages


def convert_to_openai_messages(
    messages: List[ClientMessage],
) -> List[ChatCompletionMessageParam]:
    """Convert client messages to OpenAI message format."""
    openai_messages = []

    for message in messages:
        msg_dict: chat.ChatCompletionMessageParam = {
            "role": message.role,
            "content": message.content,
        }

        if message.toolInvocations:
            tool_calls: List[chat.ChatCompletionMessageToolCallParam] = []
            for tool in message.toolInvocations:
                if tool.get("state") == "result":
                    tool_calls.append(
                        {
                            "id": tool["toolCallId"],
                            "type": "function",
                            "function": {
                                "name": tool["toolName"],
                                "arguments": tool.get("args", "{}"),
                            },
                        }
                    )
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls

            tool_results = []
            for tool in message.toolInvocations:
                if tool.get("result"):
                    tool_results.append(
                        {
                            "tool_call_id": tool["toolCallId"],
                            "role": "tool",
                            "name": tool["toolName"],
                            "content": str(tool["result"]),
                        }
                    )
            openai_messages.extend(tool_results)

        openai_messages.append(msg_dict)

    return openai_messages


def convert_to_model_messages(
    messages: List[ClientMessage],
) -> Tuple[Optional[str], List[ModelMessage]]:
    """Convert client messages to Liquid OS model messages and extract user prompt if last message is from user."""
    model_messages = []
    user_prompt = None

    for msg in messages:
        if msg.role == "user":
            model_messages.append(
                ModelRequest(parts=[UserPromptPart(content=msg.content)])
            )
            # Store user prompt if this is the last message
            if msg == messages[-1]:
                user_prompt = msg.content
        elif msg.role == "assistant":
            parts = []
            tool_returns = []
            if msg.toolInvocations:
                for tool in msg.toolInvocations:
                    if tool.get("state") == "result":
                        parts.append(
                            ToolCallPart(
                                tool_name=tool["toolName"],
                                args=ArgsDict(args_dict=tool["args"]),
                                tool_call_id=tool["toolCallId"],
                            )
                        )
                        # Store tool returns to add after ModelResponse
                        tool_returns.append(
                            ModelRequest(
                                parts=[
                                    ToolReturnPart(
                                        tool_name=tool["toolName"],
                                        tool_call_id=tool["toolCallId"],
                                        content=str(tool["result"]),
                                    )
                                ]
                            )
                        )

            if not parts and msg.content:
                parts.append(TextPart(content=msg.content))

            if parts:  # Add ModelResponse if there are parts
                model_messages.append(ModelResponse(parts=parts, kind="response"))
                # Add any tool returns after the ModelResponse
                model_messages.extend(tool_returns)

    return user_prompt, model_messages
