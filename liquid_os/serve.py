import json
import logging
from typing import AsyncIterator, List

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ._client_prompt import convert_to_model_messages
from ._schema import ClientMessage, ToolCallRequest
from .agent import Agent
from .messages import ArgsDict

# load_dotenv(".env.local")
# logfire.configure()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RequestSchema(BaseModel):
    messages: List[ClientMessage]


async def stream_response(
    agent: Agent, messages: List[ClientMessage]
) -> AsyncIterator[str]:
    """Stream the chat response."""
    logger.info(f"Received {len(messages)} messages")
    if not messages:
        logger.info("No messages received, returning early")
        return

    logger.info("Converting messages to model format...")
    user_prompt, message_history = convert_to_model_messages(messages)

    print("Running agent with user_prompt:")
    print(user_prompt)
    print("message_history:")
    print(message_history)

    # Use run instead of run_sync for async context
    result = await agent.run(
        user_prompt=user_prompt, message_history=message_history, infer_name=False
    )

    print(f"Received response: {result.data}, type: {type(result.data)}")

    # Handle ToolCallRequest
    if isinstance(result.data, ToolCallRequest):
        print(f"Tool call request: {result.data.toolName}")
        yield '9:{{"toolCallId":"{id}","toolName":"{name}","args":{args}}}\n'.format(
            id=result.data.toolCallId,
            name=result.data.toolName,
            args=json.dumps(result.data.args.args_dict)
            if isinstance(result.data.args, ArgsDict)
            else result.data.args.args_json,
        )
    # Handle string responses
    elif isinstance(result.data, str):
        logger.info(f"Text response: {result.data[:50]}...")
        yield "0:{text}\n".format(text=json.dumps(result.data))
    else:
        print(f"Unknown response type: {type(result.data)}")

    # Send final usage stats
    logger.info("Sending final usage stats...")
    yield 'e:{{"finishReason":"stop","usage":{{"promptTokens":{prompt},"completionTokens":{completion}}},"isContinued":false}}\n'.format(
        prompt=100, completion=100
    )


class SuggestedAction(BaseModel):
    title: str
    label: str
    action: str


def serve_agent(agent: Agent, suggestions: List[SuggestedAction] = []):
    app = FastAPI()

    @app.post("/api/chat")
    async def handle_chat_data(request: RequestSchema, protocol: str = Query("data")):
        logger.info(f"Received chat request with protocol: {protocol}")
        response = StreamingResponse(stream_response(agent, request.messages))
        response.headers["x-vercel-ai-data-stream"] = "v1"
        logger.info("Returning streaming response")
        return response

    @app.get("/api/suggestions", response_model=List[SuggestedAction])
    async def get_suggestions():
        return suggestions

    return app
