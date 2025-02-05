from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .messages import ArgsDict, ArgsJson


class ClientMessage(BaseModel):
    id: Optional[str] = None
    role: str
    content: str
    toolInvocations: Optional[List[Dict[str, Any]]] = None
    experimental_attachments: Optional[List[Dict[str, Any]]] = None


class ToolCallRequest(BaseModel):
    """Used to indicate a tool call that should be handled by the frontend."""

    toolCallId: str
    toolName: str
    args: ArgsJson | ArgsDict
    state: Optional[str] = None
    result: Optional[str] = None
