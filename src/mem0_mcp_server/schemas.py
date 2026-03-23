"""Shared Pydantic models for the Mem0 MCP server (local mode)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ToolMessage(BaseModel):
    role: str = Field(..., description="Role of the speaker, e.g., user or assistant.")
    content: str = Field(..., description="Full text of the utterance to store.")


class AddMemoryArgs(BaseModel):
    text: Optional[str] = Field(
        None, description="Simple sentence to remember; converted into a user message when set."
    )
    messages: Optional[list[ToolMessage]] = Field(
        None,
        description=(
            "Explicit role/content history for durable storage. Provide this OR `text`."
        ),
    )
    user_id: Optional[str] = Field(None, description="Override for the user ID.")
    agent_id: Optional[str] = Field(None, description="Optional agent identifier.")
    run_id: Optional[str] = Field(None, description="Optional run identifier.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Opaque metadata to persist.")


class SearchMemoriesArgs(BaseModel):
    query: str = Field(..., description="Describe what you want to find.")
    user_id: Optional[str] = Field(None, description="User scope for the search.")
    agent_id: Optional[str] = Field(None, description="Agent scope for the search.")
    run_id: Optional[str] = Field(None, description="Run scope for the search.")
    limit: Optional[int] = Field(None, description="Optional maximum number of matches.")


class GetMemoriesArgs(BaseModel):
    user_id: Optional[str] = Field(None, description="User scope for listing.")
    agent_id: Optional[str] = Field(None, description="Agent scope for listing.")
    run_id: Optional[str] = Field(None, description="Run scope for listing.")
    limit: Optional[int] = Field(None, description="Maximum number of memories to return.")


class DeleteAllArgs(BaseModel):
    user_id: Optional[str] = Field(
        None, description="User scope to delete; defaults to server user."
    )
    agent_id: Optional[str] = Field(None, description="Optional agent scope filter.")
    run_id: Optional[str] = Field(None, description="Optional run scope filter.")
