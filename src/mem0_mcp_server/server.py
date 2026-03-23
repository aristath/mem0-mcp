"""MCP server that exposes Mem0 local memory as MCP tools."""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated, Any, Dict, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP
from mem0 import Memory
from pydantic import Field

try:
    from .schemas import (
        AddMemoryArgs,
        DeleteAllArgs,
        GetMemoriesArgs,
        SearchMemoriesArgs,
        ToolMessage,
    )
except ImportError:
    from schemas import (
        AddMemoryArgs,
        DeleteAllArgs,
        GetMemoriesArgs,
        SearchMemoriesArgs,
        ToolMessage,
    )

load_dotenv()

os.environ["MEM0_TELEMETRY"] = os.getenv("MEM0_TELEMETRY", "false")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("mem0_mcp_server")

ENV_DEFAULT_USER_ID = os.getenv("MEM0_DEFAULT_USER_ID", "default")

# Data directory
_DATA_DIR = os.getenv("MEM0_DATA_DIR", os.path.expanduser("~/.local/share/mem0"))

# Embedding model dims lookup for common models
_KNOWN_DIMS = {
    "multi-qa-MiniLM-L6-cos-v1": 384,
    "all-MiniLM-L6-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "nomic-ai/nomic-embed-text-v1.5": 768,
}


def _build_config() -> dict:
    """Build mem0 Memory config from environment variables."""
    embedder_model = os.getenv("MEM0_EMBEDDER_MODEL", "multi-qa-MiniLM-L6-cos-v1")
    embedding_dims = int(
        os.getenv("MEM0_EMBEDDING_DIMS", str(_KNOWN_DIMS.get(embedder_model, 384)))
    )

    config: dict[str, Any] = {
        "llm": {
            "provider": os.getenv("MEM0_LLM_PROVIDER", "openai"),
            "config": {
                "model": os.getenv("MEM0_LLM_MODEL", "local-model"),
                "openai_base_url": os.getenv("MEM0_LLM_BASE_URL", "http://localhost:8080/v1"),
                "api_key": os.getenv("MEM0_LLM_API_KEY", "not-needed"),
                "temperature": float(os.getenv("MEM0_LLM_TEMPERATURE", "0.1")),
                "max_tokens": int(os.getenv("MEM0_LLM_MAX_TOKENS", "2000")),
            },
        },
        "embedder": {
            "provider": os.getenv("MEM0_EMBEDDER_PROVIDER", "huggingface"),
            "config": {
                "model": embedder_model,
            },
        },
        "vector_store": {
            "provider": os.getenv("MEM0_VECTOR_STORE", "qdrant"),
            "config": {
                "collection_name": os.getenv("MEM0_COLLECTION", "mem0"),
                "path": os.getenv("MEM0_QDRANT_PATH", os.path.join(_DATA_DIR, "qdrant")),
                "embedding_model_dims": embedding_dims,
            },
        },
        "history_db_path": os.getenv(
            "MEM0_HISTORY_DB", os.path.join(_DATA_DIR, "history.db")
        ),
    }

    return config


def _mem0_call(func, *args, **kwargs):
    """Call a mem0 function and return JSON, catching errors."""
    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        logger.error("Mem0 call failed: %s", exc)
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False, default=str)


_MEMORY_INSTANCE: Optional[Memory] = None


def _get_memory() -> Memory:
    """Lazily initialize the Memory instance."""
    global _MEMORY_INSTANCE
    if _MEMORY_INSTANCE is None:
        config = _build_config()
        logger.info("Initializing mem0 Memory with config: %s", json.dumps(config, indent=2))
        os.makedirs(config["vector_store"]["config"]["path"], exist_ok=True)
        os.makedirs(os.path.dirname(config["history_db_path"]), exist_ok=True)
        _MEMORY_INSTANCE = Memory.from_config(config)
    return _MEMORY_INSTANCE


def create_server() -> FastMCP:
    """Create a FastMCP server with local mem0 tools."""

    server = FastMCP(
        "mem0",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8081")),
    )

    @server.tool(
        description="Store a new preference, fact, or conversation snippet. "
        "Requires at least one: user_id, agent_id, or run_id."
    )
    def add_memory(
        text: Annotated[
            str,
            Field(description="Plain sentence summarizing what to store."),
        ],
        messages: Annotated[
            Optional[list[Dict[str, str]]],
            Field(
                default=None,
                description="Structured conversation history with `role`/`content`. "
                "Use when you have multiple turns.",
            ),
        ] = None,
        user_id: Annotated[
            Optional[str],
            Field(default=None, description="Override the default user scope."),
        ] = None,
        agent_id: Annotated[
            Optional[str],
            Field(default=None, description="Optional agent identifier."),
        ] = None,
        run_id: Annotated[
            Optional[str],
            Field(default=None, description="Optional run identifier."),
        ] = None,
        metadata: Annotated[
            Optional[Dict[str, Any]],
            Field(default=None, description="Attach arbitrary metadata JSON."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Write durable information to mem0."""
        memory = _get_memory()
        uid = user_id or (ENV_DEFAULT_USER_ID if not (agent_id or run_id) else None)

        args = AddMemoryArgs(
            text=text,
            messages=[ToolMessage(**msg) for msg in messages] if messages else None,
            user_id=uid,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
        )

        # Build conversation payload
        conversation = (
            [m.model_dump() for m in args.messages] if args.messages else None
        )
        if not conversation:
            if args.text:
                conversation = [{"role": "user", "content": args.text}]
            else:
                return json.dumps(
                    {
                        "error": "messages_missing",
                        "detail": "Provide either `text` or `messages`.",
                    },
                    ensure_ascii=False,
                )

        kwargs: Dict[str, Any] = {}
        if args.user_id:
            kwargs["user_id"] = args.user_id
        if args.agent_id:
            kwargs["agent_id"] = args.agent_id
        if args.run_id:
            kwargs["run_id"] = args.run_id
        if args.metadata:
            kwargs["metadata"] = args.metadata

        return _mem0_call(memory.add, conversation, **kwargs)

    @server.tool(
        description="Run a semantic search over existing memories. "
        "user_id defaults to the server default if not provided."
    )
    def search_memories(
        query: Annotated[
            str, Field(description="Natural language description of what to find.")
        ],
        user_id: Annotated[
            Optional[str],
            Field(default=None, description="User scope for the search."),
        ] = None,
        agent_id: Annotated[
            Optional[str],
            Field(default=None, description="Agent scope for the search."),
        ] = None,
        run_id: Annotated[
            Optional[str],
            Field(default=None, description="Run scope for the search."),
        ] = None,
        limit: Annotated[
            Optional[int],
            Field(default=None, description="Maximum number of results."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Semantic search against existing memories."""
        memory = _get_memory()
        uid = user_id or (ENV_DEFAULT_USER_ID if not (agent_id or run_id) else None)

        kwargs: Dict[str, Any] = {"query": query}
        if uid:
            kwargs["user_id"] = uid
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id
        if limit:
            kwargs["limit"] = limit

        return _mem0_call(memory.search, **kwargs)

    @server.tool(
        description="List memories by scope. "
        "user_id defaults to the server default if not provided."
    )
    def get_memories(
        user_id: Annotated[
            Optional[str],
            Field(default=None, description="User scope for listing."),
        ] = None,
        agent_id: Annotated[
            Optional[str],
            Field(default=None, description="Agent scope for listing."),
        ] = None,
        run_id: Annotated[
            Optional[str],
            Field(default=None, description="Run scope for listing."),
        ] = None,
        limit: Annotated[
            Optional[int],
            Field(default=None, description="Maximum number of memories to return."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """List memories via scope filters."""
        memory = _get_memory()
        uid = user_id or (ENV_DEFAULT_USER_ID if not (agent_id or run_id) else None)

        kwargs: Dict[str, Any] = {}
        if uid:
            kwargs["user_id"] = uid
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id
        if limit:
            kwargs["limit"] = limit

        return _mem0_call(memory.get_all, **kwargs)

    @server.tool(description="Fetch a single memory by its memory_id.")
    def get_memory(
        memory_id: Annotated[
            str, Field(description="Exact memory_id to fetch.")
        ],
        ctx: Context | None = None,
    ) -> str:
        """Retrieve a single memory by ID."""
        memory = _get_memory()
        return _mem0_call(memory.get, memory_id)

    @server.tool(description="Overwrite an existing memory's text.")
    def update_memory(
        memory_id: Annotated[
            str, Field(description="Exact memory_id to overwrite.")
        ],
        text: Annotated[
            str, Field(description="Replacement text for the memory.")
        ],
        ctx: Context | None = None,
    ) -> str:
        """Overwrite an existing memory's text."""
        memory = _get_memory()
        return _mem0_call(memory.update, memory_id=memory_id, data=text)

    @server.tool(description="Delete one memory by its memory_id.")
    def delete_memory(
        memory_id: Annotated[
            str, Field(description="Exact memory_id to delete.")
        ],
        ctx: Context | None = None,
    ) -> str:
        """Delete a memory by ID."""
        memory = _get_memory()
        return _mem0_call(memory.delete, memory_id)

    @server.tool(
        description="Delete every memory in the given user/agent/run scope."
    )
    def delete_all_memories(
        user_id: Annotated[
            Optional[str],
            Field(default=None, description="User scope to delete; defaults to server user."),
        ] = None,
        agent_id: Annotated[
            Optional[str],
            Field(default=None, description="Optional agent scope to delete."),
        ] = None,
        run_id: Annotated[
            Optional[str],
            Field(default=None, description="Optional run scope to delete."),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Bulk-delete every memory in the confirmed scope."""
        memory = _get_memory()
        uid = user_id or ENV_DEFAULT_USER_ID

        args = DeleteAllArgs(user_id=uid, agent_id=agent_id, run_id=run_id)
        kwargs = args.model_dump(exclude_none=True)
        return _mem0_call(memory.delete_all, **kwargs)

    @server.tool(description="Get the history of changes for a specific memory.")
    def memory_history(
        memory_id: Annotated[
            str, Field(description="Memory ID to get history for.")
        ],
        ctx: Context | None = None,
    ) -> str:
        """Get the audit trail for a memory."""
        memory = _get_memory()
        return _mem0_call(memory.history, memory_id)

    @server.prompt()
    def memory_assistant() -> str:
        """Get help with memory operations."""
        return """You are using a local Mem0 MCP server for persistent memory.

Quick Start:
1. Store memories: Use add_memory to save facts, preferences, or conversations
2. Search memories: Use search_memories for semantic queries
3. List memories: Use get_memories to browse by user/agent/run scope
4. Update/Delete: Use update_memory and delete_memory for modifications
5. History: Use memory_history to see changes to a specific memory

All memories are scoped by user_id (defaults to the configured default user).
Data is stored locally — nothing leaves your machine."""

    return server


def main() -> None:
    """Run the MCP server over stdio."""
    server = create_server()
    logger.info("Starting local Mem0 MCP server (default user=%s)", ENV_DEFAULT_USER_ID)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
