"""Thread-local OpenAI client caching for improved performance.

This module provides a caching mechanism that reuses OpenAI client instances
per thread and endpoint configuration, avoiding the overhead of creating new
HTTP clients for each API request.
"""
from __future__ import annotations

import threading
from typing import Optional

_thread_local = threading.local()


def get_openai_client(api_dict: Optional[dict] = None):
    """Return a thread-local cached OpenAI client.

    Creating a new OpenAI client per request is expensive (new HTTP client,
    TLS/session setup, etc.) and can significantly slow down local inference
    backends like vLLM that can otherwise serve quickly.

    `api_dict` supports:
    - api_base: str
    - api_key: str
    - timeout: float (seconds)
    """
    import openai

    api_base = None
    api_key = None
    timeout = None
    if api_dict:
        api_base = api_dict.get("api_base")
        api_key = api_dict.get("api_key")
        timeout = api_dict.get("timeout")

    cache = getattr(_thread_local, "openai_clients", None)
    if cache is None:
        cache = {}
        _thread_local.openai_clients = cache

    key = (api_base, api_key, timeout)
    client = cache.get(key)
    if client is None:
        kwargs = {}
        if api_base:
            kwargs["base_url"] = api_base
        if api_key:
            kwargs["api_key"] = api_key
        if timeout is not None:
            kwargs["timeout"] = timeout
        client = openai.OpenAI(**kwargs) if kwargs else openai.OpenAI()
        cache[key] = client
    return client

