# -*- coding: utf-8 -*-
"""
LocalVLLMAgent -- a drop-in model backend that talks to a LOCAL vLLM

It implements the same `generate(user_message, system_message, max_new_tokens,
temperature)` contract as BaseAgent / APIAgent, so Coordinator,
coordinator_agentic, and coordinator_react accept it unchanged. This is how the
open-weight Step-2 agentic / ReAct / static runs get vLLM's continuous batching
without rewriting the multi-turn loop: many samples run concurrently against one
server, and the server batches them internally.

Sampling is pulled from the project's own per-family table
(base_agent._resolve_sampling_params) so the numbers match the rest of the
pipeline. The server is addressed via $VLLM_BASE_URL (default localhost:8000).

"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Optional

import requests

from base_agent import _resolve_sampling_params

logger = logging.getLogger(__name__)


class LocalVLLMAgent:
    def __init__(
        self,
        model_name: str,
        served_model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
        max_retries: int = 4,
        retry_delay: float = 2.0,
        request_timeout: float = 600.0,
    ):
        self.model_name = model_name
        # vLLM's OpenAI server serves under the name you launched it with;
        # by default that's the full HF repo string.
        self.served_model_id = served_model_id or model_name
        base = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000")
        self.chat_url = base.rstrip("/") + "/v1/chat/completions"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_timeout = request_timeout

        # Resolve sampling once (same table as the HF path).
        self._sp = _resolve_sampling_params(model_name, temperature)

        # Stats (mirror BaseAgent.get_statistics keys).
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0
        self.tool_call_count = 0

    # ------------------------------------------------------------------
    def _payload(self, user_message: str, system_message: Optional[str],
                 max_tok: int, temperature: Optional[float]) -> Dict[str, Any]:
        msgs = []
        if system_message:
            msgs.append({"role": "system", "content": system_message})
        msgs.append({"role": "user", "content": user_message})

        payload: Dict[str, Any] = {
            "model": self.served_model_id,
            "messages": msgs,
            "max_tokens": max_tok,
            "repetition_penalty": 1.1,   # match base_agent
        }

        # Per-call temperature override falls back to the constructor value.
        temp = self.temperature if temperature is None else temperature
        sp = self._sp if temp == self.temperature else _resolve_sampling_params(self.model_name, temp)

        if not sp.get("do_sample", True):
            payload.update({"temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0})
        else:
            top_k = sp["top_k"]
            payload.update({
                "temperature": sp["temperature"],
                "top_p": sp["top_p"],
                "top_k": (-1 if (top_k is None or top_k == 0) else int(top_k)),
                "min_p": (sp["min_p"] if sp["min_p"] is not None else 0.0),
            })
        return payload

    # ------------------------------------------------------------------
    def generate(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        max_tok = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        payload = self._payload(user_message, system_message, max_tok, temperature)
        headers = {"Authorization": "Bearer EMPTY", "Content-Type": "application/json"}

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.time()
                resp = requests.post(self.chat_url, json=payload, headers=headers,
                                     timeout=self.request_timeout)
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"] or ""

                dt = time.time() - start
                self.generation_count += 1
                self.total_generation_time += dt
                usage = data.get("usage") or {}
                self.total_tokens_generated += int(usage.get("completion_tokens", 0))
                return text
            except Exception as e:  # noqa: BLE001 -- surface after retries
                last_err = e
                if attempt < self.max_retries:
                    wait = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning("vLLM request failed (attempt %d/%d): %s -- retrying in %.1fs",
                                   attempt, self.max_retries, e, wait)
                    time.sleep(wait)
        raise RuntimeError(f"LocalVLLMAgent.generate failed after {self.max_retries} attempts: {last_err}")

    # ------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        n = self.generation_count
        return {
            "name": "LocalVLLMAgent",
            "generations": n,
            "tool_calls": self.tool_call_count,
            "tokens_generated": self.total_tokens_generated,
            "total_time_s": round(self.total_generation_time, 1),
            "avg_tokens_per_gen": (self.total_tokens_generated / n if n else 0),
            "avg_time_per_gen_s": (round(self.total_generation_time / n, 2) if n else 0),
        }
