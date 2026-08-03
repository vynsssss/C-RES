# -*- coding: utf-8 -*-
"""
vLLM backends for the agentic / ReAct / KB coordinators.

Each of AgenticCoordinator, ReactCoordinator, KBGroundingCoordinator subclasses
Coordinator (or BaseAgent) and calls self.generate(user_message=...,
system_message=..., max_new_tokens=...). They inherit ALL their multi-turn
orchestration (tool-selection turns, the ReAct loop, KB relevance filtering)
from those classes; only the single generate() call touches the model.

So, exactly like the static VLLMCoordinator, we swap generate() to hit a local
vLLM OpenAI server and leave everything else untouched. Rather than duplicate
that swap three times.

"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from base_agent import _resolve_sampling_params
from vllm_local_agent import LocalVLLMAgent

from coordinator_agentic import AgenticCoordinator
from coordinator_react import ReactCoordinator
from coordinator_kb import KBGroundingCoordinator
from coordinator_kb_synth import KBSynthCoordinator


class VLLMGenerateMixin:
    """Provides vLLM-backed generate() + tokenizer + matched context_window.

    Must appear BEFORE the coordinator class in the subclass MRO so that this
    __init__ and generate() take precedence.
    """

    _DEFAULT_CAP = 16384

    def _vllm_init(
        self,
        model_name: str,
        base_url: Optional[str],
        max_new_tokens: int,
        temperature: float,
        tool_registry=None,
        served_model_id: Optional[str] = None,
    ) -> None:
        # tokenizer is required: these coordinators call apply_chat_template /
        # encode directly. CPU-only, no weights, no GPU.
        from transformers import AutoTokenizer
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = "cuda"
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = max(0.0, min(2.0, temperature))
        self.tool_registry = tool_registry
        self.decisions_made = 0

        # context_window: replicate BaseAgent's min(detected, 16384).
        override = os.environ.get("VLLM_CONTEXT_WINDOW")
        if override:
            self.context_window = int(override)
        else:
            detected = self._detect_max_position(model_name)
            self.context_window = (
                min(detected, self._DEFAULT_CAP) if detected > 0 else self._DEFAULT_CAP
            )

        self._sampling_params: Dict[str, Any] = _resolve_sampling_params(
            model_name, temperature
        )
        self._agent = LocalVLLMAgent(
            model_name=model_name,
            served_model_id=served_model_id or model_name,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    @staticmethod
    def _detect_max_position(model_name: str) -> int:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            val = getattr(cfg, "max_position_embeddings", 0) or 0
            if not val:
                tc = getattr(cfg, "text_config", None)
                if tc is not None:
                    val = getattr(tc, "max_position_embeddings", 0) or 0
            return int(val)
        except Exception:
            return 0

    def generate(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if system_message is None:
            system_message = getattr(self, "system_prompt", None)
        return self._agent.generate(
            user_message=user_message,
            system_message=system_message,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
            temperature=temperature,
        )

    def get_statistics(self) -> Dict[str, Any]:
        stats = self._agent.get_statistics()
        stats["backend"] = "vllm-server"
        stats["model_name"] = self.model_name
        return stats


class VLLMAgenticCoordinator(VLLMGenerateMixin, AgenticCoordinator):
    """AgenticCoordinator with generation via the vLLM server."""

    def __init__(self, model_name, base_url=None, max_new_tokens=8192,
                 temperature=0.0, tool_registry=None, **kwargs):
        self._vllm_init(model_name, base_url, max_new_tokens, temperature, tool_registry)


class VLLMReactCoordinator(VLLMGenerateMixin, ReactCoordinator):
    """ReactCoordinator with generation via the vLLM server."""

    def __init__(self, model_name, base_url=None, max_new_tokens=8192,
                 temperature=0.0, tool_registry=None, **kwargs):
        self._vllm_init(model_name, base_url, max_new_tokens, temperature, tool_registry)


class VLLMKBGroundingCoordinator(VLLMGenerateMixin, KBGroundingCoordinator):
    """KBGroundingCoordinator with generation via the vLLM server.

    KB needs its retriever wired in addition to the generation swap, so this
    __init__ takes the KB-specific params (kb_retriever/kb_cache/selective/
    retrieve_n) and sets them just like KBGroundingCoordinator.__init__ does.
    """

    def __init__(self, model_name, base_url=None, max_new_tokens=8192,
                 temperature=0.0, tool_registry=None,
                 kb_retriever=None, kb_cache=None, selective=False,
                 retrieve_n=5, **kwargs):
        self._vllm_init(model_name, base_url, max_new_tokens, temperature, tool_registry)
        # KB-specific attributes (mirror KBGroundingCoordinator.__init__)
        self.kb = kb_retriever
        self.kb_cache = kb_cache
        self.selective = selective
        self.retrieve_n = retrieve_n


class VLLMKBSynthCoordinator(VLLMGenerateMixin, KBSynthCoordinator):
    """KBSynthCoordinator (KB docs through the multi-turn synthesize pipeline)
    with generation via the vLLM server. Sets kb_cache/kb/retrieve_n like
    KBSynthCoordinator.__init__ (note: no `selective` here -- that's only in the
    single-shot KBGroundingCoordinator)."""

    def __init__(self, model_name, base_url=None, max_new_tokens=8192,
                 temperature=0.0, tool_registry=None,
                 kb_cache=None, kb_retriever=None, retrieve_n=5, **kwargs):
        self._vllm_init(model_name, base_url, max_new_tokens, temperature, tool_registry)
        self.kb_cache = kb_cache
        self.kb = kb_retriever
        self.retrieve_n = retrieve_n
