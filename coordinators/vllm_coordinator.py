# -*- coding: utf-8 -*-
"""
VLLMCoordinator -- a Coordinator whose generate() calls a local vLLM

It subclasses the project's own Coordinator, so synthesize(), the prompt
construction, the think-strip / _classify_output / _parse_json_response parsing,
answer_changed logic, and the trajectory fields are ALL inherited unchanged.
Only the single generate() call is redirected to the vLLM server. That keeps
Step-2 output byte-compatible with the transformers path (same schema, same
parsing), while the server's continuous batching makes large models (gemma_31b,
70B) actually finish.
"""

from __future__ import annotations

import os

from typing import Any, Dict, Optional

from coordinator import Coordinator
from base_agent import _resolve_sampling_params
from vllm_local_agent import LocalVLLMAgent


class VLLMCoordinator(Coordinator):
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        served_model_id: Optional[str] = None,
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
        tool_registry=None,
    ):
 
        from transformers import AutoTokenizer
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = "cuda"
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = max(0.0, min(2.0, temperature))  # same clamp as BaseAgent
        self.tool_registry = tool_registry
     
        _DEFAULT_CAP = 16384
        override = os.environ.get("VLLM_CONTEXT_WINDOW")
        if override:
            self.context_window = int(override)
        else:
            detected = self._detect_max_position(model_name)
            self.context_window = min(detected, _DEFAULT_CAP) if detected > 0 else _DEFAULT_CAP
        self.decisions_made = 0

 
        self._sampling_params: Dict[str, Any] = _resolve_sampling_params(
            model_name, temperature
        )

        # The actual generation backend: local vLLM OpenAI server.
        self._agent = LocalVLLMAgent(
            model_name=model_name,
            served_model_id=served_model_id or model_name,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _detect_max_position(model_name: str) -> int:
        """Read max_position_embeddings from the model config WITHOUT loading
        weights, mirroring BaseAgent's auto-detect """
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
            system_message = self.system_prompt
        return self._agent.generate(
            user_message=user_message,
            system_message=system_message,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        stats = self._agent.get_statistics()
        stats["backend"] = "vllm-server"
        stats["model_name"] = self.model_name
        return stats
