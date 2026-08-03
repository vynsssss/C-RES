# -*- coding: utf-8 -*-
"""
Base Agent

Provides:
- Text generation via model.generate() with auto chat template
- Optional tool calling via tool_registry (Step 2)
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-model recommended sampling parameters.
# ---------------------------------------------------------------------------
_MODEL_SAMPLING_PARAMS: Dict[str, Dict[str, Any]] = {
    # OLMo-3-7B-Instruct / OLMo-3-7B-Think / OLMo-3.1-32B-Instruct / OLMo-3.1-32B-Think
    "olmo": {
        "top_p": 0.95,
        "top_k": 0,     # 0 = disabled
        "min_p": None,
    },
    # Qwen3-4B-Instruct-2507 / Qwen3-30B-A3B-Instruct-2507
    "qwen3": {
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
    },
    "llama": {
        "top_p": 0.9,
        "top_k": 0,     # 0 = disabled
        "min_p": None,
    },
    # Gemma 4 E2B/E4B/26B/31B — Google-recommended chat config.
    "gemma": {
        "top_p": 0.95,
        "top_k": 64,
        "min_p": None,
    },
}


def _resolve_sampling_params(model_name: str, temperature: float) -> Dict[str, Any]:
    """
    Return HF generate() sampling kwargs for the given model + temperature.

    - temperature == 0.0  ->  greedy (do_sample=False, no sampling kwargs)
    - temperature  > 0.0  ->  model-specific top_p / top_k / min_p from the table.
                              The caller-supplied temperature always takes priority.

    """
    if temperature == 0.0:
        return {"do_sample": False}

    name_lower = model_name.lower()
    matched_key = next((k for k in _MODEL_SAMPLING_PARAMS if k in name_lower), None)

    if matched_key is None:
        logger.warning(
            "No sampling config for '%s'. Falling back to top_p=0.9, top_k=0.",
            model_name,
        )
        params = {"top_p": 0.9, "top_k": 0, "min_p": None}
    else:
        params = _MODEL_SAMPLING_PARAMS[matched_key]

    return {
        "do_sample":   True,
        "temperature": temperature,
        "top_p":       params["top_p"],
        "top_k":       params["top_k"],   # 0 or positive int -- never None
        "min_p":       params["min_p"],   # float or None
    }


class BaseAgent(ABC):
    """
    Base class for all agents.

    - Step 1: tool_registry=None -> pure generation, no tools
    - Step 2: tool_registry=registry -> enables call_tool()

    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        context_window: int = 0,
        tool_registry=None,
        model_name: str = "",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = max(0.0, min(2.0, temperature))
        self.max_new_tokens = max_new_tokens
        self.tool_registry = tool_registry
        self.model_name = model_name

        if context_window > 0:
            self.context_window = context_window
        else:
            detected = getattr(getattr(model, "config", None),
                               "max_position_embeddings", 0)
            # Cap at 16384 -- larger values need more KV cache memory.
            # Models like Llama-3.1 (128K) and Qwen3 (262K) technically
            # support more, but 16K is safe for single/dual A100 GPUs.
            _DEFAULT_CAP = 16384
            self.context_window = min(detected, _DEFAULT_CAP) if detected > 0 else _DEFAULT_CAP
            logger.info(
                "[Context] %s: model config = %s, using %d (cap %d)",
                model_name, detected or "not found", self.context_window, _DEFAULT_CAP,
            )


        self._sampling_params: Dict[str, Any] = _resolve_sampling_params(
            self.model_name, self.temperature
        )
        self._log_sampling_params()


        self.generation_count = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0
        self.tool_call_count = 0

    def _log_sampling_params(self) -> None:
        """Print resolved sampling params once at construction -- self-documenting runs."""
        sp = self._sampling_params
        if not sp.get("do_sample", True):
            logger.info("[Sampling] %s -> greedy (do_sample=False)", self.model_name)
            return
        top_k_str = str(sp["top_k"]) if sp["top_k"] != 0 else "-- (disabled)"
        min_p_str = str(sp["min_p"]) if sp["min_p"] is not None else "-- (disabled)"
        logger.info(
            "[Sampling] %s -> do_sample=True  temp=%.2f  top_p=%.2f  top_k=%s  min_p=%s",
            self.model_name, sp["temperature"], sp["top_p"], top_k_str, min_p_str,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a response using the model's chat template.

        Args:
            user_message:   The user-facing prompt.
            system_message: Overrides self.system_prompt for this call only.
            max_new_tokens: Overrides self.max_new_tokens for this call only.
            temperature:    Overrides self.temperature for this call only.
                            Rare -- kept for callers that need a one-off
                            temperature (currently unused by the pipeline).

        Returns:
            Generated text (assistant turn only, no prompt echo).

        Raises:
            ValueError:  If prompt is too long to leave room for generation.
        """
        if system_message is None:
            system_message = self.system_prompt
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # Use cached params for the normal case (temperature=None or same value).
        # Only fall back to a fresh resolve for the rare per-call override.
        if temperature is None or temperature == self.temperature:
            sp = self._sampling_params
        else:
            sp = _resolve_sampling_params(self.model_name, temperature)

        # ------------------------------------------------------------------
        # Build chat-formatted prompt via the model's own chat template
        # ------------------------------------------------------------------
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # ------------------------------------------------------------------
        # Tokenise
        # ------------------------------------------------------------------
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_window,
            padding=False,
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]
        effective_max = min(max_new_tokens, self.context_window - input_len - 10)

        if effective_max < 50:
            raise ValueError(
                f"Prompt too long ({input_len} tokens); "
                f"only {effective_max} tokens left for generation."
            )

        # ------------------------------------------------------------------
        # Build generate() kwargs
        # ------------------------------------------------------------------
        # Always-required kwargs -- never conditionally filtered.
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens":     effective_max,
            "do_sample":          sp["do_sample"],
            "repetition_penalty": 1.1,
            "pad_token_id":       self.tokenizer.pad_token_id,
            "eos_token_id":       self.tokenizer.eos_token_id,
            "renormalize_logits": True,
        }

        if sp["do_sample"]:
            gen_kwargs["temperature"] = sp["temperature"]
            gen_kwargs["top_p"]       = sp["top_p"]
            gen_kwargs["top_k"]       = sp["top_k"]   # 0 or positive int
            if sp["min_p"] is not None:
                gen_kwargs["min_p"]   = sp["min_p"]

        # ------------------------------------------------------------------
        # Generate
        # ------------------------------------------------------------------
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the new tokens (exclude echoed prompt)
        new_tokens = outputs[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        gen_time = time.time() - start
        self.generation_count += 1
        self.total_tokens_generated += len(new_tokens)
        self.total_generation_time += gen_time

        return text

    # ------------------------------------------------------------------
    # Tool Calling (Step 2 only -- ignored in Step 1)
    # ------------------------------------------------------------------
    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool through the registry."""
        if self.tool_registry is None:
            return {"success": False, "error": "No tool registry available"}
        try:
            # ToolRegistry's method is .call() -- not .call_tool().
            # The previous .call_tool() spelling was a latent bug that
            # would AttributeError if any agent ever invoked this path
            # (cultural_helper bypasses BaseAgent and calls registry.call
            # directly, so step2 never hit it).
            result = self.tool_registry.call(tool_name, **kwargs)
            self.tool_call_count += 1
            return result
        except KeyError:
            logger.error("Tool '%s' not registered", tool_name)
            return {"success": False, "error": f"Tool '{tool_name}' not found"}
        except Exception as e:
            logger.error("Tool '%s' failed: %s", tool_name, e)
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "name":               self.name,
            "generations":        self.generation_count,
            "tool_calls":         self.tool_call_count,
            "tokens_generated":   self.total_tokens_generated,
            "total_time_s":       round(self.total_generation_time, 1),
            "avg_tokens_per_gen": (
                self.total_tokens_generated / self.generation_count
                if self.generation_count > 0 else 0
            ),
            "avg_time_per_gen_s": (
                round(self.total_generation_time / self.generation_count, 2)
                if self.generation_count > 0 else 0
            ),
        }