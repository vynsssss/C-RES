# -*- coding: utf-8 -*-
"""
Two modes:
- Step 1: evaluate(prompt) -> generate reasoning + parse yes/no/neutral
- Step 2: synthesize(baseline_reasoning, evidence) -> updated reasoning + final answer

Output format:
    Plain-text "Answer: <Yes|No|Neither>\\nExplanation: <...>" is the PRIMARY
    format produced and parsed. 
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple

from base_agent import BaseAgent

_ATLAS_TOTAL_CAP   = 10000  # max chars total for Cultural Atlas evidence
_ATLAS_AXIS_CAP    = 3000   # max chars per individual axis (not 5000 like wiki --
                            #   Core Concepts + Etiquette average 5k+ each, so
                            #   axis_cap=5000 leaves only 1-2 axes for Japan/Palestine.
                            #   3000 gives 3-7 axes with good depth per axis.)
_ATLAS_FIELD_CAP   = 4000   # max chars per individual content field value
                            # (bumped from 500 after Atlas re-scrape recovered
                            # full content -- was clipping ~50% of single-text axes)
_ATLAS_LIST_CAP    =   15   # max list items per field (Etiquette.Eating has up to 24)
                            # Without caps, Atlas averages ~24k chars per country
                            # (Japan: 36k, Palestine: 38k) which overflows the
                            # 16384-token context window and causes ValueError.

# ---------------------------------------------------------------------------
# Reading-turn (multi-turn synthesize) -- per-model max_new_tokens ceilings
# ---------------------------------------------------------------------------
# In multi-turn synthesize(), each tool's evidence is first compressed by a
# "reading turn" (extract scenario-relevant points), then synthesis sees only
# the compressed summaries. This bounds total tokens per call regardless of
# how big the raw evidence is.

_READING_MAX_TOKENS = {
    "olmo_7b":        2048,
    "olmo_7b_think":  3072,
    "olmo_32b":       2048,
    "olmo_32b_think": 3584,
}
_READING_MAX_TOKENS_FLOOR   = 512    # never less than this, even with tight context
_READING_MAX_TOKENS_DEFAULT = 1024   # for unknown model identifiers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns 
# ---------------------------------------------------------------------------
_RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
"""Strip Qwen3 <think>...</think> blocks before any parsing."""

# Canonical NorMAD label space (gold labels). Model outputs of "neither"
# (from paper-verbatim Step 1 / Step 2 prompts) are canonicalised to
# "neutral" before scoring so the model and gold share one label space.
_VALID_ANSWERS = frozenset({"yes", "no", "neutral"})


def _canonicalise(raw: str) -> str:
    """Map model output to the NorMAD canonical label space.

    'neither' -> 'neutral'. 'yes' / 'no' / 'neutral' / 'unknown' pass
    through unchanged. Any non-canonical label (e.g. OLMo 'modifies')
    also passes through -- downstream whitelist gate rejects it.
    """
    d = (raw or "").lower().strip()
    if d == "neither":
        return "neutral"
    return d


# ---------------------------------------------------------------------------
# Legacy CONCLUSION:/DECISION: labels 
# ---------------------------------------------------------------------------
_RE_CONCLUDE = re.compile(r"conclusion:\s*\*{0,2}\s*(yes|no|neutral|neither)", re.IGNORECASE)
_RE_DECIDE   = re.compile(r"decision:\s*\*{0,2}\s*(yes|no|neutral|neither)",   re.IGNORECASE)
"""Legacy CONCLUSION:/DECISION: labels -- kept for old data compatibility."""

# ---------------------------------------------------------------------------
# Answer/Explanation plain-text format  
# ---------------------------------------------------------------------------

_RE_ANSWER = re.compile(
    r"answer\s*\(?[^)]*?\)?\s*:\s*\*{0,2}\s*(?:[123]\)?\s*)?(yes|no|neutral|neither)",
    re.IGNORECASE,
)
"""Captures 'yes'/'no'/'neutral'/'neither' after an 'Answer:' label."""

# Number-form fallback: "Answer: 1" / "Answer: 2" / "Answer: 3" -- model picked
# the option NUMBER from the Options list instead of the word. NorMAD prompt is:
#     Options:
#     1) Yes
#     2) No
#     3) Neither
# so 1->yes, 2->no, 3->neither (canonicalised to neutral by _canonicalise).
_RE_ANSWER_NUMBER = re.compile(
    r"answer\s*\(?[^)]*?\)?\s*:\s*\*{0,2}\s*([123])\b",
    re.IGNORECASE,
)
_OPTION_NUMBER_TO_WORD = {"1": "yes", "2": "no", "3": "neither"}

# ---------------------------------------------------------------------------
# Output quality classification -- detect degenerate / empty model output
# ---------------------------------------------------------------------------
def _classify_output(text: str) -> str:
    """
    Classify raw model output quality BEFORE parsing.

    Returns one of:
      'ok'           -- text looks coherent, proceed with parsing
      'empty'        -- model produced nothing (< 20 chars)
      'degenerated'  -- repetitive loops, gibberish, model overwhelmed by context

    Applied universally to all models. Detection heuristics:
      1. Very short output -> 'empty'
      2. Last 100 words have   3 unique words -> repetition loop
      3. Non-breaking space flood (OLMo atlas pattern) -> degenerated
      4. Repeated 5+ char substring appears 5+ times in last 500 chars -> loop
    """
    if len(text) < 20:
        return "empty"

    # Check last 100 words for repetition
    words = text.split()
    if len(words) > 30:
        tail_words = words[-100:] if len(words) >= 100 else words[-30:]
        unique_ratio = len(set(w.lower() for w in tail_words)) / len(tail_words)
        if unique_ratio < 0.08:  # < 8% unique words -> degeneration
            return "degenerated"

    # Non-breaking space flood (\xa0) -- OLMo atlas/wiki pattern
    if text.count("\xa0") > len(text) * 0.05:  # > 5% of chars are \xa0
        return "degenerated"

    # Repeated substring pattern in tail (e.g. "Thus. Thus. Thus.")
    tail = text[-500:]
    for chunk_len in (5, 8, 12):
        if len(tail) < chunk_len * 6:
            continue
        # Check if any substring of this length repeats 5+ times
        chunks = [tail[i:i+chunk_len] for i in range(0, len(tail) - chunk_len, chunk_len)]
        from collections import Counter
        most_common_count = Counter(chunks).most_common(1)[0][1]
        if most_common_count >= max(5, len(chunks) * 0.4):
            return "degenerated"

    return "ok"


# ---------------------------------------------------------------------------
# Reasoning / explanation extraction patterns
# ---------------------------------------------------------------------------
# Legacy REASONING:/CONCLUSION: format
_RE_REASONING_WITH_END = re.compile(
    r"reasoning:\s*(.+?)(?=\s*(?:CONCLUSION|DECISION)\s*:)",
    re.IGNORECASE | re.DOTALL,
)
_RE_REASONING_TO_END = re.compile(
    r"reasoning:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
_RE_SPLIT_CONCLUDE = re.compile(r"(?:CONCLUSION|DECISION):", re.IGNORECASE)
"""Used to split response before the conclusion label."""

# Current Explanation: capture group -- primary in current format
_RE_EXPLANATION_TO_END = re.compile(
    r"explanation\s*:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
_RE_SPLIT_ANSWER = re.compile(r"answer\s*\(?[^)]*?\)?\s*:", re.IGNORECASE)
"""Used to find the boundary between leading prose and the Answer: label."""


class Coordinator(BaseAgent):
    """
    Coordinator agent for cultural reasoning.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_new_tokens: int = 2048,
        context_window: int = 0,
        temperature: float = 0.0,
        tool_registry=None,
        model_name: str = "",
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            temperature=temperature,
            tool_registry=tool_registry,
            model_name=model_name,
        )
        self.decisions_made = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "Coordinator"

    @property
    def system_prompt(self) -> str:
        """System prompt for Step 1 evaluation.
        """
        return (
            "Analyze cultural scenarios carefully and answer the question.\n\n"
            "Format your response EXACTLY as:\n"
            "Answer: <Yes|No|Neither>\n"
            "Explanation: <your analysis>"
        )

    @property
    def synthesis_system_prompt(self) -> str:
        """System prompt for Step 2 synthesis.
        """
        return (
            "You previously analyzed a cultural scenario and gave an initial "
            "answer with reasoning. You now have new cultural evidence from "
            "research tools.\n\n"
            "Do not make any extra inferences about actions outside of the "
            "given context and the provided cultural evidence. Only align to "
            "the scenario and the evidence given.\n\n"
            "INSTRUCTIONS:\n"
            "1. Review your initial reasoning and answer.\n"
            "2. Read the evidence. Identify which specific pieces are "
            "relevant to the scenario, quoting them in your reasoning. "
            "If no evidence is relevant, state exactly: \"No relevant "
            "evidence found for this situation.\"\n"
            "3. For the relevant evidence, analyze whether it supports or "
            "contradicts your initial answer.\n"
            "4. State your final answer (Yes, No, or Neither). If it differs "
            "from your initial answer, explain what specific evidence "
            "justifies the change. If it matches your initial answer, "
            "explain what supports keeping it.\n\n"
            "Format your response EXACTLY as:\n"
            "Answer: <Yes|No|Neither>\n"
            "Explanation: <your analysis>"
        )

    @property
    def reading_system_prompt(self) -> str:
        """System prompt for the reading turn in multi-turn synthesize().
        """
        return (
            "You will be given a situation and cultural evidence.\n"
            "Your task is to extract from the evidence the cultural "
            "information that is relevant to the situation.\n\n"
            "Quote relevant passages directly from the evidence, then "
            "briefly explain how each relates to the situation. Focus on "
            "cultural norms, attitudes, or practices."
        )

    # ------------------------------------------------------------------
    # Abstract method implementation
    # ------------------------------------------------------------------
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Delegates to evaluate()."""
        return self.evaluate(input_data.get("prompt", ""))

    # ------------------------------------------------------------------
    # Multi-turn helpers (used by synthesize())
    # ------------------------------------------------------------------
    def _model_short_name(self) -> str:
        """
        Map self.model_name to short identifier for per-model config.
        """
        name_lower = self.model_name.lower()
        families = ["qwen", "llama", "olmo", "mistral", "gemma"]
        family = next((f for f in families if f in name_lower), None)
        if family is None:
            parts = self.model_name.split("/")
            family = parts[-1].split("-")[0].lower() if parts else "base"
        size_match = re.search(r"(\d+\.?\d*)\s*[bB]", self.model_name)
        size = f"{size_match.group(1)}b" if size_match else "base"
        identifier = f"{family}_{size}"
        if "think" in name_lower:
            identifier += "_think"
        return identifier

    def _read_evidence(
        self,
        scenario_for_reading: str,
        evidence_block: str,
        tool_label: str,
    ) -> Tuple[str, str]:
        """Reading turn: extract situation-relevant points from one tool's evidence.

        Args:
            scenario_for_reading: Country/Story (or RoT/Story etc.) only --
                NO Task description, NO Options, NO Answer prompt. Keeping
                the judgment framing out of Turn 1 prevents small models
                from preemptively answering yes/no.
            evidence_block: Pre-formatted text from one tool (Atlas or Wiki).
            tool_label: Human-readable label for logging (e.g. "Cultural Atlas").

        Returns:
            (relevance_text, status) where status is one of:
              - "ok"           -> useful summary
              - "empty"        -> model produced nothing
              - "degenerated"  -> model output was malformed/repetitive
              - "trimmed"      -> input too long to leave room for output
        """
        user_message = (
            f"{scenario_for_reading}\n\n"
            f"{evidence_block}\n\n"
            "What in this evidence is relevant to the situation above?"
        )

        # Per-model ceiling on max_new_tokens
        short = self._model_short_name()
        ceiling = _READING_MAX_TOKENS.get(short, _READING_MAX_TOKENS_DEFAULT)

        # Measure actual input tokens (model-specific tokenizer)
        messages = [
            {"role": "system", "content": self.reading_system_prompt},
            {"role": "user",   "content": user_message},
        ]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_tokens = len(self.tokenizer.encode(formatted))

        # Adaptive: never exceed context room or per-model ceiling
        context_room = self.context_window - input_tokens - 50
        adaptive_max = min(ceiling, context_room)

        # If even the floor doesn't fit, evidence is too large for this turn
        if adaptive_max < _READING_MAX_TOKENS_FLOOR:
            logger.warning(
                "[Reading:%s] Insufficient context -- input=%d, room=%d (need >=%d)",
                tool_label, input_tokens, context_room, _READING_MAX_TOKENS_FLOOR,
            )
            return "", "trimmed"

        # Apply floor
        adaptive_max = max(adaptive_max, _READING_MAX_TOKENS_FLOOR)

        logger.info(
            "[Reading:%s] input=%d tokens, max_new=%d (model=%s, ceiling=%d)",
            tool_label, input_tokens, adaptive_max, short, ceiling,
        )

        raw = self.generate(
            user_message=user_message,
            system_message=self.reading_system_prompt,
            max_new_tokens=adaptive_max,
        )

        stripped = _RE_THINK.sub("", raw, count=0).strip()
        status = _classify_output(stripped)

        if status in ("empty", "degenerated"):
            logger.warning(
                "[Reading:%s] %s output (%d chars)",
                tool_label, status, len(stripped),
            )
            return stripped, status

        return stripped, "ok"

    # ------------------------------------------------------------------
    # Step 1: Evaluate
    # ------------------------------------------------------------------
    def evaluate(self, prompt: str) -> Dict[str, Any]:
        """
        Step 1: Generate initial analysis and parse answer.

        Returns:
            Dict with decision, reasoning, raw_response, output_status.
            output_status is one of: 'ok', 'empty', 'degenerated'.
        """
        raw = self.generate(user_message=prompt)

        # Strip think blocks once -- reuse for both parsers
        stripped = _RE_THINK.sub("", raw, count=0).strip()

        # Classify output quality before parsing
        output_status = _classify_output(stripped)

        if output_status in ("empty", "degenerated"):
            # Don't attempt to parse garbage -- record honestly
            self.decisions_made += 1
            return {
                "decision":      output_status,
                "reasoning":     "",
                "raw_response":  raw,
                "output_status": output_status,
            }

        parsed    = self._parse_json_response(stripped)
        decision  = parsed["decision"]
        reasoning = parsed["reasoning"]

        self.decisions_made += 1

        return {
            "decision":      decision,
            "reasoning":     reasoning,
            "raw_response":  raw,
            "output_status": "ok",
        }

    # ------------------------------------------------------------------
    # Step 2: Synthesize (multi-turn -- read each tool, then synthesize)
    # ------------------------------------------------------------------
    def _build_reading_scenario(
        self,
        prompt_type: str,
        story: str,
        country: str,
        value: str = "",
        rot: str = "",
    ) -> str:
        """
        Compact situation context for the per-tool reading turn.
        """
        if prompt_type == "story_country_value":
            return f"Country: {country}\nRoT: {value}\nStory: {story}"
        if prompt_type == "story_rot":
            return f"Rule-of-thumb: {rot}\nStory: {story}"
        # story_country (default)
        return f"Country: {country}\nStory: {story}"

    def synthesize(
        self,
        prompt_type: str,
        story: str,
        country: str,
        baseline_reasoning: str,
        baseline_answer: str,
        hofstede_data: Optional[Dict[str, Any]] = None,
        cultural_atlas_data: Optional[Dict[str, Any]] = None,
        wikipedia_data: Optional[Dict[str, Any]] = None,
        value: str = "",
        rot: str = "",
    ) -> Dict[str, Any]:
        """
        Step 2: Multi-turn synthesize -- Turn 1 reads each tool, Turn 2 judges.

        Pipeline:
          - Hofstede always passes through (small, ~250 tokens).
          - Atlas/Wiki: format full evidence -> reading turn -> relevance summary.
          - Synthesis (Turn 2) sees: scenario + initial analysis + compressed
            evidence + paper-verbatim Options/Answer block.

        Turn 1 user prompt: situation context only (no Task, no Options).
        Turn 2 user prompt: paper-verbatim Task + scenario + initial analysis
                            + cultural evidence + Options + Answer block.

        Returns:
            Dict with decision, reasoning, raw_response, answer_changed,
            tools_used, output_status, reading_status, reading_outputs,
            wiki_truncation.

        reading_outputs: the per-tool reading-turn text from Turn 1 is
            preserved alongside reading_status, so trajectories now contain
            both "what each tool said when read" and "what the synthesis
            decided after combining them" -- useful for ablation analysis.
        """
        if prompt_type == "story_country_value":
            scenario_for_reading = (
                f"Country: {country}\n"
                f"RoT: {value}\n"
                f"Story: {story}"
            )
            task_lines = (
                "Task: You will be given a short story involving 2 or more "
                "characters about everyday situations. You will also be given "
                "a country of relevance, along with a rule-of-thumb. Read the "
                "story and reason if the action by the character in the story "
                "is socially acceptable in the given country with the "
                "associated rule-of-thumb. The answer options are Yes, No or "
                "Neither."
            )
            scenario_block = (
                f"Country: {country}\n"
                f"RoT: {value}\n"
                f"Story: {story}"
            )
        elif prompt_type == "story_rot":
            scenario_for_reading = (
                f"Rule-of-thumb: {rot}\n"
                f"Story: {story}"
            )
            task_lines = (
                "Task: You will be given a short story involving 2 or more "
                "characters about everyday situations. You will also be given "
                "a rule-of-thumb. Read the story and reason if the action by "
                "the character in the story is socially acceptable under the "
                "given rule-of-thumb alone. The answer options are Yes, No or "
                "Neither."
            )
            scenario_block = (
                f"Rule-of-thumb: {rot}\n"
                f"Story: {story}"
            )
        else:  # story_country
            scenario_for_reading = (
                f"Country: {country}\n"
                f"Story: {story}"
            )
            task_lines = (
                "Task: You will be given a short story involving 2 or more "
                "characters about everyday situations. You will also be given "
                "a country of relevance. Read the story and reason if the "
                "action by the character in the story is socially acceptable "
                "in the given country. The answer options are Yes, No or "
                "Neither."
            )
            scenario_block = (
                f"Country: {country}\n"
                f"Story: {story}"
            )

        _EVIDENCE_PLACEHOLDER = "{{EVIDENCE}}"
        skeleton_user = (
            f"{task_lines}\n\n"
            f"{scenario_block}\n\n"
            f"Initial Analysis: {baseline_reasoning}\n"
            f"Initial Answer: {baseline_answer}\n\n"
            f"Cultural Evidence:\n"
            f"{_EVIDENCE_PLACEHOLDER}\n\n"
            f"Options:\n"
            f"1) Yes\n"
            f"2) No\n"
            f"3) Neither\n\n"
            f"Answer (options Yes, No or Neither):\n"
            f"Explanation:"
        )

        #    Measure skeleton tokens
        skeleton_messages = [
            {"role": "system", "content": self.synthesis_system_prompt},
            {"role": "user",   "content": skeleton_user},
        ]
        skeleton_formatted = self.tokenizer.apply_chat_template(
            skeleton_messages, tokenize=False, add_generation_prompt=True
        )
        skeleton_tokens = len(self.tokenizer.encode(skeleton_formatted))

        # Calculate evidence token budget
        available_input = self.context_window - self.max_new_tokens - 100
        evidence_token_budget = max(available_input - skeleton_tokens, 500)

        logger.info(
            "[Budget] context=%d, skeleton=%d tokens, evidence budget=%d tokens",
            self.context_window, skeleton_tokens, evidence_token_budget,
        )

        # Fill evidence -- multi-turn for large tools
        # Hofstede: passes through unchanged (small).
        # Atlas/Wiki: format full -> Turn 1 reading -> relevance summary.
        evidence_parts: List[str] = []
        tools_used: List[str] = []
        reading_status:  Dict[str, str] = {}
        reading_outputs: Dict[str, str] = {}   # NEW: per-tool reading-turn text
        tokens_used = 0

        # 1. Hofstede -- always emit a block when requested, even on failure.
        if hofstede_data is not None:
            if hofstede_data.get("success"):
                block = self._format_hofstede(hofstede_data)
                reading_status["hofstede"]  = "ok"
                reading_outputs["hofstede"] = block   # what synthesis actually sees
                tools_used.append("hofstede_tool")
            else:
                block = (
                    "[Hofstede Cultural Dimensions]\n"
                    "  Not available for this country."
                )
                reading_status["hofstede"] = "unavailable"
                # No reading_outputs entry on unavailable -- only ok statuses.
            block_tokens = len(self.tokenizer.encode(block))
            if tokens_used + block_tokens < evidence_token_budget:
                evidence_parts.append(block)
                tokens_used += block_tokens + 5

        # 2. Cultural Atlas -- always emit a block when requested.
        if cultural_atlas_data is not None:
            if cultural_atlas_data.get("retrieved"):
                atlas_block = self._format_cultural_atlas(
                    cultural_atlas_data, max_chars=_ATLAS_TOTAL_CAP,
                )
                if len(atlas_block) > 50:
                    relevance, status = self._read_evidence(
                        scenario_for_reading, atlas_block, "Cultural Atlas",
                    )
                    reading_status["atlas"] = status
                    if status == "ok":
                        reading_outputs["atlas"] = relevance   # save reading-turn text
                        block = (
                            "[Cultural Atlas -- relevant points]\n"
                            f"{relevance}"
                        )
                        tools_used.append("cultural_atlas_tool")
                    else:
                        # Reading turn failed (degenerated/empty/trimmed).
                        # Surface the failure so the model knows.
                        block = (
                            "[Cultural Atlas -- relevant points]\n"
                            f"  Reading-turn output was {status}; treat as unavailable."
                        )
                        # On failure, still record what came out (may be empty
                        # or partial garbage) -- useful for diagnosis.
                        if relevance:
                            reading_outputs["atlas"] = relevance
                else:
                    reading_status["atlas"] = "empty"
                    block = (
                        "[Cultural Atlas]\n"
                        "  Not available for this country."
                    )
            else:
                reading_status["atlas"] = "unavailable"
                block = (
                    "[Cultural Atlas]\n"
                    "  Not available for this country."
                )
            block_tokens = len(self.tokenizer.encode(block))
            if tokens_used + block_tokens < evidence_token_budget:
                evidence_parts.append(block)
                tokens_used += block_tokens + 5

        # 3. Wikipedia -- always emit a block when requested.
        if wikipedia_data is not None:
            if wikipedia_data.get("retrieved"):
                short = self._model_short_name()
                reading_ceiling = _READING_MAX_TOKENS.get(short, _READING_MAX_TOKENS_DEFAULT)
                wiki_token_budget = (
                    self.context_window - reading_ceiling - 250
                )

                wiki_block, wiki_truncation = self._format_wikipedia(
                    wikipedia_data,
                    tokenizer=self.tokenizer,
                    max_tokens=wiki_token_budget,
                )

                if wiki_truncation["input_truncated"]:
                    logger.info(
                        "[Wikipedia] Input truncated to fit budget: kept %d/%d sections "
                        "(dropped: %s)",
                        len(wiki_truncation["sections_kept"]),
                        wiki_truncation["sections_total"],
                        wiki_truncation["sections_dropped"],
                    )

                if len(wiki_block) > 50:
                    relevance, status = self._read_evidence(
                        scenario_for_reading, wiki_block, "Wikipedia",
                    )
                    reading_status["wiki"] = status
                    if status == "ok":
                        reading_outputs["wiki"] = relevance   # save reading-turn text
                        block = (
                            "[Wikipedia -- relevant points]\n"
                            f"{relevance}"
                        )
                        tools_used.append("wikipedia_rag")
                    else:
                        block = (
                            "[Wikipedia -- relevant points]\n"
                            f"  Reading-turn output was {status}; treat as unavailable."
                        )
                        # Same as atlas: keep partial output on failure for diagnosis
                        if relevance:
                            reading_outputs["wiki"] = relevance
                else:
                    reading_status["wiki"] = "empty"
                    block = (
                        "[Wikipedia]\n"
                        "  Not available for this country."
                    )
            else:
                reading_status["wiki"] = "unavailable"
                block = (
                    "[Wikipedia]\n"
                    "  Not available for this country."
                )
                wiki_truncation = {
                    "input_truncated":  False,
                    "sections_total":   0,
                    "sections_kept":    [],
                    "sections_dropped": [],
                }
            block_tokens = len(self.tokenizer.encode(block))
            if tokens_used + block_tokens < evidence_token_budget:
                evidence_parts.append(block)
                tokens_used += block_tokens + 5
        else:
            wiki_truncation = {
                "input_truncated":  False,
                "sections_total":   0,
                "sections_kept":    [],
                "sections_dropped": [],
            }

        evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "No evidence available."

        #    Assemble final prompt
        user_message = skeleton_user.replace(_EVIDENCE_PLACEHOLDER, evidence_text)

        #    Adaptive max_new_tokens
        final_messages = [
            {"role": "system", "content": self.synthesis_system_prompt},
            {"role": "user",   "content": user_message},
        ]
        final_formatted = self.tokenizer.apply_chat_template(
            final_messages, tokenize=False, add_generation_prompt=True
        )
        actual_input_tokens = len(self.tokenizer.encode(final_formatted))
        adaptive_max = self.context_window - actual_input_tokens - 50  # 50 safety margin
        # Floor: at least 512 tokens -- enough for Answer + short Explanation
        # Ceiling: never exceed self.max_new_tokens (caller's intended cap)
        adaptive_max = max(512, min(adaptive_max, self.max_new_tokens))

        logger.info(
            "[Adaptive] input=%d tokens, generation=%d tokens (was fixed %d)",
            actual_input_tokens, adaptive_max, self.max_new_tokens,
        )

        raw = self.generate(
            user_message=user_message,
            system_message=self.synthesis_system_prompt,
            max_new_tokens=adaptive_max,
        )

        #    Classify output quality
        stripped = _RE_THINK.sub("", raw, count=0).strip()
        output_status = _classify_output(stripped)

        if output_status in ("empty", "degenerated"):
            self.decisions_made += 1
            return {
                "decision":         output_status,
                "reasoning":        "",
                "raw_response":     raw,
                "answer_changed":   True,  # different from baseline (it's a failure label)
                "tools_used":       tools_used,
                "output_status":    output_status,
                "reading_status":   reading_status,
                "reading_outputs":  reading_outputs,
                "wiki_truncation":  wiki_truncation,
            }

        #    Parse
        parsed         = self._parse_json_response(stripped)
        decision       = parsed["decision"]
        reasoning      = parsed["reasoning"]
        answer_changed = decision != baseline_answer.lower().strip()

        self.decisions_made += 1

        # Classify: parsed decision of "unknown" means coherent output but
        # no conclusion found -- different from degenerated/empty
        if decision == "unknown":
            final_status = "unknown"
        else:
            final_status = "ok"

        return {
            "decision":         decision,
            "reasoning":        reasoning,
            "raw_response":     raw,
            "answer_changed":   answer_changed,
            "tools_used":       tools_used,
            "output_status":    final_status,
            "reading_status":   reading_status,
            "reading_outputs":  reading_outputs,
            "wiki_truncation":  wiki_truncation,
        }

    # ------------------------------------------------------------------
    # Evidence formatters (tool results -> clean text for model)
    # ------------------------------------------------------------------
    @staticmethod
    def _format_hofstede(data: Dict[str, Any]) -> str:
        scores = data.get("scores", {})
        lines = [
            "[Hofstede Cultural Dimensions]",
            "Scores are 0-100 on Hofstede's six cultural dimensions.",
        ]
        for dim, score in scores.items():
            label = dim.replace("_", " ").title()
            lines.append(f"  {label}: {score}/100")
        return "\n".join(lines)

    # Axes to skip entirely -- not useful for cultural norm judgment
    _ATLAS_SKIP = frozenset({
        "References", "Population Statistics", "Dates of Significance", "Naming",
    })

    # Priority order -- most relevant axes for cultural norm assessment first
    _ATLAS_PRIORITY = [
        "Core Concepts",          # core values and beliefs
        "Etiquette",              # social norms -- directly relevant
        "Do's and Don'ts",        # explicit behavioral rules
        "Communication",          # interaction styles
        "Family",                 # family structure and expectations
        "Greetings",              # social customs
        "Religion",               # religious context
        "Business Culture",       # workplace norms
        "Other Considerations",   # catch-all, sometimes useful
    ]

    @staticmethod
    def _format_cultural_atlas(data: Dict[str, Any], max_chars: int = _ATLAS_TOTAL_CAP) -> str:
        """
        Format Cultural Atlas data for the synthesis prompt.

        Args:
            data:      Tool result dict with 'axes' key.
            max_chars: Budget from dynamic evidence allocation.
                       Falls back to _ATLAS_TOTAL_CAP if not specified.

        Per-axis and per-field caps still apply within the total budget:
          - _ATLAS_AXIS_CAP  (3000): prevents one huge axis eating everything
          - _ATLAS_FIELD_CAP (4000): truncates verbose individual values
          - _ATLAS_LIST_CAP  (15):   limits list items

        Axes are processed in priority order (Core Concepts -> Etiquette -> ...)
        and irrelevant ones (References, Population Statistics, etc.) are skipped.
        """
        axes = data.get("axes", {})
        if not axes:
            return "[Cultural Atlas]\n  No data available"

        # Build axis names in priority order, then any remaining
        ordered = []
        seen = set()
        for name in Coordinator._ATLAS_PRIORITY:
            if name in axes:
                ordered.append(name)
                seen.add(name)
        for name in axes:
            if name not in seen and name not in Coordinator._ATLAS_SKIP and "in Australia" not in name:
                ordered.append(name)

        lines = ["[Cultural Atlas]"]
        total_chars = len(lines[0])

        for axis_name in ordered:
            axis_data = axes[axis_name]
            content = axis_data.get("content", {})

            axis_lines = []
            axis_chars = 0

            if isinstance(content, dict):
                for k, v in content.items():
                    if axis_chars >= _ATLAS_AXIS_CAP:
                        break
                    if isinstance(v, str) and v:
                        v_text = v[:_ATLAS_FIELD_CAP] + "..." if len(v) > _ATLAS_FIELD_CAP else v
                        line = f"    {k}: {v_text}"
                    elif isinstance(v, list) and v:
                        line = f"    {k}: {'; '.join(str(i) for i in v[:_ATLAS_LIST_CAP])}"
                    else:
                        continue
                    axis_lines.append(line)
                    axis_chars += len(line) + 1

                if axis_lines:
                    axis_lines.insert(0, f"  {axis_name}:")

            elif isinstance(content, str) and content:
                v_text = content[:_ATLAS_FIELD_CAP] + "..." if len(content) > _ATLAS_FIELD_CAP else content
                axis_lines.append(f"  {axis_name}: {v_text}")

            if not axis_lines:
                continue

            block = "\n".join(axis_lines)
            if total_chars + len(block) + 1 > max_chars:
                break

            lines.append(block)
            total_chars += len(block) + 1

        return "\n".join(lines) if len(lines) > 1 else "[Cultural Atlas]\n  No data available"

    @staticmethod
    def _format_wikipedia(
        data: Dict[str, Any],
        tokenizer=None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Format Wikipedia tool result -- feed ALL sections in natural order.

        Args:
            data:       Tool result dict with 'sections_raw' key.
            tokenizer:  HuggingFace tokenizer (required if max_tokens set).
            max_tokens: Optional token budget for the formatted block.
                        If set and the full text exceeds it, truncation
                        happens at the last complete [SECTION:] boundary
                        -- so the model never sees a half-cut section.
                        If None: no truncation, return everything.

        Returns:
            (formatted_text, truncation_info) where truncation_info has:
              - "input_truncated": bool
              - "sections_total":  int -- sections in the source data
              - "sections_kept":   List[str] -- names included in formatted text
              - "sections_dropped":List[str] -- names cut by truncation

        """
        empty_info = {
            "input_truncated":  False,
            "sections_total":   0,
            "sections_kept":    [],
            "sections_dropped": [],
        }

        sections_raw = data.get("sections_raw", {})
        if sections_raw:
            section_names = list(sections_raw.keys())
            parts = ["[Wikipedia]"]
            for name in section_names:
                parts.append(f"[SECTION: {name}]\n{sections_raw[name]}")
            full_text = "\n\n".join(parts)

            info = {
                "input_truncated":  False,
                "sections_total":   len(section_names),
                "sections_kept":    list(section_names),
                "sections_dropped": [],
            }

            # If no budget specified, return as-is
            if tokenizer is None or max_tokens is None:
                return full_text, info

            # Check budget
            tokens = tokenizer.encode(full_text)
            if len(tokens) <= max_tokens:
                return full_text, info

            # Over budget -- truncate at last complete section boundary.
            # Decode 20 tokens shy of the cap to leave room for the trailing
            # "[...remaining sections truncated]" marker after boundary trim.
            truncated_text = tokenizer.decode(tokens[:max(max_tokens - 20, 0)])
            last_section_marker = truncated_text.rfind("\n\n[SECTION:")
            if last_section_marker > 0:
                truncated_text = truncated_text[:last_section_marker]

            # Identify which sections survived. The leading "[Wikipedia]\n\n"
            # plus first "[SECTION: X]\n..." chunks are present; any name
            # not found in truncated_text was dropped.
            sections_kept = re.findall(r"\[SECTION: ([^\]]+)\]", truncated_text)
            sections_dropped = [n for n in section_names if n not in sections_kept]

            final_text = truncated_text + "\n\n[...remaining sections truncated]"

            return final_text, {
                "input_truncated":  True,
                "sections_total":   len(section_names),
                "sections_kept":    sections_kept,
                "sections_dropped": sections_dropped,
            }

        # Legacy fallbacks -- pre-sections_raw format
        results = data.get("results", [])
        if results:
            return "[Wikipedia]\n" + "\n".join(results), empty_info

        content = data.get("content", "") or data.get("summary", "")
        if content:
            return f"[Wikipedia]\n  {content}", empty_info

        return "[Wikipedia]\n  No data available", empty_info

    # ------------------------------------------------------------------
    # Response parser (multi-format with priority chain)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json_response(stripped: str) -> dict:
        """
        Parse model response. Function name is historical -- it now handles
        multiple formats with the following priority:

          1. Bare answer ("Yes" / "No." / "Neither" alone)
          2. Plain-text Answer:/Explanation: format  (PRIMARY)
          3. JSON {"Answer": ..., "Explanation": ...}      
          4. JSON {"conclusion": ..., "reasoning": ...}   
          5. Regex REASONING:/CONCLUSION: format          
        """
        import json as _json

        # ---- 1. Bare-answer fallback -------------------------------------
        # Rescues two patterns:
        #   (a) the WHOLE output is just "Yes" / "No." / "Neither"
        #   (b) output STARTS with a bare answer word, then Explanation:
        bare = stripped.lower().strip().rstrip('.,!? "\'')
        if bare in ("yes", "no", "neither", "neutral"):
            return {
                "decision": _canonicalise(bare),
                "reasoning": "(bare answer; no explanation emitted)",
            }
        # Pattern (b): leading bare answer word at start of line.
        m_bare = re.match(
            r"^\s*(yes|no|neither|neutral)\b\s*[.,!?]?\s*\n",
            stripped,
            re.IGNORECASE,
        )
        if m_bare:
            decision = _canonicalise(m_bare.group(1))
            # Try to grab Explanation: text that follows
            m_exp = _RE_EXPLANATION_TO_END.search(stripped)
            explanation = (
                m_exp.group(1).strip().rstrip('"\'} \n\r\t')
                if m_exp else ""
            )
            return {"decision": decision, "reasoning": explanation}

        # ---- 2. Plain-text Answer:/Explanation: (PRIMARY format) ---------
        # Try word form first ("Answer: Yes"), then number form ("Answer: 1").
        m_ans = _RE_ANSWER.search(stripped)
        m_num = None if m_ans else _RE_ANSWER_NUMBER.search(stripped)
        if m_ans or m_num:
            if m_ans:
                decision = _canonicalise(m_ans.group(1))
            else:
                decision = _canonicalise(_OPTION_NUMBER_TO_WORD[m_num.group(1)])
            if decision in _VALID_ANSWERS:
                m_exp = _RE_EXPLANATION_TO_END.search(stripped)
                explanation = (
                    m_exp.group(1).strip().rstrip('"\'} \n\r\t')
                    if m_exp else ""
                )
                # If no Explanation: header found, take prose BEFORE Answer:
                if not explanation:
                    parts = _RE_SPLIT_ANSWER.split(stripped, maxsplit=1)
                    if parts and len(parts[0].strip()) > 30:
                        explanation = parts[0].strip()
                return {"decision": decision, "reasoning": explanation}

        # ---- 3-4. JSON (legacy) ------------------------------------------
        text = stripped

        # Strip markdown code fences if present
        text = re.sub(r"^\s*```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()

        # Try clean JSON parse -- accept either the new {"Answer", "Explanation"}
        # (older patched runs) or the original {"conclusion", "reasoning"}
        try:
            obj = _json.loads(text)
            ans_field = obj.get("Answer", obj.get("answer", obj.get("conclusion")))
            exp_field = obj.get("Explanation", obj.get("explanation", obj.get("reasoning")))
            conclusion = _canonicalise(str(ans_field) if ans_field is not None else "unknown")
            reasoning  = str(exp_field) if exp_field is not None else ""
            if conclusion in _VALID_ANSWERS:
                return {"decision": conclusion, "reasoning": reasoning}
        except (_json.JSONDecodeError, ValueError, AttributeError):
            pass

        # Partial/truncated JSON -- conclusion at the start, reasoning may be cut off
        m = re.search(
            r'"(?:conclusion|answer)"\s*:\s*"(yes|no|neutral|neither)"',
            text, re.I,
        )
        if m:
            conclusion = _canonicalise(m.group(1))
            r = re.search(
                r'"(?:reasoning|explanation)"\s*:\s*"(.+)',
                text, re.I | re.DOTALL,
            )
            reasoning = r.group(1).rstrip('"} \n\r\t') if r else ""
            return {"decision": conclusion, "reasoning": reasoning}

        # ---- 5. Final fallback -- old REASONING:/CONCLUSION: regex -------
        decision = Coordinator._parse_decision_stripped(stripped)
        reasoning = Coordinator._parse_reasoning_stripped(stripped)
        return {"decision": decision, "reasoning": reasoning}

    # ------------------------------------------------------------------
    # Parsers -- work on already-stripped text (think blocks removed by caller)
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_think_blocks(response: str) -> str:
        """
        Remove <think>...</think> blocks from raw model output.
        """
        return _RE_THINK.sub("", response, count=0).strip()

    def _parse_decision(self, response: str) -> str:
        """Parse yes/no/neutral -- strips think blocks first, then delegates."""
        return self._parse_decision_stripped(_RE_THINK.sub("", response, count=0).strip())

    def _parse_reasoning(self, response: str) -> str:
        """Extract reasoning text -- strips think blocks first, then delegates."""
        return self._parse_reasoning_stripped(_RE_THINK.sub("", response, count=0).strip())

    @staticmethod
    def _parse_decision_stripped(stripped: str) -> str:
        """
        Parse yes/no/neutral from an already-stripped (no think blocks) response.

        Tries in order:
        1. Answer: <yes/no/neither/neutral>     (current format)
        2. CONCLUSION: ...                       (legacy / fallback)
        3. DECISION:   ...                       (legacy / fallback)
        4. "unknown"   if none matched
        """
        m = _RE_ANSWER.search(stripped)
        if m:
            return _canonicalise(m.group(1))
        m = _RE_ANSWER_NUMBER.search(stripped)
        if m:
            return _canonicalise(_OPTION_NUMBER_TO_WORD[m.group(1)])
        m = _RE_CONCLUDE.search(stripped)
        if m:
            return _canonicalise(m.group(1))
        m = _RE_DECIDE.search(stripped)
        if m:
            return _canonicalise(m.group(1))
        return "unknown"

    @staticmethod
    def _parse_reasoning_stripped(stripped: str) -> str:
        """
        Extract reasoning/explanation text from an already-stripped response.

        Tries in order:
        1. Text after "Explanation:"                         (current format)
        2. Text BEFORE "Answer:" (prose before the answer label)
        3. Text between REASONING: and CONCLUSION:/DECISION: (legacy)
        4. Text after REASONING:                              (legacy)
        5. Everything before CONCLUSION:/DECISION:            (legacy)
        6. First substantial paragraph
        7. Truncated response
        """
        # 1. Explanation: block (primary)
        m = _RE_EXPLANATION_TO_END.search(stripped)
        if m and len(m.group(1).strip()) > 30:
            return m.group(1).strip().rstrip('"\'} \n\r\t')

        # 2. Prose before Answer: (model put reasoning first, then Answer)
        parts = _RE_SPLIT_ANSWER.split(stripped, maxsplit=1)
        if len(parts) > 1 and len(parts[0].strip()) > 30:
            return parts[0].strip()

        # 3-5. Legacy paths
        m = _RE_REASONING_WITH_END.search(stripped)
        if m and len(m.group(1).strip()) > 30:
            return m.group(1).strip()

        m = _RE_REASONING_TO_END.search(stripped)
        if m and len(m.group(1).strip()) > 30:
            return m.group(1).strip()

        parts = _RE_SPLIT_CONCLUDE.split(stripped, maxsplit=1)
        if len(parts) > 1 and len(parts[0].strip()) > 30:
            return parts[0].strip()

        # 6-7. Last-ditch
        paragraphs = [p.strip() for p in stripped.split("\n\n") if len(p.strip()) > 30]
        if paragraphs:
            return paragraphs[0]

        return stripped[:200] + "..." if len(stripped) > 200 else stripped

    # ------------------------------------------------------------------
    # Comparison helper
    # ------------------------------------------------------------------
    @staticmethod
    def is_correct(decision: str, gold_answer: str) -> bool:
        """Case-insensitive comparison of model decision to gold answer.
        """
        d = _canonicalise(decision)
        g = _canonicalise(gold_answer)
        if d not in _VALID_ANSWERS:
            return False
        return d == g

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        stats = super().get_statistics()
        stats.update({
            "decisions_made": self.decisions_made,
        })
        return stats