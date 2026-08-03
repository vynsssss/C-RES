# -*- coding: utf-8 -*-
"""
Coordinator Agentic Extension

Adds two agentic tool-selection modes to the base Coordinator:

Mode 'free':
    Turn 1 — LLM selects 0-3 tools it thinks will help
    Turn 2 — Selected tools are called (external)
    Turn 3 — LLM synthesizes, explicitly deciding whether evidence is useful

Mode 'single':
    Turn 1 — LLM selects exactly 1 tool
    Turn 2 — That tool is called (external)
    Turn 3 — LLM synthesizes, explicitly deciding whether evidence is useful

Both modes return:
    {
        "selected_tools":       ["hofstede_tool"],
        "tools_retrieved":      ["hofstede_tool"],
        "evidence_used":        True,
        "decision":             "yes",
        "reasoning":            "...",
        "raw_tool_selection":   "...",
        "raw_response":         "...",
        "answer_changed":       False,
    }

Usage:
    from coordinator_agentic import AgenticCoordinator

    coord = AgenticCoordinator(model, tokenizer, device="cuda", max_new_tokens=2048)

    selection = coord.select_tools(
        prompt_type="story_country",
        story=..., country=..., value=..., rot=...,
        mode="free",
    )

    result = coord.synthesize_agentic(
        mode="free",
        prompt_type=..., story=..., country=...,
        baseline_reasoning=..., baseline_answer=...,
        hofstede_data=..., cultural_atlas_data=..., wikipedia_data=...,
        value=..., rot=...,
    )
"""

import re
import logging
from typing import Dict, Any, List, Optional

from coordinator import (
    Coordinator,
    _RE_THINK,
    _classify_output,
    _canonicalise,
    _VALID_ANSWERS,
    _ATLAS_TOTAL_CAP,
    _READING_MAX_TOKENS,
    _READING_MAX_TOKENS_DEFAULT,
)

logger = logging.getLogger(__name__)

# ── Tool metadata shown to the model ──────────────────────────────────────────
TOOL_REGISTRY_DESCRIPTION = """\
Available tools (choose by exact name):
  - hofstede_tool       : Quantitative cultural dimension scores (0-100) — \
power distance, individualism, masculinity, uncertainty avoidance, \
long-term orientation, indulgence. Best for questions about cultural values \
and behavioural tendencies.
  - cultural_atlas_tool : Detailed cultural practices, etiquette, \
communication styles, and social norms by country. Best for specific \
behavioural rules and social expectations.
  - wikipedia_rag       : General cultural background, history, religion, \
and social context retrieved from Wikipedia. Best for broad cultural context.\
"""

VALID_TOOL_NAMES = frozenset({"hofstede_tool", "cultural_atlas_tool", "wikipedia_rag"})

_RE_SELECTED_TOOLS     = re.compile(r"selected_tools:\s*(.+)",             re.IGNORECASE)
_RE_REASONING_BEFORE_T = re.compile(
    r"reasoning:\s*(.+?)(?=\s*selected_tools\s*:)", re.IGNORECASE | re.DOTALL
)
_RE_SPLIT_SELECTED     = re.compile(r"selected_tools\s*:",                  re.IGNORECASE)
_RE_EVIDENCE_USED      = re.compile(r"evidence_used:\s*(yes|no)",           re.IGNORECASE)

_RE_ANSWER_AGENTIC = re.compile(
    r"answer\s*:\s*\*{0,2}\s*(yes|no|neither|neutral)\b",
    re.IGNORECASE,
)
_RE_EVIDENCE_AGENTIC = re.compile(
    r"evidence\s*:\s*\*{0,2}\s*(yes|no)\b",
    re.IGNORECASE,
)
_RE_EXPLANATION_AGENTIC = re.compile(
    r"explanation\s*:\s*(.+?)(?:\Z|\n\s*(?:answer|evidence)\s*:)",
    re.IGNORECASE | re.DOTALL,
)


class AgenticCoordinator(Coordinator):
    """
    Extends Coordinator with two-turn agentic tool-selection modes.

    Inherits all of Coordinator's generation, parsing, and synthesis logic.
    Only adds:
        select_tools()       — Turn 1: LLM picks tools
        synthesize_agentic() — Turn 3: LLM synthesizes + decides evidence usage
    """

    # ------------------------------------------------------------------
    # System prompts
    # ------------------------------------------------------------------
    def _tool_selection_system_prompt(self, mode: str) -> str:
        if mode == "single":
            constraint = (
                "You must select EXACTLY ONE tool — the single most useful one for this scenario.\n"
                "Do NOT select zero tools and do NOT select more than one."
            )
            format_instruction = "SELECTED_TOOLS: <exactly one tool name>"
        else:  # free
            constraint = (
                "Select between 0 and 3 tools. Only select tools that will genuinely help.\n"
                "If no tool is needed, write 'none'."
            )
            format_instruction = "SELECTED_TOOLS: <comma-separated tool names, or 'none'>"

        return (
            "You are about to analyze a cultural scenario.\n"
            "Before answering, decide which research tools (if any) would help.\n\n"
            f"{TOOL_REGISTRY_DESCRIPTION}\n\n"
            f"{constraint}\n\n"
            f"Format your response as:\n"
            f"REASONING: <brief explanation of why you chose these tools>\n"
            f"{format_instruction}"
        )

    @property
    def _agentic_synthesis_system_prompt(self) -> str:
        """Agentic Step 2 synthesis system prompt.

        Uses "Neither" in the schema (matching Step 1 / paper);
        _canonicalise() maps to "neutral" downstream.
        """
        return (
            "You previously analyzed a cultural scenario and selected research tools "
            "to help. The tools have retrieved cultural evidence below.\n\n"
            "Do not make any extra inferences about actions outside of the given "
            "context and the provided cultural evidence. Only align to the scenario "
            "and the evidence given.\n\n"
            "INSTRUCTIONS:\n"
            "1. Review your initial reasoning and answer.\n"
            "2. Read the evidence. Identify which specific pieces are "
            "relevant to the scenario. If none are relevant, state so.\n"
            "3. For the relevant evidence, analyze whether it supports or "
            "contradicts your initial answer.\n"
            "4. State your final answer (Yes, No, or Neither). If it differs "
            "from your initial answer, explain what specific evidence "
            "justifies the change. If it matches your initial answer, "
            "explain what supports keeping it.\n\n"
            "Respond in this exact format:\n"
            "Answer: <Yes|No|Neither>\n"
            "Evidence: <Yes|No>\n"
            "Explanation: <your analysis>"
        )

    # ------------------------------------------------------------------
    # Turn 1: Tool Selection
    # ------------------------------------------------------------------
    def select_tools(
        self,
        prompt_type: str,
        story: str,
        country: str,
        mode: str = "free",
        value: str = "",
        rot: str = "",
    ) -> Dict[str, Any]:
        """
        Turn 1: Ask LLM which tools to use for this scenario.

        Returns:
            {
                "selected_tools":      ["hofstede_tool", ...],
                "selection_reasoning": "...",
                "raw_response":        "...",
            }
        """
        scenario = self._build_scenario(prompt_type, story, country, value, rot)

        task = (
            "Select EXACTLY ONE tool from the list above that will be most "
            "useful for answering this scenario."
            if mode == "single" else
            "Select the tools (0 to 3) that will be most useful for "
            "answering this scenario. Only select tools that genuinely help."
        )

        user_message = f"SCENARIO:\n{scenario}\n\nTASK: {task}"

        raw = self.generate(
            user_message=user_message,
            system_message=self._tool_selection_system_prompt(mode),
        )

        # Strip think blocks once — reuse for both parsers
        stripped = _RE_THINK.sub("", raw, count=0).strip()

        selected_tools      = self._parse_selected_tools_stripped(stripped, mode)
        selection_reasoning = self._parse_selection_reasoning_stripped(stripped)

        return {
            "selected_tools":      selected_tools,
            "selection_reasoning": selection_reasoning,
            "raw_response":        raw,
        }

    # ------------------------------------------------------------------
    # Turn 3: Agentic Synthesis
    # ------------------------------------------------------------------
    def synthesize_agentic(
        self,
        mode: str,
        prompt_type: str,
        story: str,
        country: str,
        baseline_reasoning: str,
        baseline_answer: str,
        selected_tools: List[str],
        hofstede_data: Optional[Dict[str, Any]] = None,
        cultural_atlas_data: Optional[Dict[str, Any]] = None,
        wikipedia_data: Optional[Dict[str, Any]] = None,
        value: str = "",
        rot: str = "",
    ) -> Dict[str, Any]:
        """
        Turn 3: Synthesize with evidence, LLM decides whether to use it.

        Uses the same dynamic evidence budgeting as standard synthesize():
          1. Build skeleton with placeholder
          2. Measure skeleton tokens with actual tokenizer
          3. Fill evidence greedily (Hofstede → Atlas → Wiki)
          4. Verify each block fits with tokenizer.encode()
        """
        # ── Build tool summary (known before evidence) ────────────────
        if selected_tools:
            tools_label = ", ".join(selected_tools)
            # retrieved_label filled after we know which tools returned data
            tool_summary_template = f"Tools you selected: {tools_label}\nTools that returned data: {{RETRIEVED}}"
        else:
            tool_summary_template = "You selected no tools."

        # ── Build skeleton with placeholder ───────────────────────────
        scenario = self._build_scenario(prompt_type, story, country, value, rot)

        _EVIDENCE_PLACEHOLDER = "{{EVIDENCE}}"
        skeleton_user = (
            f"ORIGINAL SCENARIO:\n{scenario}\n\n"
            f"YOUR INITIAL ANALYSIS:\n"
            f"{baseline_reasoning}\n"
            f"Initial answer: {baseline_answer}\n\n"
            f"TOOL RETRIEVAL SUMMARY:\n{tool_summary_template.format(RETRIEVED='pending')}\n\n"
            f"RETRIEVED EVIDENCE:\n{_EVIDENCE_PLACEHOLDER}\n\n"
            f"Now provide your final judgment."
        )

        # ── Measure skeleton tokens ───────────────────────────────────
        skeleton_messages = [
            {"role": "system", "content": self._agentic_synthesis_system_prompt},
            {"role": "user",   "content": skeleton_user},
        ]
        skeleton_formatted = self.tokenizer.apply_chat_template(
            skeleton_messages, tokenize=False, add_generation_prompt=True
        )
        skeleton_tokens = len(self.tokenizer.encode(skeleton_formatted))

        # ── Calculate evidence token budget ───────────────────────────
        available_input = self.context_window - self.max_new_tokens - 100
        evidence_token_budget = max(available_input - skeleton_tokens, 500)

        logger.info(
            "[Budget-Agentic] context=%d, skeleton=%d tokens, evidence budget=%d tokens",
            self.context_window, skeleton_tokens, evidence_token_budget,
        )

        # ── Fill evidence greedily ────────────────────────────────────
        # Reading-turn parity with standard synthesize():
        #   - Hofstede passes through RAW (small block, no reading turn).
        #   - Atlas / Wiki: format raw -> _read_evidence() -> compressed
        #     "[X -- relevant points]" block enters evidence_parts.
        #   - reading_status / reading_outputs are tracked per tool and
        #     surfaced in the return dict so trajectories store exactly
        #     the same diagnostic fields as the standard pipeline.
        tools_retrieved: List[str] = []
        evidence_parts:  List[str] = []
        reading_status:  Dict[str, str] = {}
        reading_outputs: Dict[str, str] = {}
        tokens_used = 0

        # Track Wikipedia input truncation. Mirrors synthesize() and
        # react_loop so all three pipelines surface the same diagnostic.
        wiki_truncation: Dict[str, Any] = {
            "input_truncated":  False,
            "sections_total":   0,
            "sections_kept":    [],
            "sections_dropped": [],
        }

        # Situation context for the reading turn (no Task, no Options).
        # Shared base-class helper -> identical framing to standard pipeline.
        scenario_for_reading = self._build_reading_scenario(
            prompt_type, story, country, value, rot,
        )

        # 1. Hofstede -- RAW pass-through (matches standard synthesize)
        if hofstede_data and hofstede_data.get("success"):
            block = self._format_hofstede(hofstede_data)
            block_tokens = len(self.tokenizer.encode(block))
            if tokens_used + block_tokens < evidence_token_budget:
                evidence_parts.append(block)
                tools_retrieved.append("hofstede_tool")
                reading_status["hofstede"]  = "ok"
                reading_outputs["hofstede"] = block   # what synthesis sees
                tokens_used += block_tokens + 5

        # 2. Cultural Atlas -- reading turn, then compressed block
        if cultural_atlas_data and cultural_atlas_data.get("retrieved"):
            atlas_block = self._format_cultural_atlas(
                cultural_atlas_data, max_chars=_ATLAS_TOTAL_CAP,
            )
            if len(atlas_block) > 50:
                relevance, status = self._read_evidence(
                    scenario_for_reading, atlas_block, "Cultural Atlas",
                )
                reading_status["atlas"] = status
                if status == "ok":
                    reading_outputs["atlas"] = relevance
                    block = (
                        "[Cultural Atlas -- relevant points]\n"
                        f"{relevance}"
                    )
                    block_tokens = len(self.tokenizer.encode(block))
                    if block_tokens > 30 and tokens_used + block_tokens < evidence_token_budget:
                        evidence_parts.append(block)
                        tools_retrieved.append("cultural_atlas_tool")
                        tokens_used += block_tokens + 5
                else:
                    # Reading turn failed -- surface failure to model
                    # (matches standard's diagnosis-friendly behaviour).
                    block = (
                        "[Cultural Atlas -- relevant points]\n"
                        f"  Reading-turn output was {status}; treat as unavailable."
                    )
                    if relevance:
                        reading_outputs["atlas"] = relevance
                    block_tokens = len(self.tokenizer.encode(block))
                    if tokens_used + block_tokens < evidence_token_budget:
                        evidence_parts.append(block)
                        tokens_used += block_tokens + 5
            else:
                reading_status["atlas"] = "empty"

        # 3. Wikipedia -- reading turn, then compressed block
        if wikipedia_data and wikipedia_data.get("retrieved"):
            short = self._model_short_name()
            reading_ceiling = _READING_MAX_TOKENS.get(
                short, _READING_MAX_TOKENS_DEFAULT
            )
            wiki_token_budget = (
                self.context_window - reading_ceiling - 250
            )
            wiki_block, info = self._format_wikipedia(
                wikipedia_data,
                tokenizer=self.tokenizer,
                max_tokens=wiki_token_budget,
            )
            # Capture truncation report for parity with synthesize() + react_loop.
            if info and info.get("sections_total", 0) > 0:
                wiki_truncation.update(info)
                if wiki_truncation["input_truncated"]:
                    logger.info(
                        "[Wikipedia-Agentic] Input truncated to fit budget: "
                        "kept %d/%d sections (dropped: %s)",
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
                    reading_outputs["wiki"] = relevance
                    block = (
                        "[Wikipedia -- relevant points]\n"
                        f"{relevance}"
                    )
                    block_tokens = len(self.tokenizer.encode(block))
                    if block_tokens > 30 and tokens_used + block_tokens < evidence_token_budget:
                        evidence_parts.append(block)
                        tools_retrieved.append("wikipedia_rag")
                        tokens_used += block_tokens + 5
                else:
                    block = (
                        "[Wikipedia -- relevant points]\n"
                        f"  Reading-turn output was {status}; treat as unavailable."
                    )
                    if relevance:
                        reading_outputs["wiki"] = relevance
                    block_tokens = len(self.tokenizer.encode(block))
                    if tokens_used + block_tokens < evidence_token_budget:
                        evidence_parts.append(block)
                        tokens_used += block_tokens + 5
            else:
                reading_status["wiki"] = "empty"

        evidence_text = (
            "\n\n".join(evidence_parts)
            if evidence_parts
            else "No evidence was retrieved (tools returned no data)."
        )

        # ── No evidence + single mode → preserve baseline ─────────────
        # In 'single' mode the LLM was forced to pick a tool but it returned
        # nothing (e.g. Hofstede missing for 23 NorMAD countries).  Sending
        # "No evidence was retrieved" to the model causes it to second-guess
        # a correct baseline answer → net regression, same failure mode as
        # standard synthesize().  For 'free' mode we do NOT short-circuit —
        # the model deliberately chose 0 tools or the tool failed, and the
        # EVIDENCE_USED: no tag in its response handles that case cleanly.
        if not evidence_parts and mode == "single":
            logger.info(
                "[Synthesize-Agentic] single mode, no evidence for %s "
                "— keeping baseline answer '%s'",
                country, baseline_answer,
            )
            return {
                "decision":        baseline_answer.lower().strip(),
                "reasoning":       baseline_reasoning,
                "raw_response":    "",
                "answer_changed":  False,
                "evidence_used":   False,
                "tools_retrieved": [],
                "reading_status":  reading_status,
                "reading_outputs": reading_outputs,
                "wiki_truncation": wiki_truncation,
            }

        # ── Assemble final prompt ─────────────────────────────────────
        retrieved_label = ", ".join(tools_retrieved) if tools_retrieved else "none"
        tool_summary = tool_summary_template.format(RETRIEVED=retrieved_label)

        user_message = (
            f"ORIGINAL SCENARIO:\n{scenario}\n\n"
            f"YOUR INITIAL ANALYSIS:\n"
            f"{baseline_reasoning}\n"
            f"Initial answer: {baseline_answer}\n\n"
            f"TOOL RETRIEVAL SUMMARY:\n{tool_summary}\n\n"
            f"RETRIEVED EVIDENCE:\n{evidence_text}\n\n"
            f"Now provide your final judgment."
        )

        raw = self.generate(
            user_message=user_message,
            system_message=self._agentic_synthesis_system_prompt,
        )

        # ── Classify output quality ───────────────────────────────────
        stripped = _RE_THINK.sub("", raw, count=0).strip()
        output_status = _classify_output(stripped)

        if output_status in ("empty", "degenerated"):
            self.decisions_made += 1
            return {
                "decision":        output_status,
                "reasoning":       "",
                "evidence_used":   False,
                "raw_response":    raw,
                "answer_changed":  True,
                "tools_retrieved": tools_retrieved,
                "output_status":   output_status,
                "reading_status":  reading_status,
                "reading_outputs": reading_outputs,
                "wiki_truncation": wiki_truncation,
            }

        # Strip once — parse with JSON parser (regex fallback built in)
        parsed         = self._parse_json_response_agentic(stripped)
        decision       = parsed["decision"]
        reasoning      = parsed["reasoning"]
        evidence_used  = parsed["evidence_used"]
        answer_changed = decision != baseline_answer.lower().strip()

        self.decisions_made += 1

        final_status = "unknown" if decision == "unknown" else "ok"

        return {
            "decision":        decision,
            "reasoning":       reasoning,
            "evidence_used":   evidence_used,
            "raw_response":    raw,
            "answer_changed":  answer_changed,
            "tools_retrieved": tools_retrieved,
            "output_status":   final_status,
            "reading_status":  reading_status,
            "reading_outputs": reading_outputs,
            "wiki_truncation": wiki_truncation,
        }

    # ------------------------------------------------------------------
    # Scenario builder (shared by both turns)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_scenario(
        prompt_type: str, story: str, country: str,
        value: str = "", rot: str = ""
    ) -> str:
        """Paper-aligned scenario builder (matches prompt_formatter._TASK_*)."""
        if prompt_type == "story_country_value":
            return (
                "You will be given a short story involving 2 or more characters "
                "about everyday situations. You will also be given a country of "
                "relevance, along with a rule-of-thumb. Read the story and reason "
                "if the action by the character in the story is socially acceptable "
                "in the given country with the associated rule-of-thumb.\n\n"
                f"Country: {country}\n"
                f"RoT: {value}\n"
                f"Story: {story}"
            )
        elif prompt_type == "story_rot":
            return (
                "You will be given a short story involving 2 or more characters "
                "about everyday situations. You will also be given a rule-of-thumb. "
                "Read the story and reason if the action by the character in the "
                "story is socially acceptable under the given rule-of-thumb alone.\n\n"
                f"Rule-of-thumb: {rot}\n"
                f"Story: {story}"
            )
        else:  # story_country
            return (
                "You will be given a short story involving 2 or more characters "
                "about everyday situations. You will also be given a country of "
                "relevance. Read the story and reason if the action by the "
                "character in the story is socially acceptable in the given country.\n\n"
                f"Country: {country}\n"
                f"Story: {story}"
            )

    # ------------------------------------------------------------------
    # Agentic synthesis response parser
    # ------------------------------------------------------------------
    # Priority chain (mirrors coordinator.py for consistency):
    #   1. Plain-text Answer/Evidence/Explanation  (PRIMARY)
    #   2. JSON {"conclusion", "evidence_used", "reasoning"}  (legacy fallback)
    # ------------------------------------------------------------------
    def _parse_json_response_agentic(self, stripped: str) -> dict:
        """Parse an agentic synthesis response.

        Tries plain-text Answer/Evidence/Explanation first since it's the
        prompt's primary instruction. Falls back to JSON for models that
        emit it. Final fallback is the legacy regex parser chain.

        Returns:
            {"decision": yes|no|neutral|unknown,
             "reasoning": str,
             "evidence_used": bool}
        """
        import json as _json

        # ---- 1. Plain-text Answer/Evidence/Explanation (PRIMARY) ----
        m_ans = _RE_ANSWER_AGENTIC.search(stripped)
        if m_ans:
            decision = _canonicalise(m_ans.group(1))
            if decision in _VALID_ANSWERS:
                # Evidence flag (defaults False if Evidence: line missing)
                m_ev = _RE_EVIDENCE_AGENTIC.search(stripped)
                evidence_used = m_ev is not None and m_ev.group(1).lower() == "yes"

                # Explanation -- prefer the labeled section; fall back to text
                # after the last labeled line if Explanation: not present.
                m_exp = _RE_EXPLANATION_AGENTIC.search(stripped)
                if m_exp:
                    reasoning = m_exp.group(1).strip()
                else:
                    last_label_end = max(
                        m_ans.end(),
                        m_ev.end() if m_ev else 0,
                    )
                    reasoning = stripped[last_label_end:].strip()[:2000]

                return {
                    "decision":      decision,
                    "reasoning":     reasoning,
                    "evidence_used": evidence_used,
                }

        # ---- 2. JSON (legacy / Qwen / Llama) ----
        text = stripped
        text = re.sub(r"^\s*```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()

        try:
            obj = _json.loads(text)
            conclusion = _canonicalise(str(obj.get("conclusion", "unknown")))
            evidence   = str(obj.get("evidence_used", "no")).lower().strip()
            reasoning  = str(obj.get("reasoning", ""))
            if conclusion in _VALID_ANSWERS:
                return {
                    "decision":      conclusion,
                    "reasoning":     reasoning,
                    "evidence_used": evidence in ("yes", "true", "1"),
                }
        except (_json.JSONDecodeError, ValueError, AttributeError):
            pass

        # Partial/truncated JSON
        m = re.search(r'"conclusion"\s*:\s*"(yes|no|neutral|neither)"', text, re.I)
        if m:
            conclusion = _canonicalise(m.group(1))
            if conclusion in _VALID_ANSWERS:
                ev = re.search(r'"evidence_used"\s*:\s*"(yes|no)"', text, re.I)
                evidence_used = ev.group(1).lower() == "yes" if ev else False
                r = re.search(r'"reasoning"\s*:\s*"(.+)', text, re.I | re.DOTALL)
                reasoning = r.group(1).rstrip('"} \n\r\t') if r else ""
                return {
                    "decision":      conclusion,
                    "reasoning":     reasoning,
                    "evidence_used": evidence_used,
                }

        # ---- 3. Final fallback -- legacy regex parsers ----
        decision      = self._parse_decision_stripped(stripped)
        reasoning     = self._parse_reasoning_stripped(stripped)
        evidence_used = self._parse_evidence_used_stripped(stripped)
        return {
            "decision":      decision,
            "reasoning":     reasoning,
            "evidence_used": evidence_used,
        }

    # ------------------------------------------------------------------
    # Parsers — work on already-stripped text (think blocks removed by caller)
    # ------------------------------------------------------------------
    def _parse_selected_tools_stripped(self, stripped: str, mode: str) -> List[str]:
        """
        Parse SELECTED_TOOLS: from an already-stripped tool selection response.
        Uses precompiled _RE_SELECTED_TOOLS — no regex recompilation per call.
        """
        m = _RE_SELECTED_TOOLS.search(stripped)
        if not m:
            logger.warning("No SELECTED_TOOLS found in response")
            return self._fallback_tool_parse(stripped, mode)

        raw_value = m.group(1).strip().lower()

        if raw_value in ("none", "none.", "-", ""):
            return []

        candidates = [t.strip().rstrip(".,;") for t in raw_value.split(",")]
        valid = [t for t in candidates if t in VALID_TOOL_NAMES]

        if mode == "single":
            return (valid or self._fallback_tool_parse(stripped, mode))[:1]
        return valid[:3]

    def _parse_selected_tools(self, response: str, mode: str) -> List[str]:
        """Public variant that strips think blocks first — for external callers."""
        return self._parse_selected_tools_stripped(
            _RE_THINK.sub("", response, count=0).strip(), mode
        )

    @staticmethod
    def _fallback_tool_parse(stripped: str, mode: str) -> List[str]:
        """Scan stripped text for any valid tool name mentions."""
        stripped_lower = stripped.lower()
        found = [t for t in VALID_TOOL_NAMES if t in stripped_lower]
        return found[:1] if mode == "single" else found[:3]

    @staticmethod
    def _parse_selection_reasoning_stripped(stripped: str) -> str:
        """
        Extract REASONING from an already-stripped tool selection response.
        Uses precompiled patterns.
        """
        m = _RE_REASONING_BEFORE_T.search(stripped)
        if m and len(m.group(1).strip()) > 10:
            return m.group(1).strip()
        parts = _RE_SPLIT_SELECTED.split(stripped, maxsplit=1)
        if len(parts) > 1 and len(parts[0].strip()) > 10:
            return parts[0].strip()
        return stripped[:200]

    @staticmethod
    def _parse_evidence_used_stripped(stripped: str) -> bool:
        """
        Parse EVIDENCE_USED: yes/no from an already-stripped synthesis response.
        Uses precompiled _RE_EVIDENCE_USED.
        Defaults to False if tag is missing — missing tag must not inflate counts.
        """
        m = _RE_EVIDENCE_USED.search(stripped)
        if m:
            return m.group(1).lower() == "yes"
        logger.warning("EVIDENCE_USED tag missing in synthesis response — defaulting to False")
        return False

    def _parse_evidence_used(self, response: str) -> bool:
        """Public variant that strips think blocks first — for external callers."""
        return self._parse_evidence_used_stripped(
            _RE_THINK.sub("", response, count=0).strip()
        )