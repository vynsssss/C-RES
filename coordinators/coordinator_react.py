# -*- coding: utf-8 -*-
"""
Coordinator ReAct -- Iterative Reasoning + Acting

Implements the ReAct pattern (Yao et al., 2022) for cultural reasoning:
  The model interleaves reasoning (Thought) and tool use (Action)
  in a dynamic loop, seeing each observation before deciding the next step.

Flow:
  Thought 1 -> Action 1 -> Observation 1
  Thought 2 -> Action 2 -> Observation 2  (or finish)
  Thought 3 -> Action 3 -> Observation 3  (or finish)
  ... up to max_iterations

Key difference from AgenticCoordinator:
  - Agentic: selects all tools upfront in one shot, retrieves all, synthesizes once
  - ReAct: sees each tool's output before deciding the next action

Returns same output format as other coordinators for compatibility.

Usage:
    from coordinator_react import ReactCoordinator

    coord = ReactCoordinator(model, tokenizer, device="cuda", max_new_tokens=2048)

    result = coord.react_loop(
        prompt_type=..., story=..., country=...,
        baseline_reasoning=..., baseline_answer=...,
        tool_caller=...,  # function that calls tools
        value=..., rot=...,
    )
"""

import json as _json
import logging
import re
from typing import Dict, Any, List, Optional, Callable

from coordinator import (
    Coordinator,
    _RE_THINK,
    _classify_output,
    _READING_MAX_TOKENS,
    _READING_MAX_TOKENS_DEFAULT,
)

logger = logging.getLogger(__name__)

# Max ReAct iterations (1 tool call per iteration, 3 tools available)
MAX_ITERATIONS = 3

# Tool descriptions shown to the model
REACT_TOOL_DESC = """\
You have access to the following cultural research tools:

1. hofstede_tool -- Returns Hofstede's 6 cultural dimension scores (0-100) for a country. \
Covers: Power Distance, Individualism, Masculinity, Uncertainty Avoidance, \
Long-Term Orientation, Indulgence.

2. cultural_atlas_tool -- Returns qualitative cultural etiquette and norms from the \
Cultural Atlas database. Covers: communication styles, business culture, social norms, \
greetings, dining etiquette, and more.

3. wikipedia_rag -- Returns relevant Wikipedia sections about the country's culture, \
customs, and social practices. Broad coverage but may include irrelevant information."""


class ReactCoordinator(Coordinator):
    """
    Extends Coordinator with ReAct-style iterative reasoning.
    """

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------
    @property
    def _react_system_prompt(self) -> str:
        """System prompt for the ReAct iteration loop.

        Used during Thought->Action->Observation iterations. Tells the
        model to PLAN each tool call -- not to make a final judgment yet.

        Final-answer judgment (quoting evidence, "no relevant evidence"
        phrase, support/contradict analysis) lives in
        _react_final_system_prompt and only fires after Action: finish.

        """
        return (
            "You previously analyzed a cultural scenario and gave an "
            "initial answer with reasoning. You can now use research "
            "tools iteratively to gather evidence before finalizing "
            "your answer.\n\n"
            "Do not make any extra inferences about actions outside of "
            "the given context and the tool observations you collect.\n\n"
            f"{REACT_TOOL_DESC}\n\n"
            "FORMAT -- you must follow this exact format on each turn:\n\n"
            "Thought: <what information do I still need? which tool will "
            "give it to me?>\n"
            "Action: <tool_name> OR finish\n\n"
            "If you choose a tool, you will receive an Observation with "
            "the tool's output. Then you continue with another "
            "Thought/Action.\n\n"
            "When you have enough information (or no remaining tool will "
            "help), use:\n"
            "Action: finish\n\n"
            "Each Thought should plan the NEXT step -- do not state a "
            "final yes/no answer during iterations. Save that for after "
            "Action: finish.\n\n"
            "RULES:\n"
            "- Each tool can only be called ONCE -- no repeated calls\n"
            "- You do not have to use all tools -- stop when you have "
            "enough information\n"
            "- A maximum of 3 iterations is allowed"
        )

    @property
    def _react_final_system_prompt(self) -> str:
        """System prompt for the final-answer step (after Action: finish).
        """
        return (
            "You previously analyzed a cultural scenario, gave an initial "
            "answer with reasoning, and have now collected tool "
            "observations through ReAct iterations.\n\n"
            "Do not make any extra inferences about actions outside of "
            "the given context and the observations you collected. "
            "Only align to the scenario and the observations given.\n\n"
            "INSTRUCTIONS:\n"
            "1. Review your initial reasoning, your initial answer, and "
            "all observations gathered.\n"
            "2. Identify which specific pieces of the observations are "
            "relevant to the scenario, quoting them in your reasoning. "
            "If no observation evidence is relevant, state exactly: "
            "\"No relevant evidence found for this situation.\"\n"
            "3. For the relevant evidence, analyze whether it supports "
            "or contradicts your initial answer.\n"
            "4. State your final answer (Yes, No, or Neither). If it "
            "differs from your initial answer, explain what specific "
            "evidence justifies the change. If it matches your initial "
            "answer, explain what supports keeping it.\n\n"
            "Respond with JSON only, no other text:\n"
            '{"conclusion": "yes" or "no" or "neither", '
            '"reasoning": "your analysis citing specific evidence"}'
        )

    # ------------------------------------------------------------------
    # Scenario builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_scenario(
        prompt_type: str, story: str, country: str,
        value: str = "", rot: str = "",
    ) -> str:
        """Paper-verbatim scenario builder.

        Mirrors prompt_formatter._TASK_* and the Task lines used in
        synthesize() Turn 2. Keeps wording identical across:
          - Step 1 baseline (prompt_formatter)
          - Step 2 multi-turn synthesize (Turn 2)
          - Step 2 ReAct (here)
        so the model sees the same task framing across all evaluation modes.
        """
        if prompt_type == "story_country_value":
            return (
                "Task: You will be given a short story involving 2 or more "
                "characters about everyday situations. You will also be "
                "given a country of relevance, along with a rule-of-thumb. "
                "Read the story and reason if the action by the character "
                "in the story is socially acceptable in the given country "
                "with the associated rule-of-thumb. The answer options are "
                "Yes, No or Neither.\n\n"
                f"Country: {country}\n"
                f"RoT: {value}\n"
                f"Story: {story}"
            )
        elif prompt_type == "story_rot":
            return (
                "Task: You will be given a short story involving 2 or more "
                "characters about everyday situations. You will also be "
                "given a rule-of-thumb. Read the story and reason if the "
                "action by the character in the story is socially "
                "acceptable under the given rule-of-thumb alone. The "
                "answer options are Yes, No or Neither.\n\n"
                f"Rule-of-thumb: {rot}\n"
                f"Story: {story}"
            )
        else:  # story_country
            return (
                "Task: You will be given a short story involving 2 or more "
                "characters about everyday situations. You will also be "
                "given a country of relevance. Read the story and reason "
                "if the action by the character in the story is socially "
                "acceptable in the given country. The answer options are "
                "Yes, No or Neither.\n\n"
                f"Country: {country}\n"
                f"Story: {story}"
            )

    # ------------------------------------------------------------------
    # Parse Thought/Action from model output
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_react_output(text: str) -> Dict[str, str]:
        """
        Parse model output for Thought and Action.
        Returns {"thought": "...", "action": "..."}.
        """
        # Strip think tags
        clean = _RE_THINK.sub("", text, count=0).strip()

        thought = ""
        action = ""

        # Extract Thought
        m = re.search(r'Thought:\s*(.+?)(?=\nAction:|\Z)', clean, re.DOTALL)
        if m:
            thought = m.group(1).strip()

        # Extract Action
        m = re.search(r'Action:\s*(\S+)', clean)
        if m:
            action = m.group(1).strip().lower()

        # Normalize action
        valid_actions = {"hofstede_tool", "cultural_atlas_tool", "wikipedia_rag", "finish"}
        if action not in valid_actions:
            # Try fuzzy match
            for va in valid_actions:
                if va in clean.lower():
                    action = va
                    break
            else:
                action = "finish"  # default if unparseable

        return {"thought": thought, "action": action}

    # ------------------------------------------------------------------
    # Main: ReAct loop
    # ------------------------------------------------------------------
    def react_loop(
        self,
        prompt_type: str,
        story: str,
        country: str,
        baseline_reasoning: str,
        baseline_answer: str,
        tool_caller: Callable,
        value: str = "",
        rot: str = "",
    ) -> Dict[str, Any]:
        """
        ReAct-style iterative reasoning loop.

        Args:
            tool_caller: function(tool_name, country, story) -> dict
                Returns tool output (hofstede_data, cultural_atlas_data, etc.)

        Returns same dict format as other coordinators.
        """
        scenario = self._build_scenario(
            prompt_type, story, country, value, rot
        )

        # Build initial user message. Wording aligned with synthesize() Turn 2:
        # "Initial Analysis: ..." / "Initial Answer: ..." rather than the
        # earlier all-caps "YOUR INITIAL ANALYSIS:" framing. Keeps the model's
        # task framing identical across synthesize and ReAct paths.
        initial_user = (
            f"{scenario}\n\n"
            f"Initial Analysis: {baseline_reasoning}\n"
            f"Initial Answer: {baseline_answer}\n\n"
            "You may now use tools to verify your answer. "
            "Start with a Thought about what you need to check, "
            "then choose an Action."
        )

        # Track state
        conversation_history = initial_user
        tools_called: List[str] = []
        tools_used: List[str] = []
        observations: List[Dict[str, str]] = []
        all_raw: List[str] = []
        iteration = 0
        # Raw tool-result blobs, keyed to match static/agentic trajectory
        # schema (hofstede_data / cultural_atlas_data / wikipedia_data).
        # Lets downstream ablations reformat or re-rank evidence without
        # re-calling tools -- previously impossible for ReAct trajectories.
        tool_raw_data: Dict[str, Optional[Dict[str, Any]]] = {
            "hofstede_data":       None,
            "cultural_atlas_data": None,
            "wikipedia_data":      None,
        }
        # Reading-turn parity with standard synthesize():
        #   - Hofstede observations pass through RAW.
        #   - Atlas / Wiki observations go through _read_evidence() so the
        #     model sees the same compressed "[X -- relevant points]" block
        #     as the static / agentic pipelines. We store only the compressed
        #     text in the trajectory (matches what the model saw) -- no raw
        #     observation is retained.
        reading_status:  Dict[str, str] = {}
        reading_outputs: Dict[str, str] = {}
        scenario_for_reading = self._build_reading_scenario(
            prompt_type, story, country, value, rot,
        )
        # Track Wikipedia input truncation if/when wiki gets called.
        # Mirrors synthesize()'s wiki_truncation field for analysis parity.
        wiki_truncation: Dict[str, Any] = {
            "input_truncated":  False,
            "sections_total":   0,
            "sections_kept":    [],
            "sections_dropped": [],
        }

        while iteration < MAX_ITERATIONS:
            iteration += 1

            # Generate Thought + Action
            raw = self.generate(
                user_message=conversation_history,
                system_message=self._react_system_prompt,
                max_new_tokens=1024,
            )
            all_raw.append(f"ITERATION_{iteration}:\n{raw}")

            parsed = self._parse_react_output(raw)
            thought = parsed["thought"]
            action = parsed["action"]

            logger.info(
                "[ReAct] Iter %d -- Thought: %s... Action: %s",
                iteration, thought[:80], action,
            )

            # Check for finish
            if action == "finish":
                break

            # Check if already called this tool
            if action in tools_called:
                logger.info("[ReAct] Tool %s already called -- forcing finish", action)
                break

            # Call the tool
            tools_called.append(action)
            tool_result = tool_caller(action, country, story)

            # Stash the raw tool blob into the schema-aligned dict so the
            # trajectory record can carry it (matches static + agentic).
            if action == "hofstede_tool":
                tool_raw_data["hofstede_data"] = tool_result
            elif action == "cultural_atlas_tool":
                tool_raw_data["cultural_atlas_data"] = tool_result
            elif action == "wikipedia_rag":
                tool_raw_data["wikipedia_data"] = tool_result

            # Format raw observation. For Wikipedia, this also captures any
            # truncation info into wiki_truncation.
            raw_observation = self._format_tool_result(
                action, tool_result, wiki_truncation_out=wiki_truncation,
            )

            # ── Reading-turn compression for Atlas / Wiki ─────────────
            # Match standard synthesize(): Atlas/Wiki raw evidence is first
            # compressed via _read_evidence(); only the compressed text
            # reaches the model's view. Hofstede stays raw (small block).
            observation = raw_observation
            if (action in ("cultural_atlas_tool", "wikipedia_rag")
                    and raw_observation
                    and "Not available for this country" not in raw_observation
                    and len(raw_observation) > 50):
                tool_label = (
                    "Cultural Atlas" if action == "cultural_atlas_tool"
                    else "Wikipedia"
                )
                relevance, status = self._read_evidence(
                    scenario_for_reading, raw_observation, tool_label,
                )
                short_key = "atlas" if action == "cultural_atlas_tool" else "wiki"
                reading_status[short_key] = status
                if status == "ok":
                    reading_outputs[short_key] = relevance
                    observation = (
                        f"[{tool_label} -- relevant points]\n{relevance}"
                    )
                else:
                    # Surface reading-turn failure to the model (matches
                    # standard's diagnosis-friendly behaviour).
                    observation = (
                        f"[{tool_label} -- relevant points]\n"
                        f"  Reading-turn output was {status}; treat as unavailable."
                    )
                    if relevance:
                        reading_outputs[short_key] = relevance

            # Option B: even "Not available" observations are surfaced to the
            # model. tools_used tracks tools that returned ACTUAL data (not
            # just availability markers); this matches synthesize()'s
            # tools_used semantics for parity.
            if observation and "Not available for this country" not in observation \
                    and "treat as unavailable" not in observation:
                tools_used.append(action)

            observations.append({
                "tool": action,
                "observation": observation or "(no data returned)",
            })

            # Append to conversation for next iteration
            conversation_history += (
                f"\n\nThought: {thought}\n"
                f"Action: {action}\n"
                f"Observation: {observation or '(no data returned)'}\n\n"
                f"Continue with your next Thought and Action."
            )

        # -- Extract final answer ------------------------------------------
        final_user = (
            conversation_history + "\n\n"
            "You have finished using tools. State your final answer.\n\n"
            "Options:\n"
            "1) Yes\n"
            "2) No\n"
            "3) Neither\n\n"
            "Format your response EXACTLY as:\n"
            "Answer: <Yes|No|Neither>\n"
            "Explanation: <your analysis citing specific evidence>"
        )

        final_raw = self.generate(
            user_message=final_user,
            system_message=self._react_final_system_prompt,
            max_new_tokens=1024,
        )
        all_raw.append(f"FINAL_ANSWER:\n{final_raw}")

        final_clean = _RE_THINK.sub("", final_raw, count=0).strip()
        final_status = _classify_output(final_clean)

        if final_status in ("empty", "degenerated"):
            logger.warning("[ReAct] Final answer degenerated")
            self.decisions_made += 1
            return {
                "decision":         final_status,
                "reasoning":        "",
                "raw_response":     "\n\n".join(all_raw),
                "answer_changed":   True,
                "tools_used":       tools_used,
                "selected_tools":   tools_called,
                "evidence_used":    bool(tools_used),
                "iterations":       iteration,
                "observations":     observations,
                "output_status":    f"final_{final_status}",
                "wiki_truncation":  wiki_truncation,
                "reading_status":   reading_status,
                "reading_outputs":  reading_outputs,
                # Raw tool blobs (schema parity with static + agentic)
                "hofstede_data":       tool_raw_data["hofstede_data"],
                "cultural_atlas_data": tool_raw_data["cultural_atlas_data"],
                "wikipedia_data":      tool_raw_data["wikipedia_data"],
            }

        # Parse JSON
        parsed = self._parse_json_response(final_clean)
        decision = parsed["decision"]
        reasoning = parsed["reasoning"]
        answer_changed = decision != baseline_answer.lower().strip()

        self.decisions_made += 1

        final_out_status = "unknown" if decision == "unknown" else "ok"

        return {
            "decision":         decision,
            "reasoning":        reasoning,
            "raw_response":     "\n\n".join(all_raw),
            "answer_changed":   answer_changed,
            "tools_used":       tools_used,
            "selected_tools":   tools_called,
            "evidence_used":    bool(tools_used),
            "iterations":       iteration,
            "observations":     observations,
            "output_status":    final_out_status,
            "wiki_truncation":  wiki_truncation,
            "reading_status":   reading_status,
            "reading_outputs":  reading_outputs,
            # Raw tool blobs (schema parity with static + agentic)
            "hofstede_data":       tool_raw_data["hofstede_data"],
            "cultural_atlas_data": tool_raw_data["cultural_atlas_data"],
            "wikipedia_data":      tool_raw_data["wikipedia_data"],
        }

    # ------------------------------------------------------------------
    # Format tool results as observation text
    # ------------------------------------------------------------------
    def _format_tool_result(
        self,
        tool_name: str,
        data: Optional[Dict[str, Any]],
        wiki_truncation_out: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Format a single tool's output as an observation string.

        For Wikipedia, applies the same per-model budget + boundary-aware
        truncation logic as synthesize(). Captures truncation report into
        wiki_truncation_out (if provided) so the caller can surface it
        in the final return dict for parity with synthesize().

        Option B behaviour: if a tool was tried but returned no usable
        data (e.g. country not in Hofstede CSV, Atlas country missing,
        Wikipedia fetch failure), we return an explicit "Not available
        for this country." observation rather than None. This mirrors
        synthesize()'s honest-prompt approach and lets the model see
        which tools were attempted vs. unavailable.
        """
        if data is None:
            # Tool wasn't called at all (shouldn't happen during ReAct since
            # we only invoke tool_caller on selected actions, but be safe).
            return None

        if tool_name == "hofstede_tool":
            if data.get("success"):
                return self._format_hofstede(data)
            return (
                "[Hofstede Cultural Dimensions]\n"
                "  Not available for this country."
            )

        elif tool_name == "cultural_atlas_tool":
            if data.get("retrieved"):
                return self._format_cultural_atlas(data)
            return (
                "[Cultural Atlas]\n"
                "  Not available for this country."
            )

        elif tool_name == "wikipedia_rag":
            if not data.get("retrieved"):
                return (
                    "[Wikipedia]\n"
                    "  Not available for this country."
                )
            # Wikipedia retrieved -- apply per-model budget for observation size
            short = self._model_short_name()
            reading_ceiling = _READING_MAX_TOKENS.get(short, _READING_MAX_TOKENS_DEFAULT)
            wiki_token_budget = max(
                self.context_window - reading_ceiling - 1024 - 500,
                512,
            )
            text, info = self._format_wikipedia(
                data,
                tokenizer=self.tokenizer,
                max_tokens=wiki_token_budget,
            )
            if wiki_truncation_out is not None and info["sections_total"] > 0:
                wiki_truncation_out.update(info)
                if info["input_truncated"]:
                    logger.info(
                        "[ReAct/Wikipedia] Input truncated: kept %d/%d sections "
                        "(dropped: %s)",
                        len(info["sections_kept"]),
                        info["sections_total"],
                        info["sections_dropped"],
                    )
            return text

        return None