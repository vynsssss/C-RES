# -*- coding: utf-8 -*-
"""
KB-Grounding Coordinator
KB-grounding strategy from Lertvittayakumjorn et al. (ACL 2025).

This is NOT C-RES synthesis. It deliberately reproduces the paper's pipeline:

    query -> retrieve n=5 (fused KB) -> [optional per-doc relevance filter]
          -> augment prompt with kept docs -> single generation -> answer

Differences from coordinator.Coordinator.synthesize() (all intentional, this is
the *baseline* we compare against):
  - NO baseline-first step (no initial answer fed in).
  - NO reading-turn compression.
  - Single LLM generation for the answer (plus k small yes/no calls in the
    SELECTIVE variant for the relevance check, exactly as the paper does).

Two variants (the paper's two KB-grounding settings):
  - non-selective: all retrieved docs go into the prompt.
  - selective:     each doc is passed to the SAME model with a relevance-check
                   prompt; only the docs judged relevant are kept.

Reuses Coordinator's parser/canonicaliser/scorer so output lands in the same
label space and the trajectory schema is unchanged.
"""

import logging
import re
from typing import Any, Dict, List, Tuple

from base_agent import BaseAgent
from coordinator import Coordinator, _RE_THINK, _classify_output

logger = logging.getLogger(__name__)


class KBGroundingCoordinator(BaseAgent):
    """Single-shot KB-grounding (paper-faithful baseline)."""

    def __init__(
        self,
        model,
        tokenizer,
        kb_retriever=None,
        kb_cache=None,
        device: str = "cuda",
        max_new_tokens: int = 2048,
        context_window: int = 0,
        temperature: float = 0.0,
        model_name: str = "",
        selective: bool = False,
        retrieve_n: int = 5,
    ):
        super().__init__(
            model=model, tokenizer=tokenizer, device=device,
            max_new_tokens=max_new_tokens, context_window=context_window,
            temperature=temperature, model_name=model_name,
        )
        # Either a live KBRetriever (embeds queries on the fly) OR a prebuilt
        # cache dict {"<sid>::<prompt_type>": {"query", "docs"}}. 
        if kb_retriever is None and kb_cache is None:
            raise ValueError("Provide kb_retriever or kb_cache.")
        self.kb = kb_retriever
        self.kb_cache = kb_cache
        self.selective = selective
        self.retrieve_n = retrieve_n
        self.decisions_made = 0

    @property
    def name(self) -> str:
        return "KBGroundingCoordinator"

    @property
    def system_prompt(self) -> str:
        """Format-only prompt -- same plain-text contract as Coordinator Step 1,
        so the shared parser works unchanged.
        """
        return (
            "Analyze the cultural scenario and answer the question.\n\n"
            "Format your response EXACTLY as:\n"
            "Answer: <Yes|No|Neither>\n"
            "Explanation: <your analysis>"
        )

    @property
    def relevance_system_prompt(self) -> str:
        """Per-doc relevance check for the SELECTIVE variant (paper's relevancy step).
        Wording matches the paper's NORMAD relevancy step 
        """
        return (
            "You are judging whether a piece of information is relevant and "
            "applicable to a question about a cultural scenario. "
            "Answer with only 'Yes' or 'No'."
        )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.answer(**input_data)

    # ------------------------------------------------------------------
    # Paper-verbatim NorMAD task framing (mirrors coordinator.synthesize())
    # ------------------------------------------------------------------
    @staticmethod
    def _task_and_scenario(
        prompt_type: str, story: str, country: str, value: str = "", rot: str = "",
    ) -> Tuple[str, str, str, str]:
        """Return (task_lines, header_block, story_line, retrieval_query).

        Follows the paper's NORMAD KB-grounding prompt:

            Task: ... and supporting information that may be useful. ...
            Country: <country>
            Rule: <value>
            Supporting information:
            - <doc 1>
            - <doc 2>
            Story: <story>
            Options: ...
            Answer (Yes, No or Neither):
            Explanation:                      <- our addition (kept for traces)

        header_block    = the Country/Rule lines that precede Supporting information.
        story_line      = the "Story: ..." line that follows it.
        retrieval_query = the paper's KB query template (also used as the prose
                          "Story-based question:" in the selective relevance step).
        """
        if prompt_type == "story_country_value":
            # Paper Table 4 exact task line (Country + Value == "country / rule").
            task_lines = (
                "Task: You will be given a short story involving 2 or more "
                "characters about everyday situations. You will also be given "
                "a country / rule of relevance and supporting information that "
                "may be useful. Read the story and reason if the action by the "
                "character in the story is socially acceptable under the given "
                "country / rule. Do not make any extra inferences about actions "
                "outside of what is given. Only align to the country / rule "
                "given. Respond with an answer - Yes, No or Neither."
            )
            header_block = f"Country: {country}\nRule: {value}"
            story_line = f"Story: {story}"
            query = f"In {country}, {story} Please consider the value of {value}"
        elif prompt_type == "story_rot":
            # No paper NORMAD template for rot-only; use the rule-of-thumb in the
            # Rule slot to keep the same paper-faithful layout across prompt types.
            task_lines = (
                "Task: You will be given a short story involving 2 or more "
                "characters about everyday situations. You will also be given "
                "a rule of relevance and supporting information that may be "
                "useful. Read the story and reason if the action by the "
                "character in the story is socially acceptable under the given "
                "rule. Do not make any extra inferences about actions outside "
                "of what is given. Only align to the rule given. Respond with "
                "an answer - Yes, No or Neither."
            )
            header_block = f"Rule: {rot}"
            story_line = f"Story: {story}"
            query = f"{rot} {story}"
        else:  # story_country
            # Paper Table 4 exact task line, Country-only variant.
            task_lines = (
                "Task: You will be given a short story involving 2 or more "
                "characters about everyday situations. You will also be given "
                "a country of relevance and supporting information that may be "
                "useful. Read the story and reason if the action by the "
                "character in the story is socially acceptable in the given "
                "country. Do not make any extra inferences about actions "
                "outside of what is given. Only align to the country given. "
                "Respond with an answer - Yes, No or Neither."
            )
            header_block = f"Country: {country}"
            story_line = f"Story: {story}"
            query = f"In {country}, {story}"
        return task_lines, header_block, story_line, query

    # ------------------------------------------------------------------
    # Relevance filter (SELECTIVE variant)
    # ------------------------------------------------------------------
    def _doc_is_relevant(self, question_query: str, doc_text: str) -> bool:
        """Ask the same model whether one doc is relevant (paper's relevancy step).

        Mirrors the paper's NORMAD relevancy-check prompt (Table 4): a
        story-based question + the candidate information, asking if it is
        relevant and applicable, answered Yes/No.
        """
        user = (
            "Task: You will be given a story-based question and a piece of "
            "information. Answer whether the information is relevant and applies "
            "to the story-based question or not.\n"
            f"Story-based question: \"{question_query}\"\n"
            f"Information: \"{doc_text}\"\n"
            "Is the information relevant and applicable to the question?\n"
            "Options:\n1) Yes\n2) No\nAnswer (Yes or No):"
        )
        try:
            raw = self.generate(
                user_message=user,
                system_message=self.relevance_system_prompt,
                max_new_tokens=64,   # was 8: base_agent raises ValueError <50, making selective==non-selective
            )
        except ValueError:
            # Prompt too long for a tiny budget -> treat as relevant (keep).
            return True
        stripped = _RE_THINK.sub("", raw, count=0).strip().lower()
        return stripped.startswith("y") or "yes" in stripped[:10]

    # ------------------------------------------------------------------
    # Main entry: retrieve -> (filter) -> augment -> answer
    # ------------------------------------------------------------------
    def answer(
        self,
        prompt_type: str,
        story: str,
        country: str,
        value: str = "",
        rot: str = "",
        sample_id: str = "",
    ) -> Dict[str, Any]:
        """Single-shot KB-grounded answer. Returns a synthesize()-compatible dict."""
        task_lines, header_block, story_line, query = self._task_and_scenario(
            prompt_type, story, country, value, rot,
        )

        # 1. Retrieve n docs: prefer the prebuilt cache (no embedder), else live.
        if self.kb_cache is not None:
            entry = self.kb_cache.get(f"{sample_id}::{prompt_type}")
            docs: List[Dict[str, Any]] = list(entry["docs"]) if entry else []
        else:
            kb_result = self.kb.retrieve(query=query, country=country, n=self.retrieve_n)
            docs = kb_result.get("docs", []) if kb_result.get("retrieved") else []
        n_retrieved = len(docs)

        # 2. Selective relevance filter (paper's selective variant).
        n_kept = n_retrieved
        if self.selective and docs:
            kept = [d for d in docs if self._doc_is_relevant(query, d["text"])]
            docs = kept
            n_kept = len(kept)

        # 3. Build the Supporting information block.
        if docs:
            support_lines = ["Supporting information:"]
            for d in docs:
                support_lines.append(f"- {d['text']}")
            support_text = "\n".join(support_lines)
        else:
            support_text = "Supporting information:\n- (No relevant information found.)"

        # 4. Assemble the prompt in the paper's EXACT NORMAD order:
        #    task -> Country/Rule header -> Supporting information -> Story ->
        #    Options -> Answer. We append "Explanation:" so the model also emits
        #    reasoning (kept for trajectory traces; the parser reads Answer:).
        user_message = (
            f"{task_lines}\n\n"
            f"{header_block}\n"
            f"{support_text}\n"
            f"{story_line}\n\n"
            f"Options:\n1) Yes\n2) No\n3) Neither\n\n"
            f"Answer (Yes, No or Neither):\n"
            f"Explanation:"
        )

        # 5. Adaptive max_new_tokens (same discipline as synthesize()).
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_tokens = len(self.tokenizer.encode(formatted))
        adaptive_max = max(512, min(self.context_window - input_tokens - 50,
                                    self.max_new_tokens))

        raw = self.generate(
            user_message=user_message,
            system_message=self.system_prompt,
            max_new_tokens=adaptive_max,
        )

        # 6. Parse + classify (reuse Coordinator's logic).
        stripped = _RE_THINK.sub("", raw, count=0).strip()
        output_status = _classify_output(stripped)
        self.decisions_made += 1

        if output_status in ("empty", "degenerated"):
            return {
                "decision": output_status, "reasoning": "", "raw_response": raw,
                "output_status": output_status,
                "kb_retrieved": n_retrieved, "kb_kept": n_kept,
                "kb_sources": [d["source"] for d in docs],
            }

        parsed = Coordinator._parse_json_response(stripped)
        decision = parsed["decision"]
        final_status = "unknown" if decision == "unknown" else "ok"

        return {
            "decision": decision,
            "reasoning": parsed["reasoning"],
            "raw_response": raw,
            "output_status": final_status,
            "kb_retrieved": n_retrieved,
            "kb_kept": n_kept,
            "kb_sources": [d["source"] for d in docs],
        }