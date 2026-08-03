# -*- coding: utf-8 -*-
"""
KB-through-synthesize coordinator.

This is the SECOND of two ways we study the published KB-grounding method:

  - Option 1 (coordinator_kb.py): faithful single-shot KB-grounding -- the
    paper's prompt, retrieve-then-answer in ONE turn. Reproduces the paper's
    *orchestration* (single-shot) on the paper's *source* (the bespoke KB).

  - Option 2 (THIS FILE): the SAME retrieved KB documents, but fed through OUR
    multi-turn synthesize() orchestration -- a reading turn compresses the
    retrieved docs to scenario-relevant points, then a synthesis turn judges.
    Reproduces the paper's *source* (the KB) under OUR *orchestration*.

Holding the retrieval constant (both options read the SAME kb_cache.json, so
the SAME 5 documents are retrieved per item) and varying only the orchestration
isolates the effect of the control regime -- which is the contribution of this
paper. Option 1 vs Option 2 = same docs, different handling.

DESIGN:
  retrieve 5 (from cache) -> READING TURN extracts the relevant points from
  those 5 -> SYNTHESIS judges using the baseline + the reading-turn summary.

  NON-SELECTIVE only. In the paper, "selective" is a per-document yes/no filter
  applied to the 5 retrieved docs before they enter the prompt. In OUR
  synthesize pipeline the reading turn already performs relevance handling
  (it extracts the relevant passages and drops the rest), so it IS our relevance
  mechanism. Stacking the paper's binary per-doc filter on top would double-count
  relevance and blur which mechanism does the work. 

This class SUBCLASSES Coordinator so it inherits, unchanged:
  - the reading turn (_read_evidence) and its per-model token budgeting,
  - the synthesis/reading system prompts,
  - _parse_json_response / _classify_output / is_correct,
  - the prompt-type scenario/task construction conventions.
It overrides ONLY how the evidence block is assembled: a single KB channel
sourced from the cache, instead of hofstede/atlas/wiki.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from coordinator import Coordinator, _RE_THINK, _classify_output

logger = logging.getLogger(__name__)

# Max chars of a single retrieved KB doc to show the reading turn. CultureAtlas
# summaries can be long; this bounds the reading-turn input. The reading turn
# then compresses further. 
_KB_DOC_CHAR_CAP = 4000
# Max number of retrieved docs to include (defensive; retrieval is n=5).
_KB_MAX_DOCS = 5


class KBSynthCoordinator(Coordinator):
    """KB documents fed through the multi-turn synthesize() orchestration.

    Usage:
        co = KBSynthCoordinator(model, tokenizer, kb_cache=cache,
                                context_window=..., max_new_tokens=2048,
                                model_name="Qwen/Qwen3-4B-Instruct-2507",
                                temperature=0.7)
        out = co.synthesize_kb(prompt_type, story, country,
                               baseline_reasoning, baseline_answer,
                               value=..., rot=..., sample_id=...)

    """

    def __init__(
        self,
        model,
        tokenizer,
        kb_cache: Optional[Dict[str, Any]] = None,
        kb_retriever=None,
        retrieve_n: int = 5,
        device: str = "cuda",
        max_new_tokens: int = 2048,
        context_window: int = 0,
        temperature: float = 0.0,
        model_name: str = "",
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            temperature=temperature,
            model_name=model_name,
        )
        # Exactly one of kb_cache / kb_retriever is used. 
        self.kb_cache = kb_cache
        self.kb = kb_retriever
        self.retrieve_n = retrieve_n

    @property
    def name(self) -> str:
        return "KBSynthCoordinator"

    # ------------------------------------------------------------------
    # Retrieval (cache-first)
    # ------------------------------------------------------------------
    def _retrieve_docs(
        self, query: str, country: str, sample_id: str, prompt_type: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Return (docs, retrieval_status) for this item.

        retrieval_status is one of:
          - "ok"          -> docs found (len >= 0; cache hit or live retrieval)
          - "cache_miss"  -> the key "sample_id::prompt_type" is ABSENT from the
                             cache. This is a HARD problem, not an empty result:
                             it means the cache was built from a baseline that
                             did not contain this item (e.g. a prompt_type the
                             prefetch never saw). Silently running synthesis with
                             no evidence in that case would corrupt results while
                             looking fine, so we flag it loudly instead.
          - "no_retriever"-> neither cache nor live retriever configured.
        """
        if self.kb_cache is not None:
            key = f"{sample_id}::{prompt_type}"
            if key not in self.kb_cache:
                # Distinguish ABSENT key (cache built without this item) from a
                # present-but-empty retrieval. Absent = data problem -> loud.
                logger.error(
                    "[KB-Synth] CACHE MISS for key '%s' -- this item was not in "
                    "the prefetch baseline. Synthesis would run with NO evidence. "
                    "Rebuild kb_cache.json from a baseline covering all prompt "
                    "types, or check sample_id/prompt_type formatting.", key,
                )
                return [], "cache_miss"
            entry = self.kb_cache[key]
            return list(entry.get("docs", [])), "ok"
        if self.kb is not None:
            res = self.kb.retrieve(query=query, country=country, n=self.retrieve_n)
            docs = res.get("docs", []) if res.get("retrieved") else []
            return docs, "ok"
        return [], "no_retriever"

    @staticmethod
    def _kb_query(prompt_type: str, story: str, country: str,
                  value: str = "", rot: str = "") -> str:
        """The paper's KB query template.

        Only used in the live-retriever fallback; with a cache the query was
        already used at prefetch time.
        """
        if prompt_type == "story_country_value":
            return f"In {country}, {story} Please consider the value of {value}"
        if prompt_type == "story_rot":
            return f"{rot} {story}"
        return f"In {country}, {story}"

    @staticmethod
    def _format_kb_block(docs: List[Dict[str, Any]]) -> str:
        """Format retrieved KB docs as a single evidence block for the reading turn.
        """
        if not docs:
            return ""
        lines = ["[Knowledge Base]"]
        for d in docs[:_KB_MAX_DOCS]:
            text = (d.get("text") or "").strip()
            if not text:
                continue
            if len(text) > _KB_DOC_CHAR_CAP:
                text = text[:_KB_DOC_CHAR_CAP] + "..."
            lines.append(f"- {text}")
        return "\n".join(lines) if len(lines) > 1 else ""

    # ------------------------------------------------------------------
    # Main entry: retrieve -> reading turn -> synthesize
    # ------------------------------------------------------------------
    def synthesize_kb(
        self,
        prompt_type: str,
        story: str,
        country: str,
        baseline_reasoning: str,
        baseline_answer: str,
        value: str = "",
        rot: str = "",
        sample_id: str = "",
    ) -> Dict[str, Any]:
        """KB-through-synthesize: retrieve 5 -> reading turn -> Turn-2 synthesis.

        Reuses the parent's reading turn (_read_evidence), reading/synthesis
        system prompts, scenario/task construction, parser and output classifier
        verbatim -- only the evidence channel differs (KB instead of the tools).
        """
        # 1. Build the same scenario/task skeleton the parent synthesize() uses.
        #    scenario_for_reading: Country/Story (or RoT/Story) only -- NO Task,
        #    NO Options, NO judgment question (keeps yes/no framing out of the
        #    reading turn, exactly as the parent synthesize() does).
        if prompt_type == "story_country_value":
            scenario_for_reading = f"Country: {country}\nRoT: {value}\nStory: {story}"
        elif prompt_type == "story_rot":
            scenario_for_reading = f"Rule-of-thumb: {rot}\nStory: {story}"
        else:  # story_country
            scenario_for_reading = f"Country: {country}\nStory: {story}"

        if prompt_type == "story_country_value":
            task_lines = (
                "Task: You will be given a short story involving 2 or more "
                "characters about everyday situations. You will also be given "
                "a country of relevance, along with a rule-of-thumb. Read the "
                "story and reason if the action by the character in the story "
                "is socially acceptable in the given country with the "
                "associated rule-of-thumb. The answer options are Yes, No or "
                "Neither."
            )
            scenario_block = f"Country: {country}\nRoT: {value}\nStory: {story}"
        elif prompt_type == "story_rot":
            task_lines = (
                "Task: You will be given a short story involving 2 or more "
                "characters about everyday situations. You will also be given "
                "a rule-of-thumb. Read the story and reason if the action by "
                "the character in the story is socially acceptable under the "
                "given rule-of-thumb alone. The answer options are Yes, No or "
                "Neither."
            )
            scenario_block = f"Rule-of-thumb: {rot}\nStory: {story}"
        else:  # story_country
            task_lines = (
                "Task: You will be given a short story involving 2 or more "
                "characters about everyday situations. You will also be given "
                "a country of relevance. Read the story and reason if the "
                "action by the character in the story is socially acceptable "
                "in the given country. The answer options are Yes, No or "
                "Neither."
            )
            scenario_block = f"Country: {country}\nStory: {story}"

        # 2. Retrieve the same 5 docs (cache-first).
        query = self._kb_query(prompt_type, story, country, value, rot)
        docs, retrieval_status = self._retrieve_docs(
            query, country, sample_id, prompt_type,
        )
        n_retrieved = len(docs)

        # 3. Reading turn: compress the retrieved docs to scenario-relevant
        #    points (our relevance mechanism). Same _read_evidence the
        #    parent uses for Atlas/Wiki.
        tools_used: List[str] = []
        reading_status: Dict[str, str] = {}
        reading_outputs: Dict[str, str] = {}

        if retrieval_status in ("cache_miss", "no_retriever"):
            reading_status["kb"] = retrieval_status

        kb_block = self._format_kb_block(docs)
        if len(kb_block) > 50:
            relevance, status = self._read_evidence(
                scenario_for_reading, kb_block, "Knowledge Base",
            )
            reading_status["kb"] = status
            if status == "ok":
                reading_outputs["kb"] = relevance
                evidence_block = "[Knowledge Base -- relevant points]\n" + relevance
                tools_used.append("kb_grounding")
            else:
                evidence_block = (
                    "[Knowledge Base -- relevant points]\n"
                    f"  Reading-turn output was {status}; treat as unavailable."
                )
                if relevance:
                    reading_outputs["kb"] = relevance
        else:
            reading_status.setdefault("kb", "unavailable")
            evidence_block = (
                "[Knowledge Base]\n  No relevant information retrieved."
            )

        # 4. Turn-2 synthesis prompt -- IDENTICAL skeleton to the
        #    parent synthesize() (Task -> scenario -> Initial Analysis ->
        #    Initial Answer -> Cultural Evidence -> Options -> Answer/Explanation),
        user_message = (
            f"{task_lines}\n\n"
            f"{scenario_block}\n\n"
            f"Initial Analysis: {baseline_reasoning}\n"
            f"Initial Answer: {baseline_answer}\n\n"
            f"Cultural Evidence:\n"
            f"{evidence_block}\n\n"
            f"Options:\n1) Yes\n2) No\n3) Neither\n\n"
            f"Answer (options Yes, No or Neither):\n"
            f"Explanation:"
        )

        # 5. Adaptive max_new_tokens (same discipline as the parent).
        final_messages = [
            {"role": "system", "content": self.synthesis_system_prompt},
            {"role": "user", "content": user_message},
        ]
        final_formatted = self.tokenizer.apply_chat_template(
            final_messages, tokenize=False, add_generation_prompt=True,
        )
        actual_input_tokens = len(self.tokenizer.encode(final_formatted))
        adaptive_max = self.context_window - actual_input_tokens - 50
        adaptive_max = max(512, min(adaptive_max, self.max_new_tokens))

        logger.info(
            "[KB-Synth] input=%d tokens, generation=%d tokens",
            actual_input_tokens, adaptive_max,
        )

        raw = self.generate(
            user_message=user_message,
            system_message=self.synthesis_system_prompt,
            max_new_tokens=adaptive_max,
        )

        stripped = _RE_THINK.sub("", raw, count=0).strip()
        output_status = _classify_output(stripped)

        if output_status in ("empty", "degenerated"):
            self.decisions_made += 1
            return {
                "decision": output_status,
                "reasoning": "",
                "raw_response": raw,
                "answer_changed": True,
                "tools_used": tools_used,
                "output_status": output_status,
                "reading_status": reading_status,
                "reading_outputs": reading_outputs,
                "kb_retrieved": n_retrieved,
                "retrieval_status": retrieval_status,
                "kb_kept": n_retrieved,
                "relevance_mechanism": "reading_turn",
            }

        parsed = self._parse_json_response(stripped)
        decision = parsed["decision"]
        reasoning = parsed["reasoning"]
        answer_changed = decision != baseline_answer.lower().strip()
        self.decisions_made += 1

        final_status = "unknown" if decision == "unknown" else "ok"

        return {
            "decision": decision,
            "reasoning": reasoning,
            "raw_response": raw,
            "answer_changed": answer_changed,
            "tools_used": tools_used,
            "output_status": final_status,
            "reading_status": reading_status,
            "reading_outputs": reading_outputs,
            "kb_retrieved": n_retrieved,
            "retrieval_status": retrieval_status,
            "kb_kept": n_retrieved,
            "relevance_mechanism": "reading_turn",
        }