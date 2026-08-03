#!/usr/bin/env python3
"""
ReAct (Reasoning + Acting) trajectory collector.

Unlike the agentic collector which uses 3 fixed turns:
  Turn 1: select tools → Turn 2: retrieve → Turn 3: synthesize

ReAct uses an iterative loop:
  Thought → Action → Observation → Thought → Action → Observation → ... → finish

The model sees each tool's output before deciding the next action.
Max 3 iterations (one per available tool).

Usage:
    python step2_react_collect.py \\
        --baseline /path/to/baseline.json \\
        --output-dir /path/to/output \\
        --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --temperature 0.6 \\
        --resume
"""

import argparse
import json
import logging
import os
import re
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import torch
except ModuleNotFoundError:  # vLLM path doesn't need torch
    torch = None
# (transformers load removed for vLLM-only build)

from coordinator import Coordinator   # for is_correct (canonicalises neither→neutral)

logger = logging.getLogger(__name__)

PROMPT_TYPES = ["story_country", "story_country_value", "story_rot"]


# ── Trajectory builder ──────────────────────────────────────────────────
def build_react_trajectory(
    sample: Dict[str, Any],
    prompt_type: str,
    # ReAct results
    selected_tools: List[str],
    tools_used: List[str],
    evidence_used: bool,
    iterations: int,
    observations: List[Dict[str, str]],
    # Baseline
    baseline_reasoning: str,
    baseline_answer: str,
    # Final
    updated_reasoning: str,
    final_answer: str,
    answer_changed: bool,
    raw_response: str,
    output_status: str = "ok",
    # ── Reading-turn parity fields (added for fair comparison vs static) ──
    reading_status: Optional[Dict[str, str]] = None,
    reading_outputs: Optional[Dict[str, str]] = None,
    wiki_truncation: Optional[Dict[str, Any]] = None,
    # ── Raw tool blobs (schema parity with static + agentic) ──
    hofstede_data: Optional[Dict[str, Any]] = None,
    cultural_atlas_data: Optional[Dict[str, Any]] = None,
    wikipedia_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a trajectory record from ReAct loop results.
    """
    from coordinator import _canonicalise

    gold = sample.get("answer", "").lower().strip()
    base_ans = _canonicalise(baseline_answer)
    final_ans = _canonicalise(final_answer)

    baseline_correct = Coordinator.is_correct(base_ans, gold)
    is_correct = Coordinator.is_correct(final_ans, gold)

    return {
        "sample_id":          sample["id"],
        "prompt_type":        prompt_type,
        "country":            sample.get("country", ""),
        "story":              sample.get("story", ""),
        "value":              sample.get("value", ""),
        "rot":                sample.get("rot", ""),
        "gold_answer":        gold,
        "method":             "react",

        # ReAct specifics
        "selected_tools":     selected_tools,
        "tools_used":         tools_used,
        "evidence_used":      evidence_used,
        "iterations":         iterations,
        "observations":       observations,

        # Baseline
        "baseline_reasoning": baseline_reasoning,
        "baseline_answer":    base_ans,
        "baseline_correct":   baseline_correct,

        # Final
        "reasoning":          updated_reasoning,
        "final_answer":       final_ans,
        "decision":           final_ans,
        "is_correct":         is_correct,
        "answer_changed":     answer_changed,
        "raw_response":       raw_response,
        "output_status":      output_status,
        # Reading-turn parity fields (mirror static synthesize trajectories)
        "reading_status":     reading_status or {},
        "reading_outputs":    reading_outputs or {},
        "wiki_truncation":    wiki_truncation or {},
        # Raw tool blobs (schema parity with static + agentic)
        "hofstede_data":         hofstede_data,
        "cultural_atlas_data":   cultural_atlas_data,
        "wikipedia_data":        wikipedia_data,
        "timestamp":          datetime.now().isoformat(),
    }


# ── Collector ────────────────────────────────────────────────────────────
class ReactTrajectoryCollector:
    """Collects ReAct trajectories with iterative tool use."""

    def __init__(
        self,
        coordinator,
        cultural_helper,
        baseline_records: List[Dict[str, Any]],
        output_dir: str,
        checkpoint_interval: int = 100,
    ):
        self.coordinator = coordinator
        self.cultural_helper = cultural_helper
        self.records = baseline_records
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval

        self.trajectories: List[Dict[str, Any]] = []
        self.by_prompt: Dict[str, List] = {p: [] for p in PROMPT_TYPES}
        self.stats = defaultdict(int)
        self.stats_by_prompt = {p: defaultdict(int) for p in PROMPT_TYPES}

    # ------------------------------------------------------------------
    # Checkpoint / Resume
    # ------------------------------------------------------------------
    def find_latest_checkpoint(self) -> Optional[Path]:
        checkpoints = list(self.output_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return None

        def _num(p: Path) -> int:
            m = re.search(r"checkpoint_(\d+)", p.name)
            return int(m.group(1)) if m else 0

        return max(checkpoints, key=_num)

    def load_checkpoint(self, path: Path) -> int:
        print(f"\nLoading checkpoint: {path}")
        data = json.loads(path.read_text())
        self.trajectories = data["trajectories"]
        self.stats = defaultdict(int, data.get("stats", {}))

        for t in self.trajectories:
            pt = t.get("prompt_type", "")
            if pt in self.by_prompt:
                self.by_prompt[pt].append(t)

        for pt in PROMPT_TYPES:
            trajs = self.by_prompt[pt]
            self.stats_by_prompt[pt]["total"] = len(trajs)
            self.stats_by_prompt[pt]["correct"] = sum(
                1 for t in trajs if t.get("is_correct")
            )

        completed = data["completed"]
        print(f"  Restored {completed}/{data['total']} — resuming from {completed + 1}")
        return completed

    def save_checkpoint(self, completed: int) -> None:
        path = self.output_dir / f"checkpoint_{completed}.json"
        tmp = path.with_suffix(".tmp")
        data = {
            "completed":    completed,
            "total":        len(self.records),
            "trajectories": self.trajectories,
            "stats":        dict(self.stats),
            "method":       "react",
            "timestamp":    datetime.now().isoformat(),
        }
        try:
            tmp.write_text(json.dumps(data))
            tmp.replace(path)
            print(f"\n[CHECKPOINT] {path} ({completed}/{len(self.records)})")
        except Exception as e:
            print(f"\n[CHECKPOINT FAILED] {e}")
            if tmp.exists():
                tmp.unlink()

    # ------------------------------------------------------------------
    # Tool caller (passed to ReactCoordinator)
    # ------------------------------------------------------------------
    def _make_tool_caller(self, sample: Dict[str, Any]):
        """Create a tool_caller function for this sample."""
        country = self.cultural_helper._normalize_country(sample.get("country", ""))
        story = sample.get("story", "")

        def tool_caller(tool_name: str, _country: str, _story: str):
            """Call a single tool and return its data."""
            if tool_name == "hofstede_tool":
                data = self.cultural_helper._call_tool("hofstede_tool", country=country)
                if data and data.get("success"):
                    self.cultural_helper.hofstede_calls += 1
                    return data
            elif tool_name == "cultural_atlas_tool":
                data = self.cultural_helper._call_tool("cultural_atlas_tool", country=country)
                if data and data.get("retrieved"):
                    self.cultural_helper.cultural_atlas_calls += 1
                    return data
            elif tool_name == "wikipedia_rag":
                data = self.cultural_helper._call_tool("wikipedia_rag", country=country, query=story)
                if data and data.get("retrieved"):
                    self.cultural_helper.wikipedia_calls += 1
                    return data
            return None

        return tool_caller

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def collect_all(self, start_idx: int = 0) -> None:
        total = len(self.records)
        print(f"\nReAct mode: iterative reasoning + acting")
        print(f"Processing {total - start_idx} records (start={start_idx})")
        print("=" * 70)

        for idx in range(start_idx, total):
            rec = self.records[idx]
            sample_id = rec.get("sample_id", "")
            prompt_type = rec.get("prompt_type", "")

            print(f"\n[{idx + 1}/{total}] {sample_id} — {prompt_type}")

            if "error" in rec:
                print("  SKIP (Step 1 error)")
                self.stats["skipped"] += 1
                continue

            baseline_reasoning = rec.get("reasoning", "")
            baseline_answer = rec.get("decision", "")

            if not baseline_reasoning or not baseline_answer:
                print("  SKIP (missing reasoning/decision)")
                self.stats["skipped"] += 1
                continue

            sample = {
                "id":      sample_id,
                "country": rec.get("country", ""),
                "story":   rec.get("story", ""),
                "value":   rec.get("value", ""),
                "rot":     rec.get("rot", ""),
                "answer":  rec.get("gold_answer", ""),
            }

            try:
                # ── ReAct loop ──────────────────────────────────────────
                tool_caller = self._make_tool_caller(sample)

                result = self.coordinator.react_loop(
                    prompt_type=prompt_type,
                    story=sample["story"],
                    country=sample["country"],
                    baseline_reasoning=baseline_reasoning,
                    baseline_answer=baseline_answer,
                    tool_caller=tool_caller,
                    value=sample["value"],
                    rot=sample["rot"],
                )

                traj = build_react_trajectory(
                    sample=sample,
                    prompt_type=prompt_type,
                    selected_tools=result["selected_tools"],
                    tools_used=result["tools_used"],
                    evidence_used=result["evidence_used"],
                    iterations=result["iterations"],
                    observations=result["observations"],
                    baseline_reasoning=baseline_reasoning,
                    baseline_answer=baseline_answer,
                    updated_reasoning=result["reasoning"],
                    final_answer=result["decision"],
                    answer_changed=result["answer_changed"],
                    raw_response=result["raw_response"],
                    output_status=result.get("output_status", "ok"),
                    reading_status=result.get("reading_status", {}),
                    reading_outputs=result.get("reading_outputs", {}),
                    wiki_truncation=result.get("wiki_truncation", {}),
                    hofstede_data=result.get("hofstede_data"),
                    cultural_atlas_data=result.get("cultural_atlas_data"),
                    wikipedia_data=result.get("wikipedia_data"),
                )

                self._record_trajectory(traj, prompt_type)

                out_status = result.get("output_status", "ok")
                if out_status in ("degenerated", "empty"):
                    print(
                        f"  ⚠ OUTPUT_{out_status.upper()} base={baseline_answer}"
                        f" (gold={sample['answer']})"
                        f" tools={result['selected_tools']}"
                    )
                else:
                    status = "✓" if traj["is_correct"] else "✗"
                    changed = " CHANGED" if traj["answer_changed"] else ""
                    iters = f" iters={result['iterations']}"
                    tools = f" tools={result['selected_tools']}"
                    print(
                        f"  {status} base={baseline_answer} → final={result['decision']}"
                        f" (gold={sample['answer']}){changed}{iters}{tools}"
                    )

            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()

                error_traj = {
                    "sample_id":    sample_id,
                    "prompt_type":  prompt_type,
                    "method":       "react",
                    "error":        str(e),
                    "is_correct":   False,
                    "timestamp":    datetime.now().isoformat(),
                }
                self._record_trajectory(error_traj, prompt_type, is_error=True)

            completed = idx + 1
            if completed % self.checkpoint_interval == 0:
                self.save_checkpoint(completed)
                if torch is not None and torch.cuda.is_available():
                    (torch.cuda.empty_cache() if torch is not None and torch.cuda.is_available() else None)

        self._print_stats()
        self._save_final()
        self.save_checkpoint(total)

    def _record_trajectory(
        self, traj: Dict, prompt_type: str, is_error: bool = False
    ) -> None:
        self.trajectories.append(traj)
        if prompt_type in self.by_prompt:
            self.by_prompt[prompt_type].append(traj)

        self.stats["total"] += 1
        pt_stats = self.stats_by_prompt.get(prompt_type, self.stats)

        if is_error:
            self.stats["errors"] += 1
            pt_stats["errors"] = pt_stats.get("errors", 0) + 1
        else:
            pt_stats["total"] = pt_stats.get("total", 0) + 1
            if traj.get("is_correct"):
                self.stats["correct"] += 1
                pt_stats["correct"] = pt_stats.get("correct", 0) + 1
            if traj.get("answer_changed"):
                self.stats["changed"] += 1
                pt_stats["changed"] = pt_stats.get("changed", 0) + 1

    def _print_stats(self):
        total = self.stats.get("total", 0)
        correct = self.stats.get("correct", 0)
        changed = self.stats.get("changed", 0)
        acc = correct / total * 100 if total > 0 else 0

        print("\n" + "=" * 70)
        print("  ReAct COLLECTION COMPLETE")
        print("=" * 70)
        print(f"  Total:   {total}")
        print(f"  Correct: {correct} ({acc:.1f}%)")
        print(f"  Changed: {changed}")
        print(f"  Errors:  {self.stats.get('errors', 0)}")
        print(f"  Skipped: {self.stats.get('skipped', 0)}")

        for pt in PROMPT_TYPES:
            pt_stats = self.stats_by_prompt[pt]
            pt_total = pt_stats.get("total", 0)
            pt_correct = pt_stats.get("correct", 0)
            pt_acc = pt_correct / pt_total * 100 if pt_total > 0 else 0
            print(f"    {pt:<25} {pt_correct}/{pt_total} ({pt_acc:.1f}%)")

    def _save_final(self):
        # Combined
        combined = self.output_dir / "trajectories_all_prompts.json"
        combined.write_text(
            json.dumps(self.trajectories, indent=2, ensure_ascii=False)
        )
        # Per prompt
        for pt in PROMPT_TYPES:
            if self.by_prompt[pt]:
                path = self.output_dir / f"trajectories_{pt}.json"
                path.write_text(
                    json.dumps(self.by_prompt[pt], indent=2, ensure_ascii=False)
                )
        # Stats
        stats_path = self.output_dir / "collection_stats.json"
        stats_path.write_text(json.dumps({
            "stats":    dict(self.stats),
            "method":   "react",
            "by_prompt": {
                pt: dict(self.stats_by_prompt[pt]) for pt in PROMPT_TYPES
            },
        }, indent=2))


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ReAct trajectory collector")
    parser.add_argument("--baseline", "--baseline-dir", type=str)
    parser.add_argument("--data-id", type=str, default="normad_dataset")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--tool-config", type=str, default="all",
                        choices=["hofstede", "atlas", "wiki",
                                 "hofstede_atlas", "hofstede_wiki", "atlas_wiki", "all"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm"],
                        help="vllm: use local vLLM server (skip transformers load)")
    parser.add_argument("--vllm-base-url", type=str, default=None)
    args = parser.parse_args()

    print("\n======================================================================")
    print("STEP 2 REACT: ITERATIVE REASONING + ACTING")
    print("======================================================================")
    print(f"  Model:        {args.model_name}")
    print(f"  Method:       ReAct (iterative thought-action-observation)")
    print(f"  Baseline:     {args.baseline}")
    print(f"  Output:       {args.output_dir}")
    print(f"  Tool config:  {args.tool_config}")
    print("======================================================================\n")

    # ── Load baseline ────────────────────────────────────────────────
    baseline_path = Path(args.baseline)
    if baseline_path.is_dir():
        files = list(baseline_path.glob("baseline_*.json"))
        if not files:
            raise FileNotFoundError(f"No baseline files in {baseline_path}")
        baseline_path = files[0]

    records = json.loads(baseline_path.read_text())
    print(f"Loaded {len(records)} baseline records\n")

    if args.max_samples:
        records = records[:args.max_samples * 3]
        print(f"Limited to {len(records)} records ({args.max_samples} samples)")

    # ── Load model ───────────────────────────────────────────────────
    cache_dir = os.environ.get("TRANSFORMERS_CACHE",
                               os.environ.get("HF_HOME", None))

    from vllm_coordinator_extras import VLLMReactCoordinator
    print(f"  Backend: vLLM server ({args.vllm_base_url or 'env/localhost'})")
    coordinator = VLLMReactCoordinator(
        model_name=args.model_name,
        base_url=args.vllm_base_url,
        max_new_tokens=1024,
        temperature=args.temperature,
    )

    # ── Cultural helper ──────────────────────────────────────────────
    from tool_registry import create_registry_with_all_tools
    from cultural_helper import CulturalHelper

    tool_registry = create_registry_with_all_tools()

    cultural_helper = CulturalHelper(
        tool_registry=tool_registry,
        tool_config=args.tool_config,
    )

    # ── Collector ────────────────────────────────────────────────────
    collector = ReactTrajectoryCollector(
        coordinator=coordinator,
        cultural_helper=cultural_helper,
        baseline_records=records,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
    )

    # ── Resume ───────────────────────────────────────────────────────
    start_idx = 0
    if args.resume:
        if args.resume_from:
            start_idx = collector.load_checkpoint(Path(args.resume_from))
        else:
            cp = collector.find_latest_checkpoint()
            if cp:
                start_idx = collector.load_checkpoint(cp)

    # ── Run ──────────────────────────────────────────────────────────
    print(f"\n  Model:        {args.model_name}")
    print(f"  Method:       ReAct")
    print(f"  Tool config:  {args.tool_config}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Output:       {args.output_dir}")

    collector.collect_all(start_idx)


if __name__ == "__main__":
    main()