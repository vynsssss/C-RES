# -*- coding: utf-8 -*-
"""
STEP 2 AGENTIC: Trajectory Collection with LLM Tool Selection

LLM autonomously decides which tools to use, then decides whether
the retrieved evidence is actually useful.

Two modes:
  --agentic-mode free    LLM selects 0-3 tools
  --agentic-mode single  LLM selects exactly 1 tool

Flow per baseline record:
  Turn 1 — coordinator.select_tools()            → LLM picks tools
  Turn 2 — cultural_helper.get_evidence_for_tools() → call only selected tools
  Turn 3 — coordinator.synthesize_agentic()      → LLM synthesizes + decides evidence usage

Output trajectories include all standard Step 2 fields PLUS:
  - agentic_mode
  - selected_tools        (what LLM chose in Turn 1)
  - tools_retrieved       (what actually returned data)
  - evidence_used         (LLM's decision in Turn 3)
  - selection_reasoning   (LLM's explanation for tool choice)

"""

import argparse
import json
import os
import re
import sys
import random
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import torch
except ModuleNotFoundError:  # vLLM path doesn't need torch
    torch = None
from coordinator import Coordinator   # top-level — not inside hot-path function


# ── Shared utility ─────────────────────────────────────────────────────────────
def extract_model_identifier(model_name: str) -> str:
    name_lower = model_name.lower()
    families = ["qwen", "llama", "olmo", "mistral", "gemma"]
    family = next((f for f in families if f in name_lower), None)
    if family is None:
        parts = model_name.split("/")
        family = parts[-1].split("-")[0].lower()
    size_match = re.search(r"(\d+\.?\d*)\s*[bB]", model_name)
    size = f"{size_match.group(1)}b" if size_match else "base"
    identifier = f"{family}_{size}"
    if "think" in name_lower:
        identifier += "_think"
    return identifier


def find_baseline_file(baseline_dir: str, data_id: str, model_name: str) -> Optional[str]:
    bdir = Path(baseline_dir)
    if not bdir.exists():
        return None
    model_id = extract_model_identifier(model_name)
    exact = bdir / f"baseline_{data_id}_{model_id}.json"
    if exact.exists():
        return str(exact)
    matches = sorted(bdir.glob(f"baseline_{data_id}_*.json"))
    if matches:
        print(f"[WARNING] Exact baseline not found for {model_id}, using: {matches[0].name}")
        return str(matches[0])
    return None


# ── Trajectory builder ────────────────────────────────────────────────────────
def build_agentic_trajectory(
    sample: Dict[str, Any],
    prompt_type: str,
    agentic_mode: str,
    # Turn 1
    selected_tools: List[str],
    selection_reasoning: str,
    raw_tool_selection: str,
    # Turn 2
    hofstede_data: Optional[Dict],
    cultural_atlas_data: Optional[Dict],
    wikipedia_data: Optional[Dict],
    tools_retrieved: List[str],
    # Turn 3
    baseline_reasoning: str,
    baseline_answer: str,
    updated_reasoning: str,
    final_answer: str,
    answer_changed: bool,
    evidence_used: bool,
    raw_response: str = "",
    output_status: str = "ok",
    # ── Reading-turn parity fields (added for fair comparison vs static) ──
    reading_status: Optional[Dict[str, str]] = None,
    reading_outputs: Optional[Dict[str, str]] = None,
    wiki_truncation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a complete agentic trajectory record.
    """
    gold_answer      = sample.get("answer", "")
    is_correct       = Coordinator.is_correct(final_answer, gold_answer)
    baseline_correct = Coordinator.is_correct(baseline_answer, gold_answer)

    return {
        # Sample metadata
        "sample_id":   sample.get("id", ""),
        "country":     sample.get("country", ""),
        "story":       sample.get("story", ""),
        "gold_answer": gold_answer,
        "prompt_type": prompt_type,

        # Agentic metadata
        "agentic_mode": agentic_mode,

        # Step 1: baseline
        "baseline_reasoning": baseline_reasoning,
        "baseline_answer":    baseline_answer,
        "baseline_correct":   baseline_correct,

        # Turn 1: tool selection
        "selected_tools":      selected_tools,
        "selection_reasoning": selection_reasoning,
        "raw_tool_selection":  raw_tool_selection,

        # Turn 2: retrieved evidence (raw, same format as standard Step 2)
        "hofstede_data":       hofstede_data,
        "cultural_atlas_data": cultural_atlas_data,
        "wikipedia_data":      wikipedia_data,
        "tools_retrieved":     tools_retrieved,

        # Turn 3: synthesis
        "updated_reasoning": updated_reasoning,
        "final_answer":      final_answer,
        "raw_response":      raw_response,
        "output_status":     output_status,
        "is_correct":        is_correct,
        "answer_changed":    answer_changed,
        "evidence_used":     evidence_used,

        # Reading-turn parity fields (mirror static synthesize trajectories)
        "reading_status":    reading_status or {},
        "reading_outputs":   reading_outputs or {},
        "wiki_truncation":   wiki_truncation or {},

        # Analysis flags
        "improved":  not baseline_correct and is_correct,
        "regressed": baseline_correct and not is_correct,

        "timestamp": datetime.now().isoformat(),
    }


# ── Collector ─────────────────────────────────────────────────────────────────
PROMPT_TYPES = ["story_country", "story_country_value", "story_rot"]


class AgenticTrajectoryCollector:
    """Collects agentic trajectories where LLM picks its own tools."""

    def __init__(
        self,
        coordinator,
        cultural_helper,
        baseline_records: List[Dict[str, Any]],
        output_dir: str,
        agentic_mode: str = "free",
        checkpoint_interval: int = 100,
    ):
        self.coordinator = coordinator
        self.cultural_helper = cultural_helper
        self.records = baseline_records
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.agentic_mode = agentic_mode
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
            self.stats_by_prompt[pt]["total"]   = len(trajs)
            self.stats_by_prompt[pt]["correct"] = sum(
                1 for t in trajs if t.get("is_correct")
            )

        completed = data["completed"]
        print(f"  Restored {completed}/{data['total']} — resuming from {completed + 1}")
        return completed

    def save_checkpoint(self, completed: int) -> None:
        """
        Atomically save checkpoint using tmp-file + rename.
        """
        path = self.output_dir / f"checkpoint_{completed}.json"
        tmp  = path.with_suffix(".tmp")
        data = {
            "completed":    completed,
            "total":        len(self.records),
            "trajectories": self.trajectories,
            "stats":        dict(self.stats),
            "agentic_mode": self.agentic_mode,
            "timestamp":    datetime.now().isoformat(),
        }
        try:
            tmp.write_text(json.dumps(data))   # compact — transient file
            tmp.replace(path)                   # atomic rename on POSIX
            print(f"\n[CHECKPOINT] {path} ({completed}/{len(self.records)})")
        except Exception as e:
            print(f"\n[CHECKPOINT FAILED] {e}")
            if tmp.exists():
                tmp.unlink()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def collect_all(self, start_idx: int = 0) -> None:
        total = len(self.records)
        print(f"\nAgentic mode: {self.agentic_mode.upper()}")
        print(f"Processing {total - start_idx} records (start={start_idx})")
        print("=" * 70)

        for idx in range(start_idx, total):
            rec        = self.records[idx]
            sample_id  = rec.get("sample_id", "")
            prompt_type = rec.get("prompt_type", "")

            print(f"\n[{idx + 1}/{total}] {sample_id} — {prompt_type}")

            if "error" in rec:
                print("  SKIP (Step 1 error)")
                self.stats["skipped"] += 1
                continue

            baseline_reasoning = rec.get("reasoning", "")
            baseline_answer    = rec.get("decision", "")

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
                # ── Turn 1: LLM selects tools ──────────────────────────────
                selection = self.coordinator.select_tools(
                    prompt_type=prompt_type,
                    story=sample["story"],
                    country=sample["country"],
                    mode=self.agentic_mode,
                    value=sample["value"],
                    rot=sample["rot"],
                )
                selected_tools = selection["selected_tools"]
                print(f"  Turn 1 — selected: {selected_tools}")

                # ── Turn 2: call only selected tools ───────────────────────
                evidence = self._get_evidence_for_tools(sample, selected_tools)
                print(f"  Turn 2 — retrieved: {evidence['tools_retrieved']}")

                # ── Turn 3: synthesize + decide evidence usage ─────────────
                synth = self.coordinator.synthesize_agentic(
                    mode=self.agentic_mode,
                    prompt_type=prompt_type,
                    story=sample["story"],
                    country=sample["country"],
                    baseline_reasoning=baseline_reasoning,
                    baseline_answer=baseline_answer,
                    selected_tools=selected_tools,
                    hofstede_data=evidence["hofstede_data"],
                    cultural_atlas_data=evidence["cultural_atlas_data"],
                    wikipedia_data=evidence["wikipedia_data"],
                    value=sample["value"],
                    rot=sample["rot"],
                )

                traj = build_agentic_trajectory(
                    sample=sample,
                    prompt_type=prompt_type,
                    agentic_mode=self.agentic_mode,
                    selected_tools=selected_tools,
                    selection_reasoning=selection["selection_reasoning"],
                    raw_tool_selection=selection["raw_response"],
                    hofstede_data=evidence["hofstede_data"],
                    cultural_atlas_data=evidence["cultural_atlas_data"],
                    wikipedia_data=evidence["wikipedia_data"],
                    tools_retrieved=synth["tools_retrieved"],
                    baseline_reasoning=baseline_reasoning,
                    baseline_answer=baseline_answer,
                    updated_reasoning=synth["reasoning"],
                    final_answer=synth["decision"],
                    answer_changed=synth["answer_changed"],
                    evidence_used=synth["evidence_used"],
                    raw_response=synth["raw_response"],
                    output_status=synth.get("output_status", "ok"),
                    reading_status=synth.get("reading_status", {}),
                    reading_outputs=synth.get("reading_outputs", {}),
                    wiki_truncation=synth.get("wiki_truncation", {}),
                )

                self._record_trajectory(traj, prompt_type)

                out_status = synth.get("output_status", "ok")
                if out_status in ("degenerated", "empty"):
                    print(
                        f"  ⚠ OUTPUT_{out_status.upper()} base={baseline_answer}"
                        f" (gold={sample['answer']})"
                        f" selected={selected_tools}"
                    )
                else:
                    status  = "✓" if traj["is_correct"] else "✗"
                    changed = " CHANGED" if traj["answer_changed"] else ""
                    used    = " USED_EVIDENCE" if traj["evidence_used"] else " IGNORED_EVIDENCE"
                    print(
                        f"  {status} base={baseline_answer} → final={synth['decision']}"
                        f" (gold={sample['answer']}){changed}{used}"
                    )

            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()   # top-level import — no per-call overhead

                error_traj = {
                    "sample_id":    sample_id,
                    "prompt_type":  prompt_type,
                    "agentic_mode": self.agentic_mode,
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

    def _get_evidence_for_tools(
        self, sample: Dict[str, Any], selected_tools: List[str]
    ) -> Dict[str, Any]:
        """Call only the tools the LLM selected in Turn 1."""
        hofstede_data       = None
        cultural_atlas_data = None
        wikipedia_data      = None
        tools_retrieved: List[str] = []

        country = self.cultural_helper._normalize_country(sample.get("country", ""))
        story   = sample.get("story", "")

        if "hofstede_tool" in selected_tools:
            hofstede_data = self.cultural_helper._call_tool("hofstede_tool", country=country)
            if hofstede_data and hofstede_data.get("success"):
                tools_retrieved.append("hofstede_tool")
                self.cultural_helper.hofstede_calls += 1

        if "cultural_atlas_tool" in selected_tools:
            cultural_atlas_data = self.cultural_helper._call_tool("cultural_atlas_tool", country=country)
            if cultural_atlas_data and cultural_atlas_data.get("retrieved"):
                tools_retrieved.append("cultural_atlas_tool")
                self.cultural_helper.cultural_atlas_calls += 1

        if "wikipedia_rag" in selected_tools:
            wikipedia_data = self.cultural_helper._call_tool("wikipedia_rag", country=country, query=story)
            if wikipedia_data and wikipedia_data.get("retrieved"):
                tools_retrieved.append("wikipedia_rag")
                self.cultural_helper.wikipedia_calls += 1

        self.cultural_helper.total_calls += 1

        return {
            "hofstede_data":       hofstede_data,
            "cultural_atlas_data": cultural_atlas_data,
            "wikipedia_data":      wikipedia_data,
            "tools_retrieved":     tools_retrieved,
        }

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
            if traj.get("improved"):
                self.stats["improved"] += 1
            if traj.get("regressed"):
                self.stats["regressed"] += 1
            if traj.get("evidence_used"):
                self.stats["evidence_used"] += 1

    def _print_stats(self) -> None:
        total = self.stats["total"]
        if total == 0:
            print("\nNo trajectories collected.")
            return

        print(f"\n{'=' * 70}")
        print(f"AGENTIC RESULTS — mode={self.agentic_mode} ({total} trajectories)")
        print(f"{'=' * 70}")
        correct = self.stats["correct"]
        print(f"  Accuracy:        {correct}/{total} ({correct/total:.1%})")
        print(f"  Changed:         {self.stats['changed']}")
        print(f"  Improved:        {self.stats['improved']}")
        print(f"  Regressed:       {self.stats['regressed']}")
        ev = self.stats["evidence_used"]
        print(f"  Evidence used:   {ev}/{total} ({ev/total:.1%})")
        print(f"  Errors:          {self.stats['errors']}")
        print(f"  Skipped:         {self.stats['skipped']}")

        for pt in PROMPT_TYPES:
            pt_s     = self.stats_by_prompt[pt]
            pt_total = pt_s.get("total", 0)
            if pt_total > 0:
                pt_correct = pt_s.get("correct", 0)
                print(f"  {pt}: {pt_correct}/{pt_total} ({pt_correct/pt_total:.1%})")

        # Tool selection frequency
        tool_counts: Dict[str, int] = defaultdict(int)
        for t in self.trajectories:
            for tool in t.get("selected_tools", []):
                tool_counts[tool] += 1
        if tool_counts:
            print(f"\n  Tool selection frequency (out of {total}):")
            for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
                print(f"    {tool}: {count} ({count/total:.1%})")
        none_count = sum(1 for t in self.trajectories if not t.get("selected_tools"))
        if none_count:
            print(f"    none selected: {none_count} ({none_count/total:.1%})")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_results(self) -> None:
        combined = self.output_dir / "trajectories_all_prompts.json"
        combined.write_text(json.dumps(self.trajectories, indent=2))
        print(f"\n[SAVED] {combined} ({len(self.trajectories)} trajectories)")

        for pt in PROMPT_TYPES:
            path = self.output_dir / f"trajectories_{pt}.json"
            path.write_text(json.dumps(self.by_prompt[pt], indent=2))
            print(f"[SAVED] {path} ({len(self.by_prompt[pt])})")

        stats_path = self.output_dir / "collection_stats.json"
        tool_counts: Dict[str, int] = defaultdict(int)
        for t in self.trajectories:
            for tool in t.get("selected_tools", []):
                tool_counts[tool] += 1

        sp = self.coordinator._sampling_params
        stats_data = {
            "overall":             dict(self.stats),
            "by_prompt":           {p: dict(s) for p, s in self.stats_by_prompt.items()},
            "agentic_mode":        self.agentic_mode,
            "tool_selection_freq": dict(tool_counts),
            "model":               self.coordinator.model_name,
            "sampling": {
                "temperature": sp.get("temperature", 0.0),
                "do_sample":   sp.get("do_sample", False),
                "top_p":       sp.get("top_p"),
                "top_k":       sp.get("top_k"),
                "min_p":       sp.get("min_p"),
            },
            "model_stats":         self.coordinator.get_statistics(),
            "timestamp":           datetime.now().isoformat(),
        }
        stats_path.write_text(json.dumps(stats_data, indent=2))
        print(f"[SAVED] {stats_path}")


# ── main() ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2 Agentic: LLM selects its own tools"
    )

    parser.add_argument("--baseline",     type=str, default=None)
    parser.add_argument("--baseline-dir", type=str, default=None)
    parser.add_argument("--data-id",      type=str, default=None)
    parser.add_argument("--output-dir",   type=str, required=True)
    parser.add_argument("--model-name",   type=str, required=True)
    parser.add_argument("--agentic-mode", type=str, default="free",
                        choices=["free", "single"],
                        help="free: LLM selects 0-3 tools | single: LLM selects exactly 1 tool")
    parser.add_argument("--device",              type=str,   default="cuda")
    parser.add_argument("--checkpoint-interval", type=int,   default=100)
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--resume-from",  type=str, default=None)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--temperature",  type=float, default=0.0,
                        help="Sampling temperature (0.0=greedy). "
                             "top_p/top_k/min_p auto-resolved per model.")
    parser.add_argument("--max-samples",  type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8192,
                        help="Max tokens generated per turn. Default 8192. "
                             "Use 2048 for OLMo-7B, 4096 for Think models.")
    parser.add_argument("--planned",      action="store_true",
                        help="Use 3-step planned synthesis (A→B→C) instead of monolithic")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm"],
                        help="vllm: use local vLLM server (skip transformers load)")
    parser.add_argument("--vllm-base-url", type=str, default=None)

    args = parser.parse_args()

    # ── Resolve baseline ────────────────────────────────────────────────
    if args.baseline:
        baseline_file = args.baseline
    elif args.baseline_dir and args.data_id:
        baseline_file = find_baseline_file(
            args.baseline_dir, args.data_id, args.model_name
        )
        if baseline_file is None:
            model_id = extract_model_identifier(args.model_name)
            print(f"ERROR: baseline not found for {model_id} in {args.baseline_dir}")
            sys.exit(1)
    else:
        print("ERROR: provide --baseline <path>  OR  --baseline-dir + --data-id")
        sys.exit(1)

    # ── Environment ─────────────────────────────────────────────────────
    cache_dir = os.environ.get("SCRATCHDIR", "/scratch") + "/.cache/huggingface"
    os.environ.setdefault("HF_HOME",           cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{cache_dir}/transformers")
    os.environ.setdefault("HF_DATASETS_CACHE",  f"{cache_dir}/datasets")

    random.seed(args.seed)
    if torch is not None:
        torch.manual_seed(args.seed)
    if torch is not None and torch.cuda.is_available():
        (torch.cuda.manual_seed_all(args.seed) if torch is not None and torch.cuda.is_available() else None)

    print(f"\n{'=' * 70}")
    print("STEP 2 AGENTIC: TRAJECTORY COLLECTION")
    print(f"{'=' * 70}")
    print(f"  Model:        {args.model_name}")
    print(f"  Agentic mode: {args.agentic_mode}")
    print(f"  Baseline:     {baseline_file}")
    print(f"  Output:       {args.output_dir}")
    print(f"{'=' * 70}\n")

    # ── Load baseline ────────────────────────────────────────────────────
    with open(baseline_file) as f:
        records = json.load(f)
    if args.max_samples:
        records = records[:args.max_samples]
        print(f"Loaded {len(records)} baseline records (LIMITED)")
    else:
        print(f"Loaded {len(records)} baseline records")

    error_count = sum(1 for r in records if "error" in r)
    if error_count:
        print(f"  ({error_count} error records will be skipped)")

    # ── Set up tools (backend-independent) ───────────────────────────────
    from tool_registry import create_registry_with_all_tools
    from cultural_helper import CulturalHelper

    tool_registry = create_registry_with_all_tools()
    cultural_helper = CulturalHelper(
        tool_registry=tool_registry,
        tool_config="all",
    )

    if args.planned:
        raise SystemExit("--planned agentic is not vLLM-wired; use --backend transformers")
    from vllm_coordinator_extras import VLLMAgenticCoordinator
    print(f"  Backend: vLLM server ({args.vllm_base_url or 'env/localhost'})")
    coordinator = VLLMAgenticCoordinator(
        model_name=args.model_name,
        base_url=args.vllm_base_url,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tool_registry=tool_registry,
    )

    # ── Collect ──────────────────────────────────────────────────────────
    collector = AgenticTrajectoryCollector(
        coordinator=coordinator,
        cultural_helper=cultural_helper,
        baseline_records=records,
        output_dir=args.output_dir,
        agentic_mode=args.agentic_mode,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Already complete?
    final_file = Path(args.output_dir) / "trajectories_all_prompts.json"
    if final_file.exists():
        try:
            existing = json.loads(final_file.read_text())
            count = (
                len(existing)
                if isinstance(existing, list)
                else len(existing.get("trajectories", []))
            )
            print(f"\n[ALREADY COMPLETE] {final_file}")
            print(f"  Found {count} trajectories — nothing to do.")
            print(f"  Delete the output dir to force rerun.")
            sys.exit(0)
        except Exception:
            print("[WARNING] Final file exists but unreadable — resuming from checkpoint")

    # Resume from checkpoint
    start_idx = 0
    if args.resume_from:
        cp = Path(args.resume_from)
        if cp.exists():
            start_idx = collector.load_checkpoint(cp)
        else:
            print(f"Checkpoint not found: {cp}, starting from 0")
    elif args.resume:
        latest = collector.find_latest_checkpoint()
        if latest:
            start_idx = collector.load_checkpoint(latest)
        else:
            print(f"No checkpoint in {args.output_dir}, starting from 0")

    collector.collect_all(start_idx=start_idx)
    collector.save_results()

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("STEP 2 AGENTIC COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Trajectories: {len(collector.trajectories)}")
    print(f"  Agentic mode: {args.agentic_mode}")
    coord_stats = coordinator.get_statistics()
    print(f"  Generations:  {coord_stats['generations']} "
          f"(2 per sample: selection + synthesis)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()