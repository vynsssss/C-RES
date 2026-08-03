# -*- coding: utf-8 -*-
"""
STEP 2: Trajectory Collection

Reads Step 1 baseline results, gathers cultural evidence with tools,
and synthesizes a final answer. 1 model generation per sample (synthesis only).

Flow per baseline record:
  1. Read baseline_reasoning + baseline_answer from Step 1 JSON
  2. cultural_helper.get_evidence(sample) -> calls tools, returns raw dicts
  3. coordinator.synthesize(baseline, tool_data) -> model.generate() -> final answer
  4. trajectory_builder.build_trajectory() -> structured output record

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

import torch


# ------------------------------------------------------------------
# Shared utility -- same logic as Step 1 (must produce identical model_ids)
# ------------------------------------------------------------------
def extract_model_identifier(model_name: str) -> str:
    """
    Extract short identifier from full model name.
    e.g. 'Qwen/Qwen3-4B-Instruct-2507'  -> 'qwen_4b'
         'allenai/Olmo-3-7B-Think'        -> 'olmo_7b_think'
    """
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


def find_baseline_file(
    baseline_dir: str, data_id: str, model_name: str
) -> Optional[str]:
    """Find baseline JSON from Step 1 in baseline_dir."""
    bdir = Path(baseline_dir)
    if not bdir.exists():
        return None
    model_id = extract_model_identifier(model_name)
    exact = bdir / f"baseline_{data_id}_{model_id}.json"
    if exact.exists():
        return str(exact)
    matches = sorted(bdir.glob(f"baseline_{data_id}_*.json"))
    if matches:
        print(f"[WARNING] Exact baseline not found for {model_id}")
        print(f"[WARNING] Using closest match: {matches[0].name}")
        return str(matches[0])
    return None


# ------------------------------------------------------------------
# Trajectory Collector
# ------------------------------------------------------------------
PROMPT_TYPES = ["story_country", "story_country_value", "story_rot"]

# Canonical answer set -- anything outside this becomes "unknown" for Step 2.
_VALID_BASELINE_ANSWERS = frozenset({"yes", "no", "neutral"})


class TrajectoryCollector:
    """
    Iterates Step 1 baseline records, gathers cultural evidence,
    synthesizes with coordinator, and builds trajectory records.
    """

    def __init__(
        self,
        coordinator,
        cultural_helper,
        baseline_records: List[Dict[str, Any]],
        output_dir: str,
        tool_config: str = "all",
        checkpoint_interval: int = 100,
    ):
        self.coordinator = coordinator
        self.cultural_helper = cultural_helper
        self.records = baseline_records
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tool_config = tool_config
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
        print(f"  Restored {completed}/{data['total']} -- resuming from {completed + 1}")
        return completed

    def save_checkpoint(self, completed: int) -> None:
        """
        Atomically save checkpoint using tmp-file + rename.
        """
        path = self.output_dir / f"checkpoint_{completed}.json"
        tmp  = path.with_suffix(".tmp")
        data = {
            "completed":  completed,
            "total":      len(self.records),
            "trajectories": self.trajectories,
            "stats":      dict(self.stats),
            "tool_config": self.tool_config,
            "timestamp":  datetime.now().isoformat(),
        }
        try:
            tmp.write_text(json.dumps(data))  # compact -- transient file
            tmp.replace(path)                 # atomic rename on POSIX
            print(f"\n[CHECKPOINT] {path} ({completed}/{len(self.records)})")
        except Exception as e:
            print(f"\n[CHECKPOINT FAILED] {e}")
            if tmp.exists():
                tmp.unlink()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def collect_all(self, start_idx: int = 0) -> None:
        """Process all baseline records from start_idx."""
        from trajectory_builder import build_trajectory

        total = len(self.records)
        print(f"\nProcessing {total - start_idx} records (start={start_idx})")
        print("=" * 70)

        for idx in range(start_idx, total):
            rec = self.records[idx]
            sample_id  = rec.get("sample_id", "")
            prompt_type = rec.get("prompt_type", "")

            print(f"\n[{idx + 1}/{total}] {sample_id} -- {prompt_type}")

            if "error" in rec:
                err_msg = str(rec.get("error", ""))[:80]
                print(f"  SKIP (Step 1 error: {err_msg})")
                self.stats["skipped"] += 1
                continue

            # Fill placeholders for missing/garbled baseline output.
            # Step 2 generates its own reasoning anyway, so empty reasoning
            # is harmless. A non-canonical decision (e.g. 'empty', 'unknown',
            # 'degenerated') becomes 'unknown' for the synthesis prompt.
            baseline_reasoning = (
                rec.get("reasoning", "") or "(baseline produced no reasoning)"
            )
            baseline_answer = rec.get("decision", "") or "unknown"
            if baseline_answer not in _VALID_BASELINE_ANSWERS:
                baseline_answer = "unknown"

            sample = {
                "id":      sample_id,
                "country": rec.get("country", ""),
                "story":   rec.get("story", ""),
                "value":   rec.get("value", ""),
                "rot":     rec.get("rot", ""),
                "answer":  rec.get("gold_answer", ""),
            }

            try:
                # Step 2a: gather cultural evidence (no model generation)
                evidence = self.cultural_helper.get_evidence(
                    sample, tool_config=self.tool_config
                )

                # Step 2b: synthesize with model (1 generation)
                synth = self.coordinator.synthesize(
                    prompt_type=prompt_type,
                    story=sample.get("story", ""),
                    country=sample.get("country", ""),
                    baseline_reasoning=baseline_reasoning,
                    baseline_answer=baseline_answer,
                    hofstede_data=evidence["hofstede_data"],
                    cultural_atlas_data=evidence["cultural_atlas_data"],
                    wikipedia_data=evidence["wikipedia_data"],
                    value=sample.get("value", ""),
                    rot=sample.get("rot", ""),
                )

                # Step 2c: build trajectory record
                traj = build_trajectory(
                    sample=sample,
                    prompt_type=prompt_type,
                    baseline_reasoning=baseline_reasoning,
                    baseline_answer=baseline_answer,
                    hofstede_data=evidence["hofstede_data"],
                    cultural_atlas_data=evidence["cultural_atlas_data"],
                    wikipedia_data=evidence["wikipedia_data"],
                    tools_used=evidence["tools_used"],
                    updated_reasoning=synth["reasoning"],
                    final_answer=synth["decision"],
                    answer_changed=synth["answer_changed"],
                    raw_response=synth["raw_response"],
                    output_status=synth.get("output_status", "ok"),
                    reading_status=synth.get("reading_status"),
                    reading_outputs=synth.get("reading_outputs"),
                    wiki_truncation=synth.get("wiki_truncation"),
                )

                self._record_trajectory(traj, prompt_type)

                out_status = synth.get("output_status", "ok")
                if out_status in ("degenerated", "empty"):
                    print(
                        f"  ! OUTPUT_{out_status.upper()} base={baseline_answer}"
                        f" (gold={sample['answer']})"
                        f" tools={evidence['tools_used']}"
                    )
                else:
                    status  = "[OK]" if traj["is_correct"] else "[X]"
                    changed = " CHANGED" if traj["answer_changed"] else ""
                    print(
                        f"  {status} base={baseline_answer} -> final={synth['decision']}"
                        f" (gold={sample['answer']}){changed}"
                        f" tools={evidence['tools_used']}"
                    )

            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()

                error_traj = {
                    "sample_id":   sample_id,
                    "prompt_type": prompt_type,
                    "error":       str(e),
                    "is_correct":  False,
                    "timestamp":   datetime.now().isoformat(),
                }
                self._record_trajectory(error_traj, prompt_type, is_error=True)

            # Periodic checkpoint + GPU cache flush
            completed = idx + 1
            if completed % self.checkpoint_interval == 0:
                self.save_checkpoint(completed)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self._print_stats()

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

    def _print_stats(self) -> None:
        total = self.stats["total"]
        if total == 0:
            print("\nNo trajectories collected.")
            return

        print(f"\n{'=' * 70}")
        print(f"RESULTS ({total} trajectories)")
        print(f"{'=' * 70}")
        correct = self.stats["correct"]
        print(f"  Accuracy:   {correct}/{total} ({correct/total:.1%})")
        print(f"  Changed:    {self.stats['changed']}")
        print(f"  Improved:   {self.stats['improved']}")
        print(f"  Regressed:  {self.stats['regressed']}")
        print(f"  Errors:     {self.stats['errors']}")
        print(f"  Skipped:    {self.stats['skipped']}")

        for pt in PROMPT_TYPES:
            pt_s    = self.stats_by_prompt[pt]
            pt_total = pt_s.get("total", 0)
            if pt_total > 0:
                pt_correct = pt_s.get("correct", 0)
                print(f"  {pt}: {pt_correct}/{pt_total} ({pt_correct/pt_total:.1%})")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_results(self) -> None:
        """Save trajectories: combined + per-prompt + stats."""
        combined = self.output_dir / "trajectories_all_prompts.json"
        combined.write_text(json.dumps(self.trajectories, indent=2))
        print(f"\n[SAVED] {combined} ({len(self.trajectories)} trajectories)")

        for pt in PROMPT_TYPES:
            path = self.output_dir / f"trajectories_{pt}.json"
            path.write_text(json.dumps(self.by_prompt[pt], indent=2))
            print(f"[SAVED] {path} ({len(self.by_prompt[pt])})")

        stats_path = self.output_dir / "collection_stats.json"
        sp = self.coordinator._sampling_params
        stats_data = {
            "overall":     dict(self.stats),
            "by_prompt":   {p: dict(s) for p, s in self.stats_by_prompt.items()},
            "tool_config": self.tool_config,
            "model":       self.coordinator.model_name,
            "sampling": {
                "temperature": sp.get("temperature", 0.0),
                "do_sample":   sp.get("do_sample", False),
                "top_p":       sp.get("top_p"),
                "top_k":       sp.get("top_k"),
                "min_p":       sp.get("min_p"),
            },
            "model_stats": self.coordinator.get_statistics(),
            "timestamp":   datetime.now().isoformat(),
        }
        stats_path.write_text(json.dumps(stats_data, indent=2))
        print(f"[SAVED] {stats_path}")


# ------------------------------------------------------------------
# main()
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: Collect trajectories with cultural evidence",
        allow_abbrev=False
    )

    # Baseline source (pick one)
    parser.add_argument("--baseline",     type=str, default=None,
                        help="Direct path to Step 1 baseline JSON")
    parser.add_argument("--baseline-dir", type=str, default=None,
                        help="Directory containing baselines (auto-find mode)")
    parser.add_argument("--data-id",      type=str, default=None,
                        help="Dataset identifier for auto-find (e.g. normad_dataset)")

    # Required
    parser.add_argument("--output-dir",  type=str, required=True)
    parser.add_argument("--model-name",  type=str, required=True)

    # Optional
    parser.add_argument("--device",               type=str,   default="cuda")
    parser.add_argument("--checkpoint-interval",  type=int,   default=100)
    parser.add_argument("--tool-config", type=str, default="all",
                        choices=["hofstede", "atlas", "wiki",
                                 "hofstede_atlas", "hofstede_wiki", "atlas_wiki", "all"])
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--resume-from",  type=str, default=None)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--temperature",  type=float, default=0.0,
                        help="Sampling temperature (0.0=greedy). "
                             "top_p/top_k/min_p auto-resolved per model.")
    parser.add_argument("--max-samples",  type=int, default=None,
                        help="Process only first N records (for testing)")
    parser.add_argument("--max-new-tokens", type=int, default=8192,
                        help="Max new tokens for generation (default 8192)")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm"],
                        help="Generation backend. 'vllm' talks to a local vLLM "
                             "OpenAI server (see serve_vllm.sh); skips the "
                             "in-process transformers load.")
    parser.add_argument("--vllm-base-url", type=str, default=None,
                        help="Base URL of the local vLLM server "
                             "(default http://localhost:8000 or $VLLM_BASE_URL)")

    args = parser.parse_args()

    # -- Resolve baseline file ----------------------------------------------
    if args.baseline:
        baseline_file = args.baseline
        print(f"Using baseline: {baseline_file}")
    elif args.baseline_dir and args.data_id:
        baseline_file = find_baseline_file(
            args.baseline_dir, args.data_id, args.model_name
        )
        if baseline_file is None:
            model_id = extract_model_identifier(args.model_name)
            print(f"ERROR: baseline not found")
            print(f"  Looking for: baseline_{args.data_id}_{model_id}.json")
            print(f"  In: {args.baseline_dir}")
            bdir = Path(args.baseline_dir)
            if bdir.exists():
                print("\nAvailable:")
                for f in sorted(bdir.glob("baseline_*.json")):
                    print(f"    {f.name}")
            sys.exit(1)
        print(f"Auto-found baseline: {baseline_file}")
    else:
        print("ERROR: provide --baseline <path>  OR  --baseline-dir + --data-id")
        sys.exit(1)

    # -- Environment --------------------------------------------------------
    cache_dir = os.environ.get("SCRATCHDIR", "/mnt/parscratch/users/acp24vp") + "/.cache/huggingface"
    os.environ.setdefault("HF_HOME",           cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{cache_dir}/transformers")
    os.environ.setdefault("HF_DATASETS_CACHE",  f"{cache_dir}/datasets")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"\n{'=' * 70}")
    print("STEP 2: TRAJECTORY COLLECTION")
    print(f"{'=' * 70}")
    print(f"  Model:       {args.model_name}")
    print(f"  Tool config: {args.tool_config}")
    print(f"  Baseline:    {baseline_file}")
    print(f"  Output:      {args.output_dir}")
    print(f"{'=' * 70}\n")

    # -- Load baseline ------------------------------------------------------
    with open(baseline_file) as f:
        records = json.load(f)
    if args.max_samples:
        records = records[:args.max_samples]
        print(f"Loaded {len(records)} baseline records (LIMITED for testing)")
    else:
        print(f"Loaded {len(records)} baseline records")

    error_count = sum(1 for r in records if "error" in r)
    if error_count:
        print(f"  ({error_count} error records will be skipped)")

    # -- Set up tools (backend-independent) ---------------------------------
    from tool_registry import create_registry_with_all_tools
    from cultural_helper import CulturalHelper

    tool_registry = create_registry_with_all_tools()
    cultural_helper = CulturalHelper(
        tool_registry=tool_registry,
        tool_config=args.tool_config,
    )

    # -- Build coordinator per backend --------------------------------------
    from vllm_coordinator import VLLMCoordinator
    print(f"\nBackend: vLLM server ({args.vllm_base_url or 'env/localhost'})")
    coordinator = VLLMCoordinator(
        model_name=args.model_name,
        base_url=args.vllm_base_url,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tool_registry=tool_registry,
    )

    # -- Collect ------------------------------------------------------------
    collector = TrajectoryCollector(
        coordinator=coordinator,
        cultural_helper=cultural_helper,
        baseline_records=records,
        output_dir=args.output_dir,
        tool_config=args.tool_config,
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
            print(f"  Found {count} trajectories -- nothing to do.")
            print(f"  Delete the output dir to force rerun.")
            sys.exit(0)
        except Exception:
            print("[WARNING] Final file exists but unreadable -- resuming from checkpoint")

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

    # -- Final summary ------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("STEP 2 COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Trajectories: {len(collector.trajectories)}")
    print(f"  Tool config:  {args.tool_config}")

    helper_stats = cultural_helper.get_statistics()
    print(f"  Tool calls:   {helper_stats['total_calls']}")
    print(f"  Avg tools:    {helper_stats['avg_tools_per_call']:.1f}")

    coord_stats = coordinator.get_statistics()
    print(f"  Generations:  {coord_stats['generations']}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()