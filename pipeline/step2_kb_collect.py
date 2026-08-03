# -----------------------------------------------------------------------------
# NOTE: This KB / KB-selective baseline reproduces prior-work knowledge-base
# grounding (Lertvittayakumjorn et al.). It runs on the HuggingFace transformers
# backend. 
# -----------------------------------------------------------------------------

# -*- coding: utf-8 -*-
"""
STEP 2 (KB-GROUNDING, Option 1): Faithful single-shot KB-grounding collection.

Mirrors step2_collect_trajectories.py but replaces the cultural_helper +
synthesize() pipeline with the single-shot KBGroundingCoordinator. Output is
written through build_trajectory() into the SAME schema and filenames, so it
drops into analyze.py alongside the static/agentic/ReAct runs.

The Step 1 baseline is still read (per record) so baseline_correct / improved /
regressed are computed against the same baseline the other conditions use --
even though KB-grounding itself does not consume the baseline answer (it is
single-shot). This keeps the comparison apples-to-apples.

Flow per baseline record:
  1. Read baseline_reasoning + baseline_answer from Step 1 JSON (for flags only)
  2. coordinator.answer(prompt_type, story, country, value, rot) -> single-shot
  3. build_trajectory(...) -> structured record (same schema)

Usage:
  python step2_kb_collect.py \
      --baseline   $BASELINE_FILE \
      --output-dir $OUT \
      --model-name "Qwen/Qwen3-4B-Instruct-2507" \
      --kb-index   $SCRATCHDIR/CRES-paper_kb/qwen3_4b_1024 \
      [--selective] [--retrieve-n 5] [--resume]
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
from typing import Any, Dict, List, Optional

import torch

PROMPT_TYPES = ["story_country", "story_country_value", "story_rot"]
_VALID_BASELINE_ANSWERS = frozenset({"yes", "no", "neutral"})


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


class KBTrajectoryCollector:
    """Single-shot KB-grounding collector with the standard trajectory schema."""

    def __init__(
        self, coordinator, baseline_records, output_dir, tool_config_label,
        checkpoint_interval=100,
    ):
        self.coordinator = coordinator
        self.records = baseline_records
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tool_config = tool_config_label  # "kb" or "kb_selective" (for stats)
        self.checkpoint_interval = checkpoint_interval

        self.trajectories: List[Dict[str, Any]] = []
        self.by_prompt: Dict[str, List] = {p: [] for p in PROMPT_TYPES}
        self.stats = defaultdict(int)
        self.stats_by_prompt = {p: defaultdict(int) for p in PROMPT_TYPES}

    # ---- checkpoint/resume (same pattern as step2_collect_trajectories) ----
    def find_latest_checkpoint(self) -> Optional[Path]:
        cps = list(self.output_dir.glob("checkpoint_*.json"))
        if not cps:
            return None
        return max(cps, key=lambda p: int(re.search(r"checkpoint_(\d+)", p.name).group(1))
                   if re.search(r"checkpoint_(\d+)", p.name) else 0)

    def load_checkpoint(self, path: Path) -> int:
        print(f"\nLoading checkpoint: {path}")
        data = json.loads(path.read_text())
        self.trajectories = data["trajectories"]
        self.stats = defaultdict(int, data.get("stats", {}))
        for t in self.trajectories:
            pt = t.get("prompt_type", "")
            if pt in self.by_prompt:
                self.by_prompt[pt].append(t)
        completed = data["completed"]
        print(f"  Restored {completed}/{data['total']} -- resuming from {completed + 1}")
        return completed

    def save_checkpoint(self, completed: int) -> None:
        path = self.output_dir / f"checkpoint_{completed}.json"
        tmp = path.with_suffix(".tmp")
        data = {
            "completed": completed, "total": len(self.records),
            "trajectories": self.trajectories, "stats": dict(self.stats),
            "tool_config": self.tool_config, "timestamp": datetime.now().isoformat(),
        }
        try:
            tmp.write_text(json.dumps(data))
            tmp.replace(path)
            print(f"\n[CHECKPOINT] {path} ({completed}/{len(self.records)})")
        except Exception as e:
            print(f"\n[CHECKPOINT FAILED] {e}")
            if tmp.exists():
                tmp.unlink()

    # ---- main loop ----
    def collect_all(self, start_idx: int = 0) -> None:
        from trajectory_builder import build_trajectory

        total = len(self.records)
        print(f"\nProcessing {total - start_idx} records (start={start_idx})")
        print("=" * 70)

        for idx in range(start_idx, total):
            rec = self.records[idx]
            sample_id = rec.get("sample_id", "")
            prompt_type = rec.get("prompt_type", "")
            print(f"\n[{idx + 1}/{total}] {sample_id} -- {prompt_type}")

            if "error" in rec:
                print(f"  SKIP (Step 1 error)")
                self.stats["skipped"] += 1
                continue

            baseline_reasoning = rec.get("reasoning", "") or "(baseline produced no reasoning)"
            baseline_answer = rec.get("decision", "") or "unknown"
            if baseline_answer not in _VALID_BASELINE_ANSWERS:
                baseline_answer = "unknown"

            sample = {
                "id": sample_id, "country": rec.get("country", ""),
                "story": rec.get("story", ""), "value": rec.get("value", ""),
                "rot": rec.get("rot", ""), "answer": rec.get("gold_answer", ""),
            }

            try:
                # Single-shot KB-grounded answer (no baseline fed in).
                out = self.coordinator.answer(
                    prompt_type=prompt_type,
                    story=sample["story"], country=sample["country"],
                    value=sample["value"], rot=sample["rot"],
                    sample_id=sample_id,
                )

                # answer_changed: did KB-grounding differ from the Step 1 baseline?
                answer_changed = out["decision"] != baseline_answer.lower().strip()

                # KB-grounding has no Hofstede/Atlas/Wiki tool data; pass None.
                # tools_used records the KB source(s) actually retrieved.
                traj = build_trajectory(
                    sample=sample, prompt_type=prompt_type,
                    baseline_reasoning=baseline_reasoning,
                    baseline_answer=baseline_answer,
                    hofstede_data=None, cultural_atlas_data=None, wikipedia_data=None,
                    tools_used=["kb_grounding"] if out.get("kb_retrieved", 0) else [],
                    updated_reasoning=out["reasoning"],
                    final_answer=out["decision"],
                    answer_changed=answer_changed,
                    raw_response=out["raw_response"],
                    output_status=out.get("output_status", "ok"),
                )
                # Attach KB-specific diagnostics (extra keys are harmless to analyze.py).
                traj["kb_retrieved"] = out.get("kb_retrieved", 0)
                traj["kb_kept"] = out.get("kb_kept", 0)
                traj["kb_sources"] = out.get("kb_sources", [])
                traj["kb_selective"] = self.coordinator.selective

                self._record(traj, prompt_type)

                st = out.get("output_status", "ok")
                if st in ("degenerated", "empty"):
                    print(f"  ! OUTPUT_{st.upper()} (gold={sample['answer']})")
                else:
                    mark = "[OK]" if traj["is_correct"] else "[X]"
                    chg = " CHANGED" if answer_changed else ""
                    print(f"  {mark} base={baseline_answer} -> final={out['decision']} "
                          f"(gold={sample['answer']}){chg} "
                          f"kb={out.get('kb_kept')}/{out.get('kb_retrieved')}")

            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()
                self._record({
                    "sample_id": sample_id, "prompt_type": prompt_type,
                    "error": str(e), "is_correct": False,
                    "timestamp": datetime.now().isoformat(),
                }, prompt_type, is_error=True)

            completed = idx + 1
            if completed % self.checkpoint_interval == 0:
                self.save_checkpoint(completed)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self._print_stats()

    def _record(self, traj, prompt_type, is_error=False):
        self.trajectories.append(traj)
        if prompt_type in self.by_prompt:
            self.by_prompt[prompt_type].append(traj)
        self.stats["total"] += 1
        pt = self.stats_by_prompt.get(prompt_type, self.stats)
        if is_error:
            self.stats["errors"] += 1
            pt["errors"] = pt.get("errors", 0) + 1
        else:
            pt["total"] = pt.get("total", 0) + 1
            if traj.get("is_correct"):
                self.stats["correct"] += 1
                pt["correct"] = pt.get("correct", 0) + 1
            if traj.get("answer_changed"):
                self.stats["changed"] += 1
            if traj.get("improved"):
                self.stats["improved"] += 1
            if traj.get("regressed"):
                self.stats["regressed"] += 1

    def _print_stats(self):
        total = self.stats["total"]
        if total == 0:
            print("\nNo trajectories collected.")
            return
        c = self.stats["correct"]
        print(f"\n{'=' * 70}\nRESULTS ({total} trajectories)\n{'=' * 70}")
        print(f"  Accuracy:  {c}/{total} ({c/total:.1%})")
        print(f"  Changed:   {self.stats['changed']}")
        print(f"  Improved:  {self.stats['improved']}")
        print(f"  Regressed: {self.stats['regressed']}")
        print(f"  Errors:    {self.stats['errors']}")
        print(f"  Skipped:   {self.stats['skipped']}")
        for ptn in PROMPT_TYPES:
            s = self.stats_by_prompt[ptn]
            t = s.get("total", 0)
            if t > 0:
                print(f"  {ptn}: {s.get('correct', 0)}/{t} ({s.get('correct', 0)/t:.1%})")

    def save_results(self):
        combined = self.output_dir / "trajectories_all_prompts.json"
        combined.write_text(json.dumps(self.trajectories, indent=2))
        print(f"\n[SAVED] {combined} ({len(self.trajectories)})")
        for ptn in PROMPT_TYPES:
            path = self.output_dir / f"trajectories_{ptn}.json"
            path.write_text(json.dumps(self.by_prompt[ptn], indent=2))
            print(f"[SAVED] {path} ({len(self.by_prompt[ptn])})")
        sp = self.coordinator._sampling_params
        stats_data = {
            "overall": dict(self.stats),
            "by_prompt": {p: dict(s) for p, s in self.stats_by_prompt.items()},
            "tool_config": self.tool_config,
            "model": self.coordinator.model_name,
            "kb": self.coordinator.kb.get_statistics() if self.coordinator.kb else {"mode": "cache"},
            "selective": self.coordinator.selective,
            "sampling": {
                "temperature": sp.get("temperature", 0.0),
                "do_sample": sp.get("do_sample", False),
            },
            "model_stats": self.coordinator.get_statistics(),
            "timestamp": datetime.now().isoformat(),
        }
        (self.output_dir / "collection_stats.json").write_text(json.dumps(stats_data, indent=2))
        print(f"[SAVED] {self.output_dir / 'collection_stats.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Step 2 KB-grounding (single-shot)",
                                allow_abbrev=False)
    p.add_argument("--baseline", required=True, help="Step 1 baseline JSON")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--kb-index", default=None,
                   help="Dir with kb.index/ids.json/config.json (live retrieval; "
                        "needs sentence-transformers). Omit if using --kb-cache.")
    p.add_argument("--kb-cache", default=None,
                   help="Prebuilt kb_cache.json from kb_prefetch.py (no embedder needed). "
                        "Preferred -- required for envs without sentence-transformers (Gemma).")
    p.add_argument("--kb-jsonl", default=None,
                   help="kb.jsonl path (default: <kb-index>/../kb.jsonl)")
    p.add_argument("--selective", action="store_true",
                   help="Selective KB-grounding (per-doc relevance filter)")
    p.add_argument("--retrieve-n", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-interval", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-from", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cache_dir = os.environ.get("SCRATCHDIR", "") + "/.cache/huggingface"
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{cache_dir}/transformers")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tool_label = "kb_selective" if args.selective else "kb"
    print(f"\n{'=' * 70}\nSTEP 2: KB-GROUNDING ({tool_label})\n{'=' * 70}")
    print(f"  Model:   {args.model_name}")
    print(f"  KB index:{args.kb_index}")
    print(f"  Baseline:{args.baseline}\n  Output:  {args.output_dir}\n{'=' * 70}\n")

    with open(args.baseline) as f:
        records = json.load(f)
    if args.max_samples:
        records = records[:args.max_samples]
    print(f"Loaded {len(records)} baseline records")

    # Already complete?
    final_file = Path(args.output_dir) / "trajectories_all_prompts.json"
    if final_file.exists():
        try:
            existing = json.loads(final_file.read_text())
            count = len(existing) if isinstance(existing, list) else len(existing.get("trajectories", []))
            print(f"\n[ALREADY COMPLETE] {final_file} ({count} trajectories) -- nothing to do.")
            sys.exit(0)
        except Exception:
            print("[WARNING] Final file exists but unreadable -- resuming")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if (args.device == "cuda" and torch.cuda.is_available()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype,
        device_map="auto" if args.device == "cuda" else None,
    )
    if args.device != "cuda":
        model = model.to(args.device)
    model.eval()
    print("  Model loaded\n")

 
    kb = None
    kb_cache = None
    if args.kb_cache:
        print(f"\nLoading KB cache: {args.kb_cache}")
        cache_obj = json.load(open(args.kb_cache))
        kb_cache = cache_obj.get("cache", cache_obj)  # accept raw or {meta,cache}
        print(f"  {len(kb_cache)} cached retrievals "
              f"(model={cache_obj.get('meta', {}).get('model', '?')})")
    elif args.kb_index:
        from kb_retriever import KBRetriever
        kb = KBRetriever(index_dir=args.kb_index, kb_jsonl=args.kb_jsonl,
                         device=args.device, default_n=args.retrieve_n)
    else:
        print("ERROR: provide --kb-cache (preferred) or --kb-index", file=sys.stderr)
        sys.exit(2)

    from coordinator_kb import KBGroundingCoordinator
    coordinator = KBGroundingCoordinator(
        model=model, tokenizer=tokenizer, kb_retriever=kb, kb_cache=kb_cache,
        device=args.device, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, model_name=args.model_name,
        selective=args.selective, retrieve_n=args.retrieve_n,
    )

    collector = KBTrajectoryCollector(
        coordinator=coordinator, baseline_records=records,
        output_dir=args.output_dir, tool_config_label=tool_label,
        checkpoint_interval=args.checkpoint_interval,
    )

    start_idx = 0
    if args.resume_from and Path(args.resume_from).exists():
        start_idx = collector.load_checkpoint(Path(args.resume_from))
    elif args.resume:
        latest = collector.find_latest_checkpoint()
        if latest:
            start_idx = collector.load_checkpoint(latest)

    collector.collect_all(start_idx=start_idx)
    collector.save_results()

    print(f"\n{'=' * 70}\nKB-GROUNDING COMPLETE\n  Trajectories: {len(collector.trajectories)}")
    if kb is not None:
        print(f"  KB queries:   {kb.get_statistics()['total_queries']}")
    else:
        print(f"  KB source:    cache ({len(kb_cache)} retrievals)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
