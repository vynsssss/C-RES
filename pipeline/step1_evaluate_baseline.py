# -*- coding: utf-8 -*-
"""
Step 1: Baseline Evaluation via vLLM offline batching.

USAGE
-----
  python3 step1_evaluate_baseline_vllm.py \
      --train-data  $DATA/normad_dataset.json \
      --output-dir  $SCRATCHDIR/CRES-paper_baseline_t10 \
      --data-id     normad_dataset \
      --model-name  google/gemma-4-31B-it \
      --model-id    gemma_31b \
      --temperature 1.0 \
      --tp 4 --resume

Resume is automatic: completed (sample_id, prompt_type) keys are read from
vllm_progress.jsonl in the output dir and skipped. Safe to resubmit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Project code -- the single source of truth for prompts / parsing / scoring.
from prompt_formatter import format_prompt, get_valid_prompt_types, normalize_fields
from coordinator import Coordinator, _RE_THINK, _classify_output
from base_agent import _resolve_sampling_params


# ---------------------------------------------------------------------------
# Pull the Step-1 system prompt LIVE from Coordinator  
# ---------------------------------------------------------------------------
STEP1_SYSTEM_PROMPT: str = Coordinator.system_prompt.fget(object())  # type: ignore[attr-defined]


def parse_like_evaluate(raw: str) -> Dict[str, str]:
    stripped = _RE_THINK.sub("", raw, count=0).strip()
    status = _classify_output(stripped)
    if status in ("empty", "degenerated"):
        return {"decision": status, "reasoning": "", "output_status": status}
    parsed = Coordinator._parse_json_response(stripped)
    return {
        "decision": parsed["decision"],
        "reasoning": parsed["reasoning"],
        "output_status": "ok",
    }


def build_record(sample: Dict[str, Any], idx: int, prompt_type: str,
                 raw: str) -> Dict[str, Any]:
    """Full 'ok/empty/degenerated' record -- identical schema to the HF path."""
    parsed = parse_like_evaluate(raw)
    gold = sample["answer"]
    decision = parsed["decision"]
    return {
        "sample_id":    sample.get("id", f"sample_{idx}"),
        "sample_index": idx,
        "prompt_type":  prompt_type,
        "country":      sample.get("country", ""),
        "story":        sample.get("story", ""),
        "value":        sample.get("value", ""),
        "rot":          sample.get("rot", ""),
        "gold_answer":  gold,
        "decision":     decision,
        "reasoning":    parsed["reasoning"],
        "raw_response": raw,
        "is_correct":   Coordinator.is_correct(decision, gold),
        "timestamp":    datetime.now().isoformat(),
    }


def build_sampling_params(model_name: str, temperature: float, max_tokens: int):
    from vllm import SamplingParams
    sp = _resolve_sampling_params(model_name, temperature)

    if not sp.get("do_sample", True):
        # Greedy -- vLLM treats temperature 0 as argmax.
        return SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.1,   # base_agent applies this even when greedy
            max_tokens=max_tokens,
        )

    top_k = sp["top_k"]
    vllm_top_k = -1 if (top_k is None or top_k == 0) else int(top_k)
    return SamplingParams(
        temperature=sp["temperature"],
        top_p=sp["top_p"],
        top_k=vllm_top_k,
        min_p=(sp["min_p"] if sp["min_p"] is not None else 0.0),
        repetition_penalty=1.1,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Progress file (resume) -- one JSON line per completed work key.
# ---------------------------------------------------------------------------
def load_progress(progress_path: Path) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not progress_path.exists():
        return done
    for line in progress_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            done[obj["key"]] = obj["record"]
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def append_progress(progress_path: Path, rows: List[Tuple[str, Dict[str, Any]]]) -> None:
    with progress_path.open("a") as f:
        for key, record in rows:
            f.write(json.dumps({"key": key, "record": record}) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Output writer 
# ---------------------------------------------------------------------------
def save_outputs(output_dir: Path, data_id: str, model_id: str, model_name: str,
                 records: List[Dict[str, Any]]) -> Path:
    base = f"baseline_{data_id}_{model_id}"

    main_path = output_dir / f"{base}.json"
    main_path.write_text(json.dumps(records, indent=2))
    print(f"\n[saved] {main_path}  ({len(records)} records)")

    by_pt: Dict[str, list] = defaultdict(list)
    for r in records:
        by_pt[r.get("prompt_type", "unknown")].append(r)
    for pt, recs in by_pt.items():
        p = output_dir / f"{base}_{pt}.json"
        p.write_text(json.dumps(recs, indent=2))
        print(f"[saved] {p}  ({len(recs)} records)")

    # Stats (mirror the HF stats file shape; model_stats is engine-specific here)
    total = sum(1 for r in records if "error" not in r and r["decision"] in {"yes", "no", "neutral"})
    correct = sum(1 for r in records if r.get("is_correct"))
    errors = sum(1 for r in records if "error" in r)
    by_prompt_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in records:
        if "error" in r:
            continue
        pt = r["prompt_type"]
        if r["decision"] in {"yes", "no", "neutral"}:
            by_prompt_stats[pt]["total"] += 1
            if r["is_correct"]:
                by_prompt_stats[pt]["correct"] += 1

    stats_path = output_dir / f"{base}_stats.json"
    stats_path.write_text(json.dumps({
        "model":       model_name,
        "model_id":    model_id,
        "data_id":     data_id,
        "engine":      "vllm",
        "overall":     {"total": total, "correct": correct, "errors": errors},
        "by_prompt":   {k: dict(v) for k, v in by_prompt_stats.items()},
        "timestamp":   datetime.now().isoformat(),
    }, indent=2))
    print(f"[saved] {stats_path}")

    if total:
        print(f"\nOverall: {correct}/{total} = {correct/total:.1%}  (errors: {errors})")
    return main_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 1 baseline via vLLM offline batching")
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--data-id", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--model-id", required=True,
                    help="Short id for filenames, e.g. gemma_31b (matches Step 2 lookup)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="Default 2048 (4096 for *think* models), matching the HF script")
    ap.add_argument("--max-samples", type=int, default=None)
    # vLLM engine knobs
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size (GPUs)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--chunk-size", type=int, default=1024,
                    help="Prompts per batched llm.chat() call; also the checkpoint granularity")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # max_new_tokens default mirrors the HF script.
    is_think = "think" in args.model_name.lower()
    max_new_tokens = args.max_new_tokens if args.max_new_tokens else (4096 if is_think else 2048)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / f"vllm_progress_{args.data_id}_{args.model_id}.jsonl"

    # -- Load + normalize data ---------------------------------------------
    print(f"Loading data: {args.train_data}")
    raw_data = json.loads(Path(args.train_data).read_text())
    samples = [normalize_fields(s) for s in raw_data]
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"  {len(samples)} samples")
    print(f"Model: {args.model_name}  ({args.model_id})  temp={args.temperature}  "
          f"max_new_tokens={max_new_tokens}")
    print(f"Output: baseline_{args.data_id}_{args.model_id}.json  |  tp={args.tp}\n")

    # -- Build the ordered work list (sample-major, prompt-type order) ------
    #    key = "{sample_id}_{prompt_type}" -- same key the HF checkpoint uses.
    work: List[Tuple[str, Dict[str, Any], int, str]] = []
    prebuilt: Dict[str, Dict[str, Any]] = {}   # format errors -> record now, no generation
    for idx, sample in enumerate(samples):
        sid = sample.get("id", f"sample_{idx}")
        for pt in get_valid_prompt_types(sample):
            key = f"{sid}_{pt}"
            try:
                _ = format_prompt(sample, pt)
                work.append((key, sample, idx, pt))
            except ValueError as e:
                prebuilt[key] = {"sample_id": sid, "prompt_type": pt,
                                 "error": str(e), "is_correct": False}

    # -- Resume: which keys already have records? --------------------------
    done_records: Dict[str, Dict[str, Any]] = load_progress(progress_path) if args.resume else {}
    # Persist any brand-new format-error records so resume stays consistent.
    new_prebuilt = [(k, r) for k, r in prebuilt.items() if k not in done_records]
    if new_prebuilt:
        append_progress(progress_path, new_prebuilt)
        done_records.update(dict(new_prebuilt))

    todo = [(k, s, i, pt) for (k, s, i, pt) in work if k not in done_records]
    print(f"{len(work)} generable + {len(prebuilt)} format-error; "
          f"{len(done_records)} already done -> {len(todo)} to generate\n")

    # -- Load vLLM once -----------------------------------------------------
    if todo:
        from vllm import LLM
        _eager = os.environ.get("VLLM_ENFORCE_EAGER", "0") not in ("0", "", "false", "False")
        llm = LLM(
            model=args.model_name,
            tensor_parallel_size=args.tp,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            enforce_eager=_eager,
        )
        sampling = build_sampling_params(args.model_name, args.temperature, max_new_tokens)

        start = time.time()
        for c0 in range(0, len(todo), args.chunk_size):
            chunk = todo[c0: c0 + args.chunk_size]
            convs = [
                [{"role": "system", "content": STEP1_SYSTEM_PROMPT},
                 {"role": "user", "content": format_prompt(s, pt)}]
                for (_, s, _, pt) in chunk
            ]
            try:
                outputs = llm.chat(convs, sampling)
            except TypeError:
                # Older vLLM signature fallback.
                outputs = llm.chat(convs, sampling_params=sampling)

            rows: List[Tuple[str, Dict[str, Any]]] = []
            for (key, s, i, pt), out in zip(chunk, outputs):
                text = out.outputs[0].text if out.outputs else ""
                record = build_record(s, i, pt, text)
                done_records[key] = record
                rows.append((key, record))
            append_progress(progress_path, rows)

            n_done = min(c0 + args.chunk_size, len(todo))
            elapsed = time.time() - start
            rate = n_done / elapsed if elapsed else 0.0
            print(f"  [{n_done}/{len(todo)}]  {elapsed/60:.1f} min  ({rate:.1f} gen/s)")

    # -- Assemble final records in the original work order ------------------
    all_keys_in_order: List[str] = []
    for idx, sample in enumerate(samples):
        sid = sample.get("id", f"sample_{idx}")
        for pt in get_valid_prompt_types(sample):
            all_keys_in_order.append(f"{sid}_{pt}")
    records = [done_records[k] for k in all_keys_in_order if k in done_records]

    save_outputs(output_dir, args.data_id, args.model_id, args.model_name, records)
    print(f"\nSTEP 1 (vLLM) COMPLETE")
    print(f"Use with Step 2:  --baseline "
          f"{output_dir / f'baseline_{args.data_id}_{args.model_id}.json'}")


if __name__ == "__main__":
    main()
