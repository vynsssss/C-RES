#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
judge_fetch.py — poll a judge batch and download results to {custom_id: text}.


Usage:
    python3 judge_fetch.py --job-dir judge_batches/claude_test
    # writes  <job-dir>/results.jsonl  with lines {"custom_id":..., "text":...}
    # re-run until status is finished; safe to re-run after completion.

Then:
    python3 judge_batch_parse.py --pilot judge_pilot.jsonl \
        --judge claude judge_batches/claude_test/results.jsonl \
        --id-map judge_batches/claude_test/id_map.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import requests


# ---------------- Anthropic ----------------
def anthropic_status(batch_id: str) -> dict:
    key = os.getenv("ANTHROPIC_API_KEY")
    r = requests.get(
        f"https://api.anthropic.com/v1/messages/batches/{batch_id}",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout=60)
    r.raise_for_status()
    return r.json()


def anthropic_download(results_url: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    key = os.getenv("ANTHROPIC_API_KEY")
    r = requests.get(results_url,
                     headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                     stream=True, timeout=600)
    r.raise_for_status()
    out, errs = {}, {}
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = e.get("custom_id")
        res = e.get("result", {})
        if res.get("type") == "succeeded":
            blocks = res.get("message", {}).get("content", [])
            text = next((b.get("text", "") for b in blocks if b.get("type") == "text"), "")
            out[cid] = text
        else:
            errs[cid] = f"{res.get('type')}: {res.get('error', {}).get('message', '?')}"
    return out, errs


# ---------------- OpenAI ----------------
def openai_status(batch_id: str) -> dict:
    key = os.getenv("OPENAI_API_KEY")
    r = requests.get(f"https://api.openai.com/v1/batches/{batch_id}",
                     headers={"Authorization": f"Bearer {key}"}, timeout=60)
    r.raise_for_status()
    return r.json()


def openai_download(file_id: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    key = os.getenv("OPENAI_API_KEY")
    r = requests.get(f"https://api.openai.com/v1/files/{file_id}/content",
                     headers={"Authorization": f"Bearer {key}"}, stream=True, timeout=600)
    r.raise_for_status()
    out, errs = {}, {}
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = e.get("custom_id")
        resp = e.get("response", {}) or {}
        if resp.get("status_code") == 200:
            body = resp.get("body", {})
            ch = body.get("choices", [{}])
            out[cid] = ch[0].get("message", {}).get("content", "") if ch else ""
        else:
            errs[cid] = f"http {resp.get('status_code')}"
    return out, errs


# ---------------- Gemini ----------------
def gemini_status(batch_name: str) -> dict:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    r = requests.get(
        f"https://generativelanguage.googleapis.com/v1beta/{batch_name}?key={key}",
        timeout=60)
    r.raise_for_status()
    return r.json()


def gemini_download(status_data: dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Download Gemini batch output -> {custom_id: text}.  """
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    # Locate output file name across known schema shapes.
    out_file = None
    for path in [
        ("metadata", "output", "responsesFile"),   # current
        ("response", "responsesFile"),
        ("response", "outputConfig", "fileName"),
        ("response", "outputFile"),
        ("metadata", "outputConfig", "fileName"),
    ]:
        node = status_data
        for k in path:
            node = node.get(k) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, str):
            out_file = node
            break
    if not out_file:
        dest = status_data.get("response", {}).get("dest", {}) \
            if isinstance(status_data.get("response"), dict) else {}
        out_file = dest.get("fileName") if isinstance(dest, dict) else None

    # Inline responses (small batches) as a last resort.
    if not out_file:
        inlined = (status_data.get("response", {})
                   .get("inlinedResponses", {}).get("inlinedResponses", []))
        out, errs = {}, {}
        for item in inlined:
            cid = item.get("metadata", {}).get("key") or item.get("key")
            cands = item.get("response", {}).get("candidates", [])
            parts = cands[0].get("content", {}).get("parts", []) if cands else []
            out[cid] = "".join(p.get("text", "") for p in parts)
        if out:
            return out, errs
        raise RuntimeError(f"Could not locate Gemini output file in status: "
                           f"{json.dumps(status_data)[:800]}")

    # Download the results file.
    dl = f"https://generativelanguage.googleapis.com/download/v1beta/{out_file}:download?alt=media&key={key}"
    r = requests.get(dl, timeout=600)
    if r.status_code != 200:
        for fb in [
            f"https://generativelanguage.googleapis.com/v1beta/{out_file}:download?alt=media&key={key}",
            f"https://generativelanguage.googleapis.com/download/v1beta/{out_file}?alt=media&key={key}",
            f"https://generativelanguage.googleapis.com/v1beta/{out_file}?alt=media&key={key}",
        ]:
            r = requests.get(fb, timeout=600)
            if r.status_code == 200:
                break
        if r.status_code != 200:
            raise RuntimeError(f"Gemini file download failed {r.status_code}: {r.text[:500]}")

    out, errs = {}, {}
    for line in r.text.splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = e.get("key")
        if e.get("response"):
            cands = e["response"].get("candidates", [])
            parts = cands[0].get("content", {}).get("parts", []) if cands else []
            out[cid] = "".join(p.get("text", "") for p in parts)
        elif e.get("error"):
            errs[cid] = e["error"].get("message", "unknown error")
    return out, errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--out", default=None, help="default <job-dir>/results.jsonl")
    args = ap.parse_args()

    job = Path(args.job_dir)
    man = json.loads((job / "manifest.json").read_text())
    provider = man["provider"]
    batch_id = man["batch_id"]
    out_path = Path(args.out) if args.out else job / "results.jsonl"
    print(f"[{provider}] batch {batch_id}")

    if provider == "anthropic":
        st = anthropic_status(batch_id)
        status = st.get("processing_status")
        counts = st.get("request_counts", {})
        print(f"  status={status}  counts={counts}")
        if status in ("in_progress", "validating", "cancelling"):
            print("  not finished yet — re-run later.")
            return 0
        # terminal: 'ended' (normal) or 'canceled'. Download whatever results exist.
        if not st.get("results_url"):
            print(f"  [terminal:{status}] no results_url — nothing to download.")
            return 0
        if status != "ended":
            print(f"  [warn] terminal status '{status}', not 'ended' — downloading partial results.")
        results, errs = anthropic_download(st["results_url"])
    elif provider == "openai":
        st = openai_status(batch_id)
        status = st.get("status")
        print(f"  status={status}  counts={st.get('request_counts')}")
        IN_PROGRESS = {"validating", "in_progress", "finalizing", "cancelling"}
        if status in IN_PROGRESS:
            print("  not finished yet — re-run later.")
            return 0
        # terminal: completed | expired | failed | cancelled. Download partials if any.
        if status != "completed":
            print(f"  [warn] batch is TERMINAL with status '{status}' (will NOT change). "
                  f"Downloading whatever completed; the rest are missing.")
        out_fid = st.get("output_file_id")
        if not out_fid:
            print(f"  [terminal:{status}] no output_file_id — no results to download.")
            print("  Re-submit the missing custom_ids (see id_map.json) as a new batch.")
            return 1 if status != "completed" else 0
        results, errs = openai_download(out_fid)
        # also pull the error file if present (expired/failed often populate it)
        err_fid = st.get("error_file_id")
        if err_fid:
            try:
                e_ok, e_bad = openai_download(err_fid)
                errs.update({k: "error_file" for k in e_ok})
                errs.update(e_bad)
            except Exception as ex:
                print(f"  (could not read error_file: {ex})")
    elif provider == "gemini":
        st = gemini_status(batch_id)
        state = (st.get("metadata", {}) or {}).get("state") or st.get("state")
        print(f"  state={state}")
        s = str(state or "")
        is_success = ("SUCCEEDED" in s) or ("completed" in s.lower())
        is_terminal_fail = any(w in s.upper() for w in ("FAILED", "EXPIRED", "CANCELLED", "CANCELED"))
        if not is_success and not is_terminal_fail:
            print("  not finished yet — re-run later.")
            return 0
        if is_terminal_fail:
            print(f"  [warn] batch TERMINAL with state '{state}' — attempting to download any partials.")
        try:
            results, errs = gemini_download(st)
        except Exception as ex:
            print(f"  [terminal:{state}] could not download results: {ex}")
            return 1
    else:
        print(f"unknown provider {provider}", file=sys.stderr)
        return 2

    with open(out_path, "w") as f:
        for cid, text in results.items():
            f.write(json.dumps({"custom_id": cid, "text": text}) + "\n")
    print(f"  downloaded {len(results)} results, {len(errs)} errors -> {out_path}")
    if errs:
        ex = list(errs.items())[:3]
        print(f"  example errors: {ex}")

    # Report any custom_ids that never came back (e.g. expired batch) so they can
    # be re-submitted. Compare against id_map.json (the full submitted set).
    idmap_path = job / "id_map.json"
    if idmap_path.exists():
        submitted = set(json.loads(idmap_path.read_text()).keys())
        got = set(results.keys())
        missing = sorted(submitted - got)
        if missing:
            miss_path = job / "missing_ids.txt"
            miss_path.write_text("\n".join(missing) + "\n")
            print(f"  [MISSING] {len(missing)}/{len(submitted)} custom_ids never returned "
                  f"-> {miss_path}")
            print(f"            re-submit these as a new batch, then fetch + parse BOTH "
                  f"results.jsonl files together.")
    print(f"\n  Parse with:\n    python3 judge_batch_parse.py --pilot judge_pilot.jsonl \\")
    print(f"      --judge {provider} {out_path} --id-map {job / 'id_map.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())