#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
judge_batch_submit.py — submit the cultural overeach JUDGE batch to one
provider, reusing the SAME batch endpoints/quirks as the project's
batch_submit.py, but with the JUDGE rubric system prompt per request.

Workflow:
    python3 judge_batch_submit.py --provider anthropic \
        --model claude-sonnet-4-6 --pilot judge_pilot.jsonl \
        --job-dir judge_batches/claude
    # repeat for: --provider openai --model gpt-5.2 ...
    #             --provider gemini --model gemini-3-flash-preview ...
    # then fetch each with the project's batch_fetch.py (results -> {custom_id:text})

Saves manifest.json (batch_id, provider, model) in --job-dir, same convention
as batch_submit.py, so batch_fetch.py can poll/download.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import requests

import judge_rubric as R

MAX_OUTPUT_TOKENS = 2048  # matches the baseline runs (thinking ON). 2048 fits
                          # Gemini's thinking tokens + the judge JSON; lower caps
                          # truncated the JSON when thinking consumed the budget.


def build_entries(pilot_path: str) -> List[Dict]:
    """One entry per (item, which) with the judge system + user prompt.

    custom_id is a short index token 'j{N}' (Anthropic requires
    ^[a-zA-Z0-9_-]{1,64}$, so the original 'pilot_id||which' — which contains
    ':' and '|' — is not allowed). Each entry also carries 'ref' = the full
    'pilot_id||which' so we can write an id_map for the parser to recover it.
    """
    entries = []
    idx = 0
    for line in open(pilot_path):
        if not line.strip():
            continue
        it = json.loads(line)
        for which in ("baseline", "atlas"):
            entries.append({
                "custom_id": f"j{idx}",
                "ref": f"{it['pilot_id']}||{which}",
                "system": R.JUDGE_SYSTEM,
                "user": R.build_user_prompt(it, which),
            })
            idx += 1
    return entries


# --- Anthropic (inline requests) ---
def submit_anthropic(entries, model_id) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    reqs = [{
        "custom_id": e["custom_id"],
        "params": {
            "model": model_id,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
            "system": e["system"],
            "messages": [{"role": "user", "content": e["user"]}],
        },
    } for e in entries]
    r = requests.post(
        "https://api.anthropic.com/v1/messages/batches",
        headers={"Content-Type": "application/json", "x-api-key": api_key,
                 "anthropic-version": "2023-06-01"},
        json={"requests": reqs}, timeout=300,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Anthropic submit failed {r.status_code}: {r.text[:800]}")
    return r.json()["id"]


# --- OpenAI (Files API + Batches; GPT-5.x max_completion_tokens, no temp) ---
def submit_openai(entries, model_id, work_dir: Path) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    jsonl = work_dir / "openai_judge_input.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps({
                "custom_id": e["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": e["system"]},
                        {"role": "user", "content": e["user"]},
                    ],
                    "max_completion_tokens": MAX_OUTPUT_TOKENS,
                    "response_format": {"type": "json_object"},
                },
            }, ensure_ascii=False) + "\n")
    with open(jsonl, "rb") as f:
        r = requests.post("https://api.openai.com/v1/files",
                          headers={"Authorization": f"Bearer {api_key}"},
                          files={"file": (jsonl.name, f, "application/jsonl")},
                          data={"purpose": "batch"}, timeout=300)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"OpenAI upload failed {r.status_code}: {r.text[:500]}")
    file_id = r.json()["id"]
    r = requests.post("https://api.openai.com/v1/batches",
                      headers={"Content-Type": "application/json",
                               "Authorization": f"Bearer {api_key}"},
                      json={"input_file_id": file_id,
                            "endpoint": "/v1/chat/completions",
                            "completion_window": "24h"}, timeout=120)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"OpenAI batch create failed {r.status_code}: {r.text[:800]}")
    return r.json()["id"]


# --- Gemini (Files API upload + Batches) ---
def submit_gemini(entries, model_id, work_dir: Path) -> str:
    import time
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set")

    jsonl = work_dir / "gemini_judge_input.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps({
                "key": e["custom_id"],
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": e["user"]}]}],
                    "system_instruction": {"parts": [{"text": e["system"]}]},
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": MAX_OUTPUT_TOKENS,
                        "responseMimeType": "application/json",
                        # Thinking left ON to match the baseline runs. 2048 tokens
                        # gives room for thinking + the judge JSON (truncation at
                        # 1024 was thinking eating the budget; 2048 fits both).
                    },
                },
            }, ensure_ascii=False) + "\n")
    size = jsonl.stat().st_size

    # 1. Start resumable upload — MUST send the content length up front, else
    #    the server completes a single-shot upload and returns no session URL.
    meta = requests.post(
        f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={api_key}",
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": "application/jsonl",
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": "gemini_judge_input.jsonl"}}, timeout=120)
    if meta.status_code not in (200, 201):
        raise RuntimeError(f"Gemini metadata init failed {meta.status_code}: {meta.text[:500]}")
    up_url = meta.headers.get("x-goog-upload-url") or meta.headers.get("X-Goog-Upload-URL")
    if not up_url:
        raise RuntimeError(f"No Gemini upload URL: {dict(meta.headers)}")

    # 2. Upload + finalize.
    body = open(jsonl, "rb").read()
    r = requests.post(up_url, headers={
        "Content-Length": str(len(body)),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize",
    }, data=body, timeout=600)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Gemini upload failed {r.status_code}: {r.text[:500]}")
    file_name = r.json().get("file", {}).get("name")
    if not file_name:
        raise RuntimeError(f"No file name in upload response: {r.json()}")

    # 3. Create batch (primary endpoint, then fallback) — camelCase config.
    create_url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                  f"{model_id}:batchGenerateContent?key={api_key}")
    body_batch = {"batch": {"displayName": f"judge-oc-{int(time.time())}",
                            "inputConfig": {"fileName": file_name}}}
    print(f"  [gemini] file_name={file_name}")
    print(f"  [gemini] POST {create_url.split('?')[0]}")
    r = requests.post(create_url, json=body_batch, timeout=120)
    print(f"  [gemini] primary create -> HTTP {r.status_code}; body[:300]={r.text[:300]!r}")
    if r.status_code not in (200, 201):
        fb = f"https://generativelanguage.googleapis.com/v1beta/batches?key={api_key}"
        fb_body = {"batch": {
            "displayName": f"judge-oc-{int(time.time())}",
            "model": f"models/{model_id}",
            "inputConfig": {"fileName": file_name}}}
        print(f"  [gemini] fallback POST {fb.split('?')[0]}")
        r = requests.post(fb, json=fb_body, timeout=120)
        print(f"  [gemini] fallback create -> HTTP {r.status_code}; body[:300]={r.text[:300]!r}")
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Gemini batch create failed {r.status_code}: {r.text[:800]}")
    data = r.json()
    name = data.get("name") or data.get("batch", {}).get("name")
    if not name:
        raise RuntimeError(f"No batch name in response: {data}")
    return name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["anthropic", "openai", "gemini"])
    ap.add_argument("--model", required=True, help="API model id (e.g. claude-sonnet-4-6)")
    ap.add_argument("--pilot", default="judge_pilot.jsonl")
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--max-items", type=int, default=None,
                    help="SLICE for a sanity test (e.g. 100 items -> 200 requests)")
    args = ap.parse_args()

    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    entries = build_entries(args.pilot)
    if args.max_items:
        entries = entries[:args.max_items * 2]  # ×2 (baseline+atlas)
    print(f"[{args.provider}] {len(entries)} judge requests, model={args.model}")

    # Save id_map: short custom_id -> full 'pilot_id||which' (parser needs this,
    # since Anthropic custom_ids can't contain ':' or '|').
    id_map = {e["custom_id"]: e["ref"] for e in entries}
    (job_dir / "id_map.json").write_text(json.dumps(id_map))
    print(f"  id_map -> {job_dir / 'id_map.json'} ({len(id_map)} ids)")

    if args.provider == "anthropic":
        batch_id = submit_anthropic(entries, args.model)
    elif args.provider == "openai":
        batch_id = submit_openai(entries, args.model, job_dir)
    else:
        batch_id = submit_gemini(entries, args.model, job_dir)

    manifest = {
        "provider": args.provider, "model": args.model,
        "batch_id": batch_id, "n_requests": len(entries),
        "pilot": str(Path(args.pilot).resolve()),
        "kind": "over_culturalization_judge",
    }
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  submitted batch_id = {batch_id}")
    print(f"  manifest -> {job_dir / 'manifest.json'}")
    print(f"\nFetch with the project's batch_fetch.py (or your fetch tooling) "
          f"pointed at --job-dir {job_dir}")
    print("Results should normalise to {custom_id: text}; then run judge_batch_parse.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())