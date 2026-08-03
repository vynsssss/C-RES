#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
judge_rubric.py — the cultural-reasoning rubric (over-culturalisation).

The judge scores a single explanation, BLIND to the retrieved text, and is run
identically at Step 1 (baseline_explanation) and Step 2 (atlas_explanation);
that blindness is what keeps the two steps comparable. It judges HOW the model
reasoned, not whether the verdict is correct or the cultural claims are true.

Three questions, judged independently, each returned as JSON with a
one-sentence justification:
  Q1 cultural_reasoning  (yes/no)               — verdict rests on a SPECIFIC-GROUP
                                                   attribution?
  Q2 used_given_clue     (yes/no, clue prompts) — applied the prompt's rule/value?
  Q3 general_sufficient  (yes/no/no_cultural_claim)
        could a PLAIN explanation have reached the same verdict without the
        cultural claim? (delete-the-culture test)
Derived:
  over_cult_general = cultural_reasoning AND general_sufficient=="yes"   (all prompts)
  over_cult_clue    = cultural_reasoning AND used_given_clue==False      (SRoT/SCV)

Interface:
  JUDGE_SYSTEM, build_user_prompt(item, which)
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Set


# ===========================================================================
# The over-culturalisation rubric
# ===========================================================================
JUDGE_SYSTEM = (
    "Judge HOW the model reasoned in the explanation below — not whether its "
    "verdict was correct, and not whether any cultural claim it makes is true. "
    "The questions are independent; judge each on its own.\n\n"
    "Respond with ONLY a JSON object and nothing else."
)

_Q_CULTURAL = (
    "Q1 CULTURAL_ATTRIBUTION (yes/no):\n"
    "  Does the explanation reach its verdict by appealing to what a specific "
    "group (a country, culture, region, religion, or ethnicity) does, values, or "
    "believes?\n"
    "  Yes — the reason it gives is a group claim: people in that country/culture "
    "do or value X, and that is why the verdict holds.\n"
    "  No  — the reason it gives is not tied to a specific group: the facts of the "
    "situation, a norm that applies to everyone, or the instruction it was given.\n"
    "  Examples:\n"
    "    Yes — \"In Germany punctuality is taken seriously, so arriving late was a "
    "real offense.\"\n"
    "    No  — \"Interrupting someone who's speaking is rude, so this isn't "
    "acceptable.\" (a norm for everyone, no group)\n"
    "    No  — \"The guidance was to greet on entering, and she did, so it's "
    "acceptable.\" (uses the given rule — not a group claim)\n"
)

_Q_CLUE = (
    "Q2 APPLIED_GIVEN_CLUE (yes/no):\n"
    "  The model was given this {clue_name}: \"{clue_label}\".\n"
    "  Does the explanation apply it to reach the verdict?\n"
    "  Yes — the reason it gives uses or relies on the {clue_name}.\n"
    "  No  — it decides on another basis, even if it mentions the {clue_name} in "
    "passing.\n"
)

_Q_GENERAL = (
    "Q3 GENERAL_SUFFICIENT (yes / no / no_cultural_claim):\n"
    "  When the explanation reasons from culture, could a GENERAL, non-cultural "
    "explanation (everyday social norms, or the situation itself) have reached the "
    "SAME verdict just as well, WITHOUT the cultural claim?\n"
    "  Test: remove the cultural claim. Does a plain explanation still give the "
    "same verdict?\n"
    "  yes — a plain explanation would have sufficed; the cultural framing was "
    "unnecessary.\n"
    "  no  — the verdict depends on the specific cultural fact; remove it and the "
    "reasoning no longer holds.\n"
    "  no_cultural_claim — the explanation makes no cultural claim (nothing to "
    "test).\n"
    "  Examples:\n"
    "    yes — \"In Egypt people value hospitality, so greeting warmly was right.\" "
    "(plain: \"greeting warmly is polite\" gives the same verdict)\n"
    "    yes — \"Japanese culture values politeness, so interrupting was rude.\" "
    "(plain: \"interrupting is rude\" gives the same verdict)\n"
    "    no  — \"In Japan shoes come off indoors, so keeping them on was "
    "offensive.\" (plain \"shoes-on indoors is offensive\" is not universally "
    "true — the verdict needs the cultural fact)\n"
    "    no  — \"Thai custom treats the head as sacred, so patting the child's "
    "head was wrong.\" (needs the specific cultural fact)\n"
)

CAT = {"general_sufficient": ["yes", "no", "no_cultural_claim"]}
BOOL_KEYS = ["cultural_reasoning", "used_given_clue"]
ALL_SCORE_KEYS = BOOL_KEYS + list(CAT)


def _clue_fields(item: Dict) -> Optional[Dict[str, str]]:
    pt = item.get("prompt_type", "")
    if pt == "story_rot":
        return {"name": "rule of thumb", "label": (item.get("rot") or "").strip()}
    if pt == "story_country_value":
        return {"name": "value", "label": (item.get("value") or "").strip()}
    return None


def _scenario_block(item: Dict) -> str:
    pt = item.get("prompt_type", "")
    lines = []
    if pt in ("story_country", "story_country_value") and item.get("country"):
        lines.append(f"Country: {item['country']}")
    if pt == "story_country_value" and (item.get("value") or "").strip():
        lines.append(f"Rule of thumb (value): {item['value']}")
    if pt == "story_rot" and (item.get("rot") or "").strip():
        lines.append(f"Rule of thumb: {item['rot']}")
    lines.append(f"Story: {item.get('story', '')}")
    return "\n".join(lines)


def score_keys_for(which: str, item: Dict) -> List[str]:
    keys = ["cultural_reasoning"]
    if _clue_fields(item) is not None:
        keys.append("used_given_clue")
    keys.append("general_sufficient")
    return keys


def _json_schema(keys: List[str]) -> str:
    parts = []
    for k in keys:
        if k in CAT:
            parts.append(f'"{k}": "' + "|".join(CAT[k]) + '"')
        else:
            parts.append(f'"{k}": true/false')
    just = ", ".join(f'"j_{k}": "one-sentence reason"' for k in keys)
    return "{" + ", ".join(parts) + ", " + just + "}"


def build_user_prompt(item: Dict, which: str) -> str:
    explanation = (item.get("baseline_explanation" if which == "baseline"
                            else "atlas_explanation") or "")
    clue = _clue_fields(item)
    keys = score_keys_for(which, item)
    blocks = [_Q_CULTURAL]
    if clue is not None:
        blocks.append(_Q_CLUE.format(clue_name=clue["name"], clue_label=clue["label"]))
    blocks.append(_Q_GENERAL)
    return (
        "Judge how the model reasoned in the MODEL EXPLANATION below, using the "
        "questions. Give a one-sentence reason for each.\n\n"
        + "\n".join(blocks)
        + f"\nSCENARIO:\n{_scenario_block(item)}\n"
        + f"\nMODEL EXPLANATION TO JUDGE:\n\"\"\"\n{explanation}\n\"\"\"\n\n"
        + f"Return ONLY this JSON object:\n{_json_schema(keys)}"
    )


def parse_judge_response(text: str, expected_keys: Optional[List[str]] = None) -> Optional[Dict]:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()  # Qwen3 think strip
    if t.startswith("```"):
        t = t.lstrip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    s, e = t.find("{"), t.rfind("}")
    obj = None
    if s >= 0 and e > s:
        try:
            obj = json.loads(t[s:e + 1])
        except (json.JSONDecodeError, ValueError):
            obj = None
    if obj is None:
        obj = {}
        for k in BOOL_KEYS:
            m = re.search(rf'"{k}"\s*:\s*(true|false|yes|no|1|0)', t, re.I)
            if m:
                obj[k] = m.group(1)
        m = re.search(r'"general_sufficient"\s*:\s*"?(yes|no|no_cultural_claim)"?', t, re.I)
        if m:
            obj["general_sufficient"] = m.group(1)
        if not obj:
            return None

    def as_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1")
        if isinstance(v, (int, float)):
            return bool(v)
        return None

    keys = expected_keys or [k for k in ALL_SCORE_KEYS if k in obj]
    out: Dict = {}
    for k in keys:
        if k in CAT:
            v = obj.get(k)
            if isinstance(v, str) and v.strip().lower() in CAT[k]:
                out[k] = v.strip().lower()
                out[f"j_{k}"] = str(obj.get(f"j_{k}", ""))[:300]
        else:
            b = as_bool(obj.get(k))
            if b is not None:
                out[k] = b
                out[f"j_{k}"] = str(obj.get(f"j_{k}", ""))[:300]

    core = ["cultural_reasoning"]
    if expected_keys and "used_given_clue" in expected_keys:
        core.append("used_given_clue")
    if any(c not in out for c in core):
        return None

    if out.get("cultural_reasoning") is True and "general_sufficient" in out:
        out["over_cult_general"] = (out["general_sufficient"] == "yes")
    elif out.get("cultural_reasoning") is False:
        out["over_cult_general"] = False
    if "cultural_reasoning" in out and "used_given_clue" in out:
        out["over_cult_clue"] = out["cultural_reasoning"] and (not out["used_given_clue"])
    return out
if __name__ == "__main__":
    srot = {"prompt_type": "story_rot", "rot": "One should greet people when entering a room.",
            "story": "Sarah greeted everyone.",
            "baseline_explanation": "Greeting on entry is polite, so acceptable.",
            "atlas_explanation": "In Egypt collectivist values make greeting essential, so acceptable.",
            "atlas_reading": "Etiquette: guests commonly greet those present."}
    print("keys:", score_keys_for("atlas", srot))
    print("parse:", parse_judge_response(
        '{"cultural_reasoning":true,"used_given_clue":false,"general_sufficient":"yes",'
        '"j_cultural_reasoning":"a","j_used_given_clue":"b","j_general_sufficient":"c"}',
        expected_keys=["cultural_reasoning", "used_given_clue", "general_sufficient"]))