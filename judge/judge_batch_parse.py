#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Per item: 'baseline' (Step 1, pre-knowledge) and 'atlas' (Step 2). Each yields
cultural_reasoning (Q1), used_given_clue (Q2, clue prompts), general_sufficient
(Q3), and derived over_cult_general (Q1 ∧ Q3=yes) and over_cult_clue (Q1 ∧ ¬Q2).

REPORTS
  1. AGREEMENT — kappa per dimension.
  2. TABLE A   — marginal rates by model x prompt x step.
  3. LADDER    — salience ladder: over_cult_general & Q1 rate by prompt x step.
                 (SRoT=no country; SC/SCV Step1=country salient, no retrieval;
                  Step2=+retrieval.) Rising over_cult from SRoT->country-mentioned
                  BEFORE retrieval = salience alone drives it.
  4. TABLE B   — paired Step1->Step2 shift of over_cult_general (THE headline),
                 plus raw-Q1 shift and over_cult_clue shift.
  --dump-items writes per-item states for the faithfulness parser.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict, Counter
from typing import Dict, List

import judge_rubric as R

PROMPTS = ["story_country", "story_country_value", "story_rot"]
CLUE_PROMPTS = ["story_country_value", "story_rot"]
DIMS = ["cultural_reasoning", "used_given_clue", "over_cult_general", "over_cult_clue"]


def _norm_generic(p):
    o = {}
    for l in open(p):
        if l.strip():
            d = json.loads(l)
            if "custom_id" in d and "text" in d:
                o[d["custom_id"]] = d["text"]
    return o


def _norm_anthropic(p):
    o = {}
    for l in open(p):
        if not l.strip():
            continue
        d = json.loads(l)
        cid = d.get("custom_id")
        msg = (d.get("result", {}) or {}).get("message", {})
        o[cid] = "".join(c.get("text", "") for c in msg.get("content", []) if isinstance(c, dict)) if cid else ""
    return o


def _norm_openai(p):
    o = {}
    for l in open(p):
        if not l.strip():
            continue
        d = json.loads(l)
        cid = d.get("custom_id")
        ch = (d.get("response", {}) or {}).get("body", {}).get("choices", [{}])
        o[cid] = ch[0].get("message", {}).get("content", "") if ch and cid else ""
    return o


def _norm_gemini(p):
    o = {}
    for l in open(p):
        if not l.strip():
            continue
        d = json.loads(l)
        cid = d.get("custom_id") or d.get("key")
        cand = d.get("response", d).get("candidates", [{}])
        parts = cand[0].get("content", {}).get("parts", [{}]) if cand else [{}]
        if cid:
            o[cid] = "".join(x.get("text", "") for x in parts)
    return o


def load_results(p):
    for fn in (_norm_generic, _norm_anthropic, _norm_openai, _norm_gemini):
        try:
            r = fn(p)
            if r:
                return r
        except Exception:
            pass
    return {}


def cohen_kappa(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    p1, p2 = sum(a) / n, sum(b) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (1.0 if po == 1.0 else 0.0) if pe == 1.0 else (po - pe) / (1 - pe)


def pct(a, b):
    return sum(1 for x, y in zip(a, b) if x == y) / len(a) if a else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="judge_pilot.jsonl")
    ap.add_argument("--judge", nargs=2, action="append", metavar=("NAME", "FILE"), required=True)
    ap.add_argument("--id-map", action="append", default=[])
    ap.add_argument("--frontier", default=None,
                    help="comma-separated judge names that count as 'frontier' "
                         "(e.g. anthropic,gemini,openai). Enables frontier-majority "
                         "vs open-judge agreement + a frontier_majority column.")
    ap.add_argument("--dump-items", default=None)
    ap.add_argument("--dump-labels", default=None,
                    help="CSV: one row per (item, step) with EACH judge's label + "
                         "the majority + a blank human column, for human-val later.")
    args = ap.parse_args()

    id_map: Dict[str, str] = {}
    for mp in args.id_map:
        try:
            id_map.update(json.loads(open(mp).read()))
        except OSError:
            print(f"[warn] cannot read id_map {mp}", file=sys.stderr)
    resolve = lambda c: id_map.get(c, c)
    pilot = {it["pilot_id"]: it for it in (json.loads(l) for l in open(args.pilot) if l.strip())}

    judges: Dict[str, Dict[str, Dict]] = {}
    for name, path in args.judge:
        raw = load_results(path)
        parsed, bad = {}, 0
        for cid, text in raw.items():
            ref = resolve(cid)
            if "||" not in ref:
                bad += 1; continue
            pid, which = ref.split("||")
            it = pilot.get(pid)
            if not it:
                continue
            p = R.parse_judge_response(text, expected_keys=R.score_keys_for(which, it))
            if p is None:
                bad += 1; continue
            parsed[ref] = p
        judges[name] = parsed
        print(f"[{name}] {len(raw)} results, {len(parsed)} parsed, {bad} unparseable")

    names = [n for n in judges if judges[n]]
    if not names:
        print("\nNo judge results parsed.")
        return 1
    common = set().union(*[set(judges[n]) for n in names])

    print("\n" + "=" * 70)
    print("1. INTER-JUDGE AGREEMENT")
    print("=" * 70)
    for dim in DIMS:
        head = False
        for n1, n2 in itertools.combinations(names, 2):
            ids = [c for c in common if dim in judges[n1].get(c, {}) and dim in judges[n2].get(c, {})]
            if not ids:
                continue
            if not head:
                print(f"\n  {dim}:"); head = True
            a = [judges[n1][c][dim] for c in ids]; b = [judges[n2][c][dim] for c in ids]
            print(f"    {n1:<8} vs {n2:<8} n={len(ids):<5} agree={pct(a,b):.1%} kappa={cohen_kappa(a,b):.3f}")
    if len(names) == 1:
        print(f"  (single judge '{names[0]}' — no agreement)")

    # ---- helpers for categorical/boolean majority over an arbitrary judge set ----
    def maj_over(cid, dim, judge_set):
        vals = [judges[n][cid][dim] for n in judge_set if dim in judges[n].get(cid, {})]
        if not vals:
            return None
        if all(isinstance(v, bool) for v in vals):
            t = sum(vals); f = len(vals) - t
            return None if t == f else (t > f)
        c = Counter(vals).most_common()
        return None if (len(c) > 1 and c[0][1] == c[1][1]) else c[0][0]

    # ---- 1b. PER-ANSWER-MODEL agreement (self-preference check) ----
    answer_models = sorted({pilot[c.split("||")[0]]["model"] for c in common
                            if c.split("||")[0] in pilot})
    if len(names) > 1 and len(answer_models) > 1:
        print("\n" + "=" * 70)
        print("1b. AGREEMENT BY ANSWER-MODEL (does judge agreement depend on whose")
        print("    output is judged? watch a judge's own family vs others.)")
        print("=" * 70)
        for m in answer_models:
            m_ids = [c for c in common if pilot.get(c.split("||")[0], {}).get("model") == m]
            print(f"\n  -- answer-model: {m}  (items={len(m_ids)}) --")
            for dim in DIMS:
                row = []
                for n1, n2 in itertools.combinations(names, 2):
                    ids = [c for c in m_ids if dim in judges[n1].get(c, {}) and dim in judges[n2].get(c, {})]
                    if len(ids) < 5:
                        continue
                    a = [judges[n1][c][dim] for c in ids]; b = [judges[n2][c][dim] for c in ids]
                    row.append(f"{n1[:3]}/{n2[:3]}={cohen_kappa(a,b):.2f}")
                if row:
                    print(f"    {dim:<20} " + "  ".join(row))

    # ---- 1c. FRONTIER-MAJORITY vs OPEN judges ----
    frontier = [x.strip() for x in args.frontier.split(",")] if args.frontier else []
    frontier = [f for f in frontier if f in names]
    open_judges = [n for n in names if n not in frontier]
    if frontier and open_judges:
        print("\n" + "=" * 70)
        print(f"1c. FRONTIER-MAJORITY ({'+'.join(frontier)}) vs OPEN judge(s)")
        print("    does the voted frontier panel agree with the open/local judge?")
        print("=" * 70)
        for dim in DIMS:
            print(f"\n  {dim}:")
            for oj in open_judges:
                ids = [c for c in common
                       if maj_over(c, dim, frontier) is not None and dim in judges[oj].get(c, {})]
                if len(ids) < 5:
                    continue
                a = [maj_over(c, dim, frontier) for c in ids]
                b = [judges[oj][c][dim] for c in ids]
                print(f"    frontier-maj vs {oj:<8} n={len(ids):<5} "
                      f"agree={pct(a,b):.1%} kappa={cohen_kappa(a,b):.3f}")

    def cons(cid, dim):
        v = [judges[n][cid][dim] for n in names if dim in judges[n].get(cid, {})]
        return None if not v else (sum(v) > len(v) / 2)

    # ---- TABLE A ----
    print("\n" + "=" * 70)
    print("2. TABLE A — marginal rates by model x prompt x step (consensus)")
    print("=" * 70)
    cult = defaultdict(list); clue = defaultdict(list)
    ocg = defaultdict(list); occ = defaultdict(list)
    for cid in common:
        pid, which = cid.split("||"); it = pilot.get(pid)
        if not it:
            continue
        k = (it["model"], it["prompt_type"], which)
        for store, dim in ((cult, "cultural_reasoning"), (clue, "used_given_clue"),
                           (ocg, "over_cult_general"), (occ, "over_cult_clue")):
            v = cons(cid, dim)
            if v is not None:
                store[k].append(v)
    print(f"\n  {'model':<13}{'prompt':<22}{'step':<10}{'cultural%':>10}{'usedclue%':>10}"
          f"{'ocGEN%':>9}{'ocClue%':>9}{'n':>5}")
    for model in sorted({k[0] for k in cult}):
        for pt in PROMPTS:
            for which in ("baseline", "atlas"):
                k = (model, pt, which)
                if k not in cult:
                    continue
                cr = sum(cult[k]) / len(cult[k])
                uc = f"{sum(clue[k])/len(clue[k]):.0%}" if clue.get(k) else "n/a"
                og = f"{sum(ocg[k])/len(ocg[k]):.0%}" if ocg.get(k) else "n/a"
                oc = f"{sum(occ[k])/len(occ[k]):.0%}" if occ.get(k) else "n/a"
                print(f"  {model:<13}{pt:<22}{which:<10}{cr:>10.1%}{uc:>10}{og:>9}{oc:>9}{len(cult[k]):>5}")

    # ---- SALIENCE LADDER ----
    print("\n" + "=" * 70)
    print("3. SALIENCE LADDER — over_cult_general & Q1 by prompt x step")
    print("   rung1 SRoT/baseline (no country) < rung2 SC,SCV/baseline (country,")
    print("   no retrieval) < rung3 SC,SCV/atlas (+retrieval). Rise 1->2 = salience")
    print("   alone; rise 2->3 = retrieval adds.")
    print("=" * 70)
    print(f"\n  {'model':<13}{'rung':<34}{'Q1cultural%':>12}{'ocGEN%':>9}{'n':>5}")
    ladder = [("rung1: SRoT, no country", [("story_rot", "baseline")]),
              ("rung2: country mentioned, no retr.", [("story_country", "baseline"),
                                                      ("story_country_value", "baseline")]),
              ("rung3: country + retrieval", [("story_country", "atlas"),
                                              ("story_country_value", "atlas")])]
    for model in sorted({k[0] for k in cult}):
        for label, cells in ladder:
            cr = [v for (pt, w) in cells for v in cult.get((model, pt, w), [])]
            og = [v for (pt, w) in cells for v in ocg.get((model, pt, w), [])]
            if not cr:
                continue
            ogs = f"{sum(og)/len(og):.1%}" if og else "n/a"
            print(f"  {model:<13}{label:<34}{sum(cr)/len(cr):>12.1%}{ogs:>9}{len(cr):>5}")

    # ---- TABLE B: paired shift ----
    def shift(metric, fn, prompts):
        print(f"\n  >> shift of {metric}, Step1->Step2:")
        print(f"     {'model':<13}{'prompt':<22}{'c→c':>6}{'HARMED':>8}{'helped':>8}{'oc→oc':>7}{'netΔ':>8}{'n':>5}")
        for model in sorted({pilot[p.split('||')[0]]['model'] for p in common}):
            for pt in prompts:
                cc = h = hl = oo = n = 0
                for pid, it in pilot.items():
                    if it["model"] != model or it["prompt_type"] != pt:
                        continue
                    b, a = f"{pid}||baseline", f"{pid}||atlas"
                    if b not in common or a not in common:
                        continue
                    vb, va = fn(b), fn(a)
                    if vb is None or va is None:
                        continue
                    n += 1
                    cc += (not vb and not va); h += (not vb and va)
                    hl += (vb and not va);     oo += (vb and va)
                if n:
                    print(f"     {model:<13}{pt:<22}{cc:>6}{h:>8}{hl:>8}{oo:>7}{(h-hl)/n:>+8.1%}{n:>5}")

    print("\n" + "=" * 70)
    print("4. TABLE B — paired shift (HEADLINE). HARMED=clean→over-cult; netΔ>0 =")
    print("   external knowledge INDUCED over-culturalization.")
    print("=" * 70)
    shift("over_cult_general (Q1∧Q3)", lambda c: cons(c, "over_cult_general"), PROMPTS)
    shift("cultural_reasoning (raw Q1)", lambda c: cons(c, "cultural_reasoning"), PROMPTS)
    shift("over_cult_clue (Q1∧¬Q2)", lambda c: cons(c, "over_cult_clue"), CLUE_PROMPTS)

    if args.dump_items:
        with open(args.dump_items, "w") as f:
            for pid, it in pilot.items():
                a = f"{pid}||atlas"; b = f"{pid}||baseline"
                if a not in common:
                    continue
                f.write(json.dumps({
                    "pilot_id": pid, "model": it["model"], "prompt_type": it["prompt_type"],
                    "transition": it.get("transition"),
                    "q1_atlas": cons(a, "cultural_reasoning"),
                    "ocg_baseline": cons(b, "over_cult_general") if b in common else None,
                    "ocg_atlas": cons(a, "over_cult_general"),
                }) + "\n")
        print(f"\n[dump] -> {args.dump_items}")

    if args.dump_labels:
        import csv
        from collections import Counter as _C
        BOOL_DIMS = ["cultural_reasoning", "used_given_clue",
                     "over_cult_general", "over_cult_clue"]
        CAT_DIMS = ["general_sufficient"]
        DIMS_ORDER = ["cultural_reasoning", "used_given_clue", "general_sufficient",
                      "over_cult_general", "over_cult_clue"]

        def majority(vals):
            """Majority label. Bools: True/False, 'tie' on an even split.
            Categoricals: the mode, 'tie' when the top two are level. '' if no votes."""
            if not vals:
                return ""
            if all(isinstance(v, bool) for v in vals):
                t = sum(vals); f = len(vals) - t
                return "tie" if t == f else (t > f)
            c = _C(vals).most_common()
            return "tie" if (len(c) > 1 and c[0][1] == c[1][1]) else c[0][0]

        ctx = ["pilot_id", "model", "prompt_type", "transition", "step",
               "scenario", "explanation"]
        front = [x.strip() for x in args.frontier.split(",")] if args.frontier else []
        front = [f for f in front if f in names]
        cols = list(ctx)
        for d in DIMS_ORDER:
            cols += [f"{d}_{n}" for n in names] + [f"{d}_majority"]
            if front:
                cols += [f"{d}_frontiermaj"]
            cols += [f"{d}_human"]

        with open(args.dump_labels, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for ref in sorted(common):
                pid, which = ref.split("||")
                it = pilot.get(pid)
                if not it:
                    continue
                expl = (it.get("baseline_explanation" if which == "baseline"
                               else "atlas_explanation") or "").strip()
                scen = "; ".join(p for p in [
                    f"country={it.get('country')}" if it.get("country") else "",
                    f"rot={it.get('rot')}" if (it.get("rot") or "").strip() else "",
                    f"value={it.get('value')}" if (it.get("value") or "").strip() else "",
                    f"story={it.get('story')}" if it.get("story") else "",
                ] if p)
                row = {"pilot_id": pid, "model": it.get("model"),
                       "prompt_type": it.get("prompt_type"),
                       "transition": it.get("transition"), "step": which,
                       "scenario": scen, "explanation": expl}
                for d in DIMS_ORDER:
                    vals, fvals = [], []
                    for n in names:
                        v = judges[n].get(ref, {}).get(d, "")
                        row[f"{d}_{n}"] = v
                        if v != "" and d in (BOOL_DIMS + CAT_DIMS) and v is not None:
                            vals.append(v)
                            if n in front:
                                fvals.append(v)
                    row[f"{d}_majority"] = majority(vals)
                    if front:
                        row[f"{d}_frontiermaj"] = majority(fvals)
                    row[f"{d}_human"] = ""        # fill in by hand for the val slice
                w.writerow(row)
        print(f"[labels] -> {args.dump_labels}  (one row per item×step; "
              f"fill the *_human columns for the validation slice)")
    return 0


if __name__ == "__main__":
    sys.exit(main())