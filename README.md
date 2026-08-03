# C-RES: Code Supplement

This supplement contains the core code for **C-RES**, our study of
*Cultural overeach* in cultural-norm reasoning on NormAd-ETI. It covers the
full pipeline: baseline evaluation, the three evidence-retrieval paths, the
external-knowledge (KB-grounding) reproduction, the LLM-judge that diagnoses
*Cultural overeach*

--------------------------------------------------------------------------------
## Repository layout

```
pipeline/         Step 1 baseline + Step 2 trajectory collection, prompt formatting
coordinators/     The reasoning engines (static synthesis, agentic, ReAct, KB, KB-synth)
tools/            The three cultural tools + tool registry/config + base classes
judge/            Over-culturalisation LLM judge: sampling, running, parsing, validation
prompts/          Prompt templates (verbatim from the code) and the judge rubric
```

--------------------------------------------------------------------------------
## Pipeline order

**Step 1 — Baseline (no evidence).**
`pipeline/step1_evaluate_baseline.py` runs every NormAd-ETI scenario under all
three prompt types (SC, SCV, SRoT) with no tools, establishing what each model
knows from pre-training. Prompt strings come from
`pipeline/prompt_formatter.py`.

**Step 2 — Evidence retrieval (three paths).** Each path starts from the Step 1
baseline and adds cultural evidence:
- *Static synthesis* — `pipeline/step2_collect_trajectories.py` with
  `coordinators/coordinator.py`. A reading turn compresses Cultural Atlas /
  Wikipedia evidence to scenario-relevant points; a synthesis turn then revises
  the answer (Hofstede passes through directly).
- *Agentic (one-shot selection)* — `pipeline/step2_agentic_collect.py` with
  `coordinators/coordinator_agentic.py`. The model selects 0-3 tools up front.
- *ReAct (iterative)* — `pipeline/step2_react_collect.py` with
  `coordinators/coordinator_react.py`. The model interleaves Thought -> Action
  -> Observation, up to three iterations, each tool at most once.

**Step 2 (KB-grounding).** The external-knowledge baseline and our variant:
- *KB / KB-selective* — `pipeline/step2_kb_collect.py` with
  `coordinators/coordinator_kb.py`.
- *KB-synth (our addition)* — `pipeline/step2_kb_synth_collect.py` with
  `coordinators/coordinator_kb_synth.py`, which routes retrieved KB evidence
  through the same reading -> synthesis path as the static tools, so that the
  *handling* of evidence is held constant and only the *source* differs.

**Judge — over-culturalisation.**
`judge/build_judge_sample.py` builds the per-item baseline-vs-evidence pairs;
`judge/judge_batch_submit.py` (API judges) and `judge/judge_local_run.py`
(open judge, Qwen3-32B) run them; `judge/judge_fetch.py` and
`judge/judge_batch_parse.py` collect and parse the labels; and
`judge/sample_for_annotation.py` + `judge/validate_human_agreement.py` handle
the human-validation study.

--------------------------------------------------------------------------------
## Knowledge base (KB-grounding)

We reproduce the prior KB-grounding baseline with an open, reproducible stack.
`knowledge_base/build_kb.py` fuses the three sources into a single doc store
with stable ids and source tags; `knowledge_base/build_index.py` embeds them
and builds a FAISS index; `knowledge_base/kb_retriever.py` performs dense
retrieval (top-5) at run time.

- **Embedding model:** `Qwen/Qwen3-Embedding-4B` (native dimensionality 2560,
  truncated to 1024 before indexing), replacing the retired
  `textembedding-gecko@003`.
- **Index:** FAISS (dense).

The per-source counts are printed by `build_kb.py` when the index is built; use
those printed numbers as the authoritative figures rather than any values
quoted in comments.

--------------------------------------------------------------------------------
## Models

Six open-weight answer models are evaluated. Frontier API models are used as
baselines and as judges. Exact checkpoints:

| Paper name | Identifier |
|------------|------------|
| Q-4B       | `Qwen/Qwen3-4B-Instruct-2507` |
| Q-30B      | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| G-4B       | `google/gemma-4-E4B-it` |
| G-31B      | `google/gemma-4-31B-it` |
| R1-7B      | DeepSeek-R1-Distill-Qwen-7B |
| R1-32B     | DeepSeek-R1-Distill-Qwen-32B |

The open judge is `Qwen/Qwen3-32B` (a distinct checkpoint from the answer
models; the validation script reports its human agreement separately on
Qwen-written vs other explanations to control for self-preference).

--------------------------------------------------------------------------------
## Environment

- Python 3.10+
- Local models via HuggingFace `transformers` (large models need 2 GPUs); the
  open judge can also run this way.
- FAISS for the KB index; `sentence-transformers`/`transformers` for embeddings.
- API judges/baselines via the respective providers' batch endpoints (keys read
  from the environment).

Inputs (NormAd-ETI, the KB source files) are not redistributed here and are
supplied by path at the command line. Each script's module docstring gives its
exact invocation.

--------------------------------------------------------------------------------
## A note on the prompt templates

The prompt templates in `prompts/PROMPTS.md` are presented as in the paper's
appendix. The runnable code in `pipeline/` and `coordinators/` contains the
exact strings the models received, which include a few additional formatting
lines (an explicit numbered options block and a short "do not over-infer"
instruction) that the appendix omits for brevity; these do not change the task.
