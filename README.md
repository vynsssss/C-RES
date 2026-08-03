# C-RES: Code Supplement (vLLM)

This supplement contains the core code for **C-RES**, our study of
*cultural overeach* in cultural-norm reasoning on NormAd-ETI. The
answer-generation pipeline uses the **vLLM** backend for batched,
high-throughput inference across the six open-weight models.

The code is provided for transparency and reproducibility. We do not
redistribute NormAd-ETI or the external knowledge sources; paths to those
inputs are supplied on the command line.

--------------------------------------------------------------------------------
## Repository layout

```
pipeline/         Step 1 baseline + Step 2 collectors (vLLM)
coordinators/     Reasoning engines: base coordinators + their vLLM subclasses
tools/            The three cultural tools + registry/config + base classes
judge/            Over-culturalisation LLM judge: sampling, running, validation
prompts/          Prompt templates and the judge rubric
```

--------------------------------------------------------------------------------
## Inference backend (vLLM)

Generation runs through vLLM:

- **Step 1** (`pipeline/step1_evaluate_baseline.py`) loads the model in-process
  through vLLM (`LLM()` + `SamplingParams`) and generates the baseline answers
  in batched passes.
- **Step 2** collectors talk to a **local vLLM server** (started separately),
  through the vLLM coordinators in `coordinators/`. Each collector takes
  `--backend vllm`.

The vLLM **coordinators** subclass the shared base coordinators and override
only the single `generate()` call, so the tool logic, prompt construction,
reading/synthesis flow, and trajectory format are identical to the base
classes:

- `coordinators/vllm_local_agent.py` — `LocalVLLMAgent`, the vLLM generate
  wrapper (sampling pulled from the per-model-family table in `base_agent.py`).
- `coordinators/vllm_coordinator.py` — `VLLMCoordinator` (static synthesis
  path), subclass of `Coordinator`.
- `coordinators/vllm_coordinator_extras.py` — `VLLMAgenticCoordinator`,
  `VLLMReactCoordinator`, `VLLMKBSynthCoordinator`, subclasses of the agentic /
  ReAct / KB-synth coordinators.

The base coordinators (`coordinator.py`, `coordinator_agentic.py`, etc.) are
included because the vLLM classes extend them; they hold the reasoning logic and
do not themselves load any model.

**Note on the KB baseline.** `pipeline/step2_kb_collect.py` reproduces the
prior-work KB / KB-selective grounding and runs on HuggingFace `transformers`,
not vLLM. It is a single-pass baseline separate from the C-RES vLLM pipeline,
included here for completeness of the comparison; a header comment in the file
states this.

--------------------------------------------------------------------------------
## Pipeline order

1. **Step 1 — baseline** (`pipeline/step1_evaluate_baseline.py`): every scenario
   under all three prompt types (SC, SCV, SRoT), no evidence.
2. **Step 2 — retrieval**: from the baseline, add cultural evidence via one of
   three paths — static synthesis (`step2_collect_trajectories.py`), agentic
   one-shot selection (`step2_agentic_collect.py`), or ReAct
   (`step2_react_collect.py`) — plus the KB variants
   (`step2_kb_collect.py`, `step2_kb_synth_collect.py`).
3. **Judge** (`judge/`): score the reasoning for over-culturalisation and
   validate against human annotation.

--------------------------------------------------------------------------------
## Knowledge base

`knowledge_base/build_kb.py` fuses the sources into one doc store;
`build_index.py` embeds them with `Qwen/Qwen3-Embedding-4B` and builds a FAISS index;
`kb_retriever.py` performs top-5 dense retrieval. Use the per-source counts
printed by `build_kb.py` as the authoritative figures.

--------------------------------------------------------------------------------
## Models

| Paper name | Identifier |
|------------|------------|
| Q-4B  | `Qwen/Qwen3-4B-Instruct-2507` |
| Q-30B | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| G-4B  | `google/gemma-4-E4B-it` |
| G-31B | `google/gemma-4-31B-it` |
| R1-7B | DeepSeek-R1-Distill-Qwen-7B |
| R1-32B| DeepSeek-R1-Distill-Qwen-32B |

The open judge is `Qwen/Qwen3-32B` (`judge/judge_local_run.py`, vLLM).

--------------------------------------------------------------------------------
## Environment

- Python 3.10+
- `vllm` (answer models + local judge)
- `transformers` (tokenisers/config; and the KB baseline)
- FAISS for the KB index; `sentence-transformers`/`transformers` for embeddings
- API judges/baselines via the providers' batch endpoints (keys from env)

Paths and account names have been genericised; set them for your environment.
