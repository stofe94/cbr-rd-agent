# CBR-Enhanced RD-Agent for Kaggle Data Science

> **An extension of [Microsoft's RD-Agent](https://github.com/microsoft/RD-Agent) that adds Case-Based Reasoning (CBR), a Google AI Studio backend (Gemma 4), and production-ready Docker tooling for autonomous Kaggle competition solving.**

---

## ⚠️ Current Status & Scope

This project is a **first implementation** and is specifically designed and validated for two Kaggle competitions. While the agent is architecturally capable of running on other MLE-bench competitions, **the CBR methodology has not yet been generalized** to work reliably across all ~75 MLE-bench competitions. Adaptation of the similarity heuristics, quality gates, and retrieval tuning to new competition domains requires additional work.

**Use on other competitions at your own risk — results may vary significantly.**

---

## What Is This?

[RD-Agent](https://github.com/microsoft/RD-Agent) by Microsoft Research is an autonomous R&D framework that uses LLMs to iteratively propose, implement, and evaluate data science experiments. This project extends it with:

- **Case-Based Reasoning (CBR):** The agent learns from its own experiment history. Successful solutions are stored as structured "cases" and retrieved at proposal time to guide future hypotheses — avoiding known failures and reusing proven techniques.
- **Google AI Studio Backend:** A custom backend that connects RD-Agent to Google AI Studio's API, enabling the use of **Gemma 4 (31B)** as the reasoning model and `gemini-embedding-001` for semantic retrieval.
- **A reduced RD-Agent for Docker:** only the data science scenario remains, with Docker-in-Docker support, live Kaggle leaderboard evaluation and automatic cleanup of empty runs (see [Changes to RD-Agent Internals](#changes-to-rd-agent-internals)).

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────┐
│                  CBRDataScienceRDLoop                     │
│         (extends DataScienceRDLoop from RD-Agent)         │
│                                                           │
│  direct_exp_gen() ──► builds ONE shared query             │
│                       retrieves similar Cases from KB     │
│                       retrieves relevant failure patterns │
│                       injects both into hypothesis.       │
│                       appendix (not hypothesis.reason)    │
│                       tracks token usage per phase        │
│                                                           │
│  coding()         ──► injects full case code into task    │
│                       descriptions (sub_tasks, pending,   │
│                       pipeline task, or cbr_reference.md) │
│                                                           │
│  feedback()       ──► passes result through 5-gate        │
│                       QualityGate; retains Cases to KB    │
│                       tracks token usage per phase        │
└────────────┬──────────────────────────────┬───────────────┘
             │                              │
   ┌──────────▼──────────┐      ┌───────────▼───────────┐
   │   CBRKnowledgeBase  │      │    FailureTracker     │
   │  cases.json         │      │  failure_patterns.json│
   │  code/{case_id}.py  │      │  failure_vectors.pkl  │
   │  vector_base.pkl    │      └───────────────────────┘
   └──────────┬──────────┘
              │  wraps
   ┌──────────▼──────────┐      ┌───────────────────────┐
   │  QualityGate        │      │    case_schema.py     │
   │  Gate 1: Execution  │      │  Case                 │
   │  Gate 2: Metric     │      │  ProblemSignature     │
   │  Gate 3: Improvement│      │  SolutionSignature    │
   │  Gate 4: Novelty    │      │  CaseMetrics          │
   │  Gate 5: Generalize │      │  CaseOutcome          │
   └─────────────────────┘      └───────────────────────┘
```

---

## Core Components (`rdagent/scenarios/data_science/cbr/`)

### `cbr_loop.py` — CBR-Enhanced R&D Loop

The central integration point. Subclasses RD-Agent's `DataScienceRDLoop` and overrides three methods:

- At **proposal time** (`direct_exp_gen()`): builds a single shared embedding query and passes it to both `cbr_kb.retrieve()` and `failure_tracker.get_relevant()` — warming both caches in one embedding call. Retrieved cases and failure patterns are injected into the hypothesis via `hypothesis.appendix` (not `hypothesis.reason`) to keep plan summaries clean. Loop caches are explicitly flushed at the start of each new iteration via `_flush_loop_caches()` to prevent stale cross-iteration hits.
- At **coding time** (`coding()`): injects the full code of relevant past cases directly into the task descriptions of sub-tasks, pending tasks, and pipeline tasks of the current experiment. If none of those injection paths match the experiment structure, the code is written as `cbr_reference.md` into the experiment workspace as a fallback. This gives the coding LLM concrete working implementations to reference, not just high-level hints. Task-type and data-type normalization tables (`_TASK_TYPE_GROUPS`, `_DATA_TYPE_GROUPS`) ensure that semantically equivalent labels (e.g. `binary_classification` and `tabular_classification`) are matched correctly when deciding which cases are relevant for injection.
- At **feedback time** (`feedback()`): passes the experiment result through the QualityGate pipeline and, if retained, stores the case in the knowledge base.
- **Token usage tracking**: records prompt tokens, completion tokens, and embedding tokens per phase (`propose` / `feedback`) with per-model breakdown, logged at the end of each loop iteration.
- **Token budget constants**: all display-time truncation limits are configurable via env vars (`CBR_PLAN_SUMMARY_CHARS`, `CBR_QUERY_DESC_CHARS`, `CBR_HYPOTHESIS_CODE_CHARS`, `CBR_CODING_CODE_CHARS`, `CBR_GATE5_CODE_CHARS`). Nothing is truncated at store time — all KB fields are persisted at full length. Code injection in the coding phase defaults to no truncation (`CBR_CODING_CODE_CHARS=0`).
- Includes an **A/B testing switch** (`CBR_DISABLE_RETRIEVAL=1`) that disables all three injection points while keeping token tracking, case storage, and quality gate logic active — allowing direct comparison of CBR vs. non-CBR runs with identical seeds.

### `case_schema.py` — Case Data Model

Defines the structured `Case` object that wraps an RD-Agent `DSExperiment` with CBR metadata:

- `ProblemSignature`: what kind of problem this is (competition name, metric type, data characteristics)
- `SolutionSignature`: what approach was used (model type, techniques, feature engineering)
- `CaseMetrics`: numeric results (score, baseline delta, CV results)
- `CaseOutcome`: `success`, `partial`, `failure`, or `untested`
- `plan_summary`: built from the structured `concise_*` fields of `DSHypothesis` (not from `hypothesis.reason`, which contains injected CBR context via `appendix`)
- Provenance tracking: every case records which parent case it was adapted from
- All cases are **JSON-serializable** (no pickle) for transparency and debuggability. Code snapshots are stored separately as `.py` files. Nothing is truncated at store time — display-time truncation only happens in `cbr_loop.py`.

### `case_knowledge.py` — Vector Knowledge Base

Wraps RD-Agent's `PDVectorBase` with CBR-specific logic:

- Embedding-based semantic retrieval with MMR (Maximal Marginal Relevance) to balance similarity and diversity
- Automatic consistency verification on startup: re-embeds missing cases, removes orphan vector rows
- Smart cache invalidation: retrieval results are cached per query hash; `add_case()` invalidates, but reuse-stat updates do not (so Gate 4 novelty checks are free within the same loop iteration)
- Competition isolation: passes `constraint_labels` to prevent RD-Agent's `KaggleExperienceBase` rows from polluting CBR retrieval

### `quality_gate.py` — 5-Gate Retention Pipeline

Decides whether a completed experiment is worth storing as a case:

| Gate | Name | What it checks |
|------|------|----------------|
| 1 | Execution | No runtime errors |
| 2 | Metric | A numeric result was extracted |
| 3 | Improvement | Beats the baseline or first result |
| 4 | Novelty | Not a near-duplicate (cosine similarity < `CBR_DEDUP_THRESHOLD`) |
| 5 | Generalization | LLM extracts transferable techniques (the "lesson learned") |

Gate 4 reuses the embedding query cached during `direct_exp_gen()` — zero extra API calls.

### `failure_tracker.py` — Failure Pattern Memory

Stores runtime failures with embeddings so the agent can answer *"what approaches have failed for problems like mine?"* at the next proposal step. Uses a rolling window of 200 failures. Cache is invalidated on every new failure log.

---

## Google AI Studio Backend (`rdagent/oai/backend/google_ai_studio.py`)

A custom `APIBackend` implementation for RD-Agent that routes all LLM and embedding calls to Google AI Studio instead of OpenAI. Key features:

- **Model health tracking**: per-model response time and timeout rate statistics with adaptive timeouts per attempt; unhealthy models are ranked lower for fallback selection
- **Stream inactivity detection**: chunk-level hang detection (configurable inactivity threshold), distinct from the wall-clock request timeout
- **Dynamic token estimation**: auto-adjusts `max_tokens` per call based on whether the response schema contains code fields
- **JSON repair pipeline**: structured output responses go through `_strip_thinking_channel_tags` → `_sanitize_json_string_escapes` → `_repair_json_for_schema` before being returned to the caller, handling common Gemma 4 output quirks
- **Schema compression**: `_compress_schema_for_small_model` strips titles, descriptions, defaults, and examples from JSON schemas to reduce token overhead on structured calls
- **Output integrity guard**: `_apply_output_integrity_guard` validates structured outputs and flags degenerate repetitions before returning to the caller
- **Smart retry wrapper**: integrates with RD-Agent's retry pattern via `wrap_google_call`

---

## Changes to RD-Agent Internals

This project is a fork of RD-Agent, reduced to the data science scenario. All
modifications are made directly to the source, and every part of RD-Agent that
the data science loop needs is included in this repository.
Changes beyond the `cbr/` module and the Google backend include, at a high level:

- **Reduced to the data science scenario**: the other RD-Agent scenarios
  (Qlib finance, LLM fine-tuning, reinforcement learning, general model
  extraction and the legacy Kaggle loop) are removed, together with their
  coders, benchmarks, Docker images and the Flask log server. The `rdagent` CLI
  provides `data_science`, `grade_summary`, `ui`, `health_check` and
  `collect_info`; `rdagent ui` always starts the data science UI
- **Docker-in-Docker support**: substantial changes to make the agent's internal
  experiment execution work reliably within a containerized host environment,
  including socket forwarding, path resolution, and container lifecycle handling
- **Log and result persistence**: restructured to separate competition runs more
  cleanly and produce more readable output, with terminal output captured to a
  dedicated log file alongside the standard RD-Agent log directory
- **Automatic cleanup**: integrated auto-cleanup of failed, empty, or
  result-less experiment runs to keep the workspace and log directories lean
- **Live Kaggle leaderboard evaluation**: added live Kaggle leaderboard scoring
  as a parallel evaluation path alongside the existing MLE-bench result tracking,
  allowing real submission-based feedback during runs
- **UI adjustments**: various changes to the RD-Agent UI for clearer result
  display and experiment overview
- Various smaller fixes and configuration additions throughout the data science
  scenario and connected modules

---

## Getting Started

### Prerequisites

> 💡 **Local installation (without Docker):** see [INSTALL.md](INSTALL.md) for
> a step-by-step guide covering Windows (WSL2) and native Linux. No separate
> RD-Agent installation needed — this repository includes all RD-Agent code the
> data science loop depends on.

- Docker (with access to `/var/run/docker.sock`)
- A Google AI Studio API key with access to Gemma 4 and `gemini-embedding-001`
- Kaggle API credentials (for competition data); accept the rules of each competition
  on kaggle.com first ("Join Competition"), otherwise the data download fails

The Docker image is built locally from this repository; `exe_docker_linux.sh`
does that on each start (Docker's layer cache keeps rebuilds short). To build it
by hand:

```bash
docker build -t local_cbr-rdagent:latest .
```

### Configuration

Copy `config.env.example` to `~/tmp/config.env` and adjust as needed. The file contains sensible defaults for all tuning parameters — the most important ones are:

```bash
# Backend / model
BACKEND=rdagent.oai.backend.google_ai_studio.GoogleAIStudioAPIBackend
GOOGLE_AI_STUDIO_CHAT_MODEL=gemma-4-31b-it
GOOGLE_AI_STUDIO_EMBEDDING_MODEL=gemini-embedding-001

# CBR loop (comment out and use DataScienceRDLoop to disable CBR entirely)
DS_RD_LOOP=rdagent.scenarios.data_science.cbr.cbr_loop.CBRDataScienceRDLoop

# CBR tuning
CBR_TOP_K=2
CBR_CASE_SIMILARITY_THRESHOLD=0.5
CBR_DEDUP_THRESHOLD=0.92
CBR_DISABLE_RETRIEVAL=0   # set to 1 for ablation / baseline run
```

Copy `secrets.env.example` to `~/tmp/secrets.env` and fill in your credentials:

```bash
GOOGLE_AI_STUDIO_API_KEY=...
KAGGLE_USERNAME=...
KAGGLE_KEY=...
GHCR_TOKEN=          # leave empty; only the maintainer pulls a pre-built image
```

> ⚠️ `secrets.env` must **never** be committed. Verify it is listed in `.gitignore` before your first `git add`.

### Running

`exe_docker_linux.sh` expects the following files to exist in `~/tmp/` before
launch; they are mounted into the container at runtime (the image contains only
the `.example` versions):

```
~/tmp/
├── config.env       # agent configuration
└── secrets.env      # credentials
```

The script also creates the persistent workspace and log directories
automatically if they don't exist yet:

```
~/cbr-rdagent/
├── workspace/
│   └── knowledge_base/
└── log/
```

Then run:

```bash
bash exe_docker_linux.sh
# Prompts: "Which competition to run? (e.g. spaceship-titanic)"
```

The script will:
1. Build the image from this repository (the maintainer's `GHCR_TOKEN` pulls a
   private pre-built image instead)
2. Start the container with workspace, log, and Docker socket (`/var/run/docker.sock`) mounted
3. Source `config.env` and `secrets.env` inside the container
4. Execute `run_linux.sh` from inside the image (`/root/run_linux.sh`), substituting
   the competition name at runtime via `sed` — no manual file placement needed for
   the entrypoint script

The container runs the agent only. To follow a run in the RD-Agent UI, start it
in the running container and open `http://localhost:19899` (the script publishes
the port):

```bash
docker exec -it cbr-rdagent rdagent ui --data-science --log-dir=./log
```

On Windows with Docker Desktop, `exe_docker_windows.bat` does the same, with the
configuration in `%USERPROFILE%\tmp\` and the workspace in `%USERPROFILE%\cbr-rdagent\`.

### Inspecting Results

```bash
bash run_inspect.sh
```

Clears the pickle cache, runs `rdagent grade_summary` over `./log` to produce a
structured results overview, and opens the RD-Agent UI on that log directory. It
needs the local installation ([INSTALL.md](INSTALL.md)) and runs in the directory
that contains `log/` (for Docker runs: `~/cbr-rdagent`).

---

## CBR A/B Testing

To compare CBR vs. baseline with the same competition and seed, run
`exe_docker_linux.sh` twice and switch retrieval in `~/tmp/config.env` in between
(the container reads its settings from that file, not from the calling shell):

```bash
# Baseline: retrieval disabled, but case storage still active
CBR_DISABLE_RETRIEVAL=1

# CBR-enabled
CBR_DISABLE_RETRIEVAL=0
```

Compare best metric, iterations-to-best, token totals, and failure rate across runs.

---

## Citation

The method and its evaluation are described in the accompanying paper
([arXiv:2606.05250](https://arxiv.org/abs/2606.05250)). If you use CBR-RD-Agent,
please cite it (see also [CITATION.cff](CITATION.cff)):

```bibtex
@misc{stocker_cbr_rdagent_2026,
  author        = {Stocker, Felix},
  title         = {Towards Persistent Case-Based Memory for Autonomous Data Science:
                   A CBR-Augmented R\&D-Agent with a Locally Deployable Small Language Model},
  year          = {2026},
  eprint        = {2606.05250},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2606.05250}
}
```

## Contributing

Suggestions are welcome as issues or pull requests; see [CONTRIBUTING.md](CONTRIBUTING.md).
Changes per release are listed in [CHANGELOG.md](CHANGELOG.md).

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

### Required Attribution

If you fork, extend, or redistribute this project (in whole or in part), the MIT License
requires you to preserve the following copyright notice in your `LICENSE` file:

```
Copyright (c) 2026 stofe94
Portions Copyright (c) Microsoft Corporation (RD-Agent, https://github.com/microsoft/RD-Agent)
```

This is the same requirement this project fulfills for its own upstream dependency,
Microsoft's RD-Agent — see *Third-Party Attribution* below.

### Third-Party Attribution

This project builds on **[Microsoft RD-Agent](https://github.com/microsoft/RD-Agent)**, which is also licensed under the MIT License.

```
MIT License
Copyright (c) Microsoft Corporation
```

Per the MIT License terms, the original copyright notice is preserved above. This project is an independent extension and is not affiliated with, endorsed by, or supported by Microsoft.

The Google AI Studio backend uses the **[Google Gen AI Python SDK](https://github.com/googleapis/python-genai)** (`google-genai`), licensed under the Apache 2.0 License.

---

## Disclaimer

This project is not affiliated with Microsoft. The RD-Agent components used here are provided "as is" without warranty. See [Microsoft's RD-Agent disclaimer](https://github.com/microsoft/RD-Agent#disclaimer) for the upstream project's terms.

---

## Author

**Felix Stocker**, data scientist (CAE, crash simulation, deep learning on 3D geometries)

- E-mail: [felix.stocker@sto-eng.de](mailto:felix.stocker@sto-eng.de)
- LinkedIn: [linkedin.com/in/felixstocker](https://www.linkedin.com/in/felixstocker)
- Competency profile: [stofe94.github.io/competency_profile](https://stofe94.github.io/competency_profile/)
