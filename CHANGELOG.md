# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [0.1.0] — 2026-09-25

First public release: Microsoft's RD-Agent, reduced to its data science scenario and
extended with Case-Based Reasoning, developed and validated on two Kaggle competitions
(`spaceship-titanic`, `nomad2018-predict-transparent-conductors`).

### Added

- **Case-Based Reasoning (CBR):** `CBRDataScienceRDLoop` retrieves similar past cases
  and known failure patterns at proposal time, injects the full code of relevant cases
  at coding time, and passes every result through a five-gate quality pipeline
  (execution, metric, improvement, novelty, generalization) before storing it as a case.
- **Knowledge base:** JSON-serialized cases with separate code snapshots, embedding-based
  retrieval with MMR, consistency checks on startup, and a failure tracker for runtime
  errors.
- **A/B switch:** `CBR_DISABLE_RETRIEVAL=1` turns retrieval and injection off while case
  storage, quality gates and token tracking stay active, for CBR vs. baseline runs.
- **Google AI Studio backend:** Gemma 4 (`gemma-4-31b-it`) as reasoning model and
  `gemini-embedding-001` for retrieval, with model health tracking, stream hang
  detection, dynamic token limits, JSON repair and schema compression for small models.
- **Docker:** image with Docker-in-Docker support for the experiment containers, start
  scripts for Linux and Windows that build the image locally.
- **Evaluation:** live Kaggle leaderboard scoring next to the MLE-bench grading, per-run
  log and result separation, automatic cleanup of empty or failed runs.
- **Documentation:** README, INSTALL.md, `config.env.example`, `secrets.env.example`.

### Removed

- The RD-Agent scenarios this project does not use: Qlib finance, LLM fine-tuning,
  reinforcement learning, general model extraction and the legacy Kaggle loop, with
  their coders, benchmarks, Docker images and the Flask log server.

[0.1.0]: https://github.com/stofe94/cbr-rd-agent/releases/tag/v0.1.0
