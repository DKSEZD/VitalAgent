# VitalAgent: A Tool-Augmented Agent for Reactive and Proactive Physiological Monitoring over Wearable Health Data

[**Benchmark Documentation**](agent/benchmarks/vitalbench/README.md)

Official implementation of the paper *"VitalAgent: A Tool-Augmented Agent for Reactive and Proactive Physiological Monitoring over Wearable Health Data"*.

**VitalAgent** is a tool-augmented ECG/PPG agent that operates directly on raw
physiological signals for reactive health question answering and proactive
monitoring. This repository also includes **VitalBench**, the benchmark and
evaluation pipeline used for longitudinal physiological monitoring.

VitalAgent has two engines:

- **Reactive engine** (query-driven): `Planner → Tools → Validation gate → LLM agent`,
  with a re-plan loop. Answers questions about a recording window by
  dynamically invoking raw-signal analysis and state-grounded tools, then
  synthesizing a grounded answer.
- **Proactive engine** (signal-driven): streams long-term recordings
  window-by-window, maintains physiological context, applies deterministic
  **rule** checks and a guideline-grounded **LLM judge** on flagged windows,
  and emits intervention events.

## Repository layout

```
agent/
  reactive/        Reactive pipeline: planner, validation, LLM agent, replan
  proactive/       Proactive pipeline: tracker, rule layer, LLM judge
  tools/           Modality-aware tool registry
  benchmarks/
    vitalbench/    VitalBench construction
  evaluation/
    reactive/      VitalBench reactive evaluation harness
    proactive/     Proactive monitoring evaluation harness
  mhealth/         MonitoringState schema & per-dataset state builders
  data/            Dataset loaders
  encoder/         Signal processors & ECGFounder runtime
  llm/             Shared OpenAI-compatible LLM client
  demo/, web_demo.py, chat_ui/   Live web demo
scripts/vitalbench/   Benchmark build/generate scripts
dataset/vitalbench/   The released VitalBench benchmark
tests/                Unit test suite
```

## Paper-to-code guide

| Paper content | Main implementation |
|---|---|
| **VitalBench Benchmark: Dataset Sources and Windowing** and Table 1 | [`agent/data/`](agent/data/), [`scripts/vitalbench/download_vitalbench_data.sh`](scripts/vitalbench/download_vitalbench_data.sh), [`agent/benchmarks/vitalbench/README.md`](agent/benchmarks/vitalbench/README.md) |
| **VitalBench Benchmark: Reactive QA Data Construction** and Tables 2-4 | [`agent/benchmarks/vitalbench/`](agent/benchmarks/vitalbench/), [`agent/mhealth/state_builders/`](agent/mhealth/state_builders/), [`scripts/vitalbench/`](scripts/vitalbench/), [`dataset/vitalbench/final/`](dataset/vitalbench/final/) |
| **VitalBench Benchmark: Proactive Monitoring Data Construction** | [`agent/data/streaming.py`](agent/data/streaming.py), [`agent/proactive/`](agent/proactive/), [`agent/evaluation/proactive/`](agent/evaluation/proactive/) |
| Figure 1 and **Method: Overview / System Architecture** | [`agent/reactive/`](agent/reactive/), [`agent/proactive/`](agent/proactive/), [`agent/state/mhealth_state_store.py`](agent/state/mhealth_state_store.py), [`agent/tools/registry.py`](agent/tools/registry.py) |
| **Method: System Architecture - Physiological Memory / Unified Reasoning Layer** | [`agent/state/`](agent/state/), [`agent/reactive/planner.py`](agent/reactive/planner.py), [`agent/reactive/validation.py`](agent/reactive/validation.py), [`agent/reactive/pipeline.py`](agent/reactive/pipeline.py) |
| **Method: Tool Interface** and Table 5 | [`agent/tools/`](agent/tools/), [`agent/tools/registry.py`](agent/tools/registry.py) |
| **Experiment Setup / Results: Performance in Reactive QA** and Table 6 | [`agent/evaluation/reactive/vitalbench_eval.py`](agent/evaluation/reactive/vitalbench_eval.py) |
| **Results: Ablation / Tool-failure Robustness** and Table 7 | [`agent/evaluation/reactive/ablation_no_planner.py`](agent/evaluation/reactive/ablation_no_planner.py), [`agent/reactive/validation.py`](agent/reactive/validation.py), [`agent/evaluation/reactive/vitalbench_eval.py`](agent/evaluation/reactive/vitalbench_eval.py) |
| **Results: Proactive Performance** and Table 8 | [`agent/proactive/`](agent/proactive/), [`agent/evaluation/proactive/`](agent/evaluation/proactive/) |

## Installation

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # runtime dependencies
uv sync --extra dev     # + pytest / ruff (for running tests)
```

Copy the environment template and fill in your values:

```bash
cp .env.example .env
```

Key settings (see `.env.example` for the full list):

- `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL` — any OpenAI-compatible API
  (the template is configured for DeepSeek; per-role overrides exist for the
  agent / planner / proactive judge).
- `DATASET_{ICENTIA11K,AFPPGECG,PPG_DALIA,WESAD}_ROOT` — local paths to the raw
  datasets (not shipped here; see [Data sources](#data-sources)).
- `DATASET_VITALBENCH_ROOT` — the benchmark the evaluator reads
  (default `./dataset/vitalbench/final`, which ships with the repo).

The ECGFounder-backed ECG diagnosis tool needs the ECGFounder model under
`external/ECGFounder` (auto-cloned/downloaded on first use, or place it
manually). It is only required for ECG diagnosis questions.

## Usage

Three console entry points are installed by `uv sync`:

### Web demo

```bash
uv run vitalagent-web-demo          # serves http://127.0.0.1:7860
```

Then open `http://127.0.0.1:7860`, start a live session, and watch the proactive
engine stream windows (HR / rhythm / intervention flags) while you ask the
reactive agent questions about the current window.

The UI defaults are set for an Icentia11k demo patient (`01182`, segments
`00,01`). If you are only using the bundled tiny sample included in this repo,
enter patient `00012` and segment `00` in the UI, or point
`DATASET_ICENTIA11K_ROOT` at a local Icentia11k mirror. Missing local records may
be fetched from PhysioNet by the loader.

### Reactive evaluation (VitalBench)

```bash
uv run vitalbench-eval --conditions agent --tier A --max-samples 50 \
    --output-dir ./out
```

Reactive conditions: `agent`, `agent_no_validation`, `agent_no_replan`,
`agent_no_planner`, `agent_no_tools`, `agent_all_tools`, and the frozen
`agent_no_planner_paper_v1` reproduction condition.
The evaluator uses raw-signal mode by default. Add `--dataset <name>` to filter
to one dataset, and run `uv run vitalbench-eval --help` for all options.

Two explicit evaluation protocols keep the clean-data ablation configuration
separate from the extended robustness experiments:

```bash
# Table 7 clean-data ablation protocol. Perturbation and extended tool strategies are
# intentionally rejected in this mode.
uv run vitalbench-eval --evaluation-protocol paper_v1 \
    --conditions agent agent_no_validation agent_no_replan \
    agent_no_planner_paper_v1 --tier A --output-dir ./out/paper_v1

# Extended tool-strategy and perturbation protocol. This is the default.
uv run vitalbench-eval --evaluation-protocol supplement_v2 \
    --conditions agent_no_validation agent_no_planner agent_no_tools \
    agent_all_tools --tier A --output-dir ./out/supplement_v2
```

`paper_v1` freezes the original random-tool selection, argument-filling logic,
and validation configuration. `supplement_v2` uses the per-dataset applicable
tool pool, batched argument filling, canonical locator overrides, and the
no-tools/all-tools strategies. Every prediction, trace, and `eval.json` records
the active protocol; results from the two protocols should not be mixed in one
comparison table.

### Proactive evaluation

```bash
uv run vitalagent-proactive-eval --dataset icentia11k --output-dir ./out
```

Sweeps every patient in the chosen dataset (`icentia11k` or `afppgecg`), writing
a per-patient report JSON plus a combined `multi_patient_summary.csv` under
`--output-dir`. Pass `--patient-id <id>` to evaluate a single patient, add
`--llm-mode live` to force the LLM judge on (the default `env` mode follows
`LLM_PROACTIVE_ENABLED`), and run `uv run vitalagent-proactive-eval --help` for
all options.

## VitalBench

The released benchmark lives at `dataset/vitalbench/final/` (QA + grounding
states, with a dev/test split). To rebuild it from raw signals, or for the full
template taxonomy, schema, and reproducibility notes, see
[agent/benchmarks/vitalbench/README.md](agent/benchmarks/vitalbench/README.md).

## Data sources

Raw recordings are **not** redistributed here — download each from its host and
point the matching environment variable at your local copy.

| Name | Modality | Host | Env var |
|------|----------|------|---------|
| `icentia11k` | single-lead ECG | [PhysioNet](https://physionet.org/content/icentia11k-continuous-ecg/1.0/)  | `DATASET_ICENTIA11K_ROOT` |
| `afppgecg` | finger PPG + ECG reference | [Zenodo](https://zenodo.org/records/11242869) | `DATASET_AFPPGECG_ROOT` |
| `ppg_dalia` | wrist BVP field study | [UCI / PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) | `DATASET_PPG_DALIA_ROOT` |
| `wesad` | chest/wrist wearable | [WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) | `DATASET_WESAD_ROOT` |

## License

VitalAgent source code and documentation are licensed under the
[MIT License](LICENSE). The VitalBench artifacts under `dataset/vitalbench/`
are licensed separately under
[CC BY-NC-SA 4.0](dataset/vitalbench/LICENSE) and remain subject to the terms
and attribution requirements of their source datasets. Raw physiological
recordings are not redistributed by this repository.

## Citation

Citation details are withheld during double-blind review.

```bibtex
@misc{anonymous2026vitalagent,
      title={VitalAgent: A Tool-Augmented Agent for Reactive and Proactive Physiological Monitoring over Wearable Health Data},
      author={Anonymous Authors},
      year={2026},
      note={Submitted for anonymous review},
}
```
