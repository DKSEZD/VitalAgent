# VitalBench

A state-grounded question-answering benchmark over real physiological signals
(ECG / PPG). Raw waveforms are converted into structured *monitoring states*,
QA is generated deterministically from a template registry, and a balanced,
seeded subset is selected for evaluation. No LLM is used in construction, so the
ground truth is fully reproducible from a fixed seed.

## What's in the benchmark

The released artifact lives under `dataset/vitalbench/final/`:

```
dataset/vitalbench/final/
  qa.jsonl          states.jsonl      question_ids.txt
  manifest.json     split_manifest.json
  dev/   {qa,states}.jsonl + manifest
  test/  {qa,states}.jsonl + manifest
```

Scale: **1862** QA samples grounded in **1617** unique states, split into
**557** dev / **1305** test, across four sources (`afppgecg` 556,
`icentia11k` 556, `ppg_dalia` 400, `wesad` 350).

Each line of `qa.jsonl` is one question with these fields:

`question_id`, `template_id`, `tier`, `question_type`, `target`, `time_scope`,
`source_state_id`, `dataset`, `modality`, `patient_id`, `window_*_s`,
`question`, `answer`, `answer_type`, `answer_set`, `ground_truth_source`,
`evaluation_method`, `numeric_tolerance`, `evidence`, `expected_tools`,
`difficulty_tier`, `phrasing_index`, `generation_params`, `notes`.

## Using the benchmark

The evaluator reads the benchmark from `DATASET_VITALBENCH_ROOT`, which defaults
to `dataset/vitalbench/final` (see `agent.config`). No setup is needed to
inspect or load the bundled QA/states. Running the default raw-signal evaluator
also requires the matching raw dataset roots for the selected samples.

Sanity check:

```bash
uv run python -c "from agent.benchmarks.vitalbench.eval_loader import load_qa_jsonl; print(len(list(load_qa_jsonl('dataset/vitalbench/final/qa.jsonl'))))"
# -> 1862
```

## Data sources

VitalBench is built from four public datasets. The raw recordings are not
redistributed here — download each from its host and point the build at your
local copy through its dataset-root environment variable.

| Name | Modality | Signal | Host | Env var |
|------|----------|--------|------|---------|
| `icentia11k` | ECG | single-lead ambulatory ECG | [PhysioNet](https://physionet.org/content/icentia11k-continuous-ecg/1.0/) | `DATASET_ICENTIA11K_ROOT` |
| `afppgecg` | PPG (ECG reference) | finger PPG with simultaneous ECG | [Zenodo](https://zenodo.org/records/11242869) | `DATASET_AFPPGECG_ROOT` |
| `ppg_dalia` | PPG | wrist BVP, daily-activity field study | [UCI / PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) | `DATASET_PPG_DALIA_ROOT` |
| `wesad` | PPG | chest- and wrist-worn wearable | [WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) | `DATASET_WESAD_ROOT` |

`afppgecg` is the canonical identifier; the inputs `af_ppg_ecg` and the legacy
`zenodo*` names are normalized to it by
`agent.mhealth.schemas.canonicalize_dataset_name`.

## Rebuilding from raw

Rebuilding regenerates the benchmark with the *current* code. Output is written
to `dataset/vitalbench/eval_subsets/`, separate from the bundled
`dataset/vitalbench/final/`, so a rebuild never overwrites the released set.
See [Reproducibility](#reproducibility) for why a rebuild does not reproduce
the released set exactly.

**End to end.** Point the four dataset roots at your local copies (in `.env` or
inline) and run the build script (Bash; on Windows use Git Bash or WSL):

```bash
./scripts/vitalbench/build_vitalbench.sh
```

It generates QA over every available patient/subject into the `full/` pool, then
assembles the seeded subset at `dataset/vitalbench/eval_subsets/final/`.

**Single source.** To generate one source/subject manually, use one paradigm
and swap the source plus its input flags:

```bash
uv run python -m scripts.vitalbench.generate_vitalbench \
  --source <icentia11k|afppgecg|wesad|ppg_dalia> \
  <source input flags> \
  --output        dataset/vitalbench/full/<source>_qa.jsonl \
  --states-output dataset/vitalbench/full/<source>_states.jsonl
```

| Source | Required input flags |
|--------|----------------------|
| `icentia11k` | `--local-root <root> --patient-id <id> --segment-id <id>` (optionally `--duration-sec`) |
| `afppgecg` | `--dataset-root <root> --patient-id <id>` |
| `wesad` | `--wesad-pickle <SX.pkl>` |
| `ppg_dalia` | `--ppg-dalia-pickle <SX.pkl>` |

Run `... generate_vitalbench --help` for the full windowing and
threshold options.

## Templates

17 templates (`templates.py`), typed by `template_types.py`:

- **Tier** — `A` current-window / local; `B` monitoring-window aggregate.
- **Question type** — `single_verify` (yes/no), `single_choose` (closed set),
  `single_query` (numeric, tolerance-scored).
- **Ground truth** — `dataset_annotation`, `derived_from_signal`,
  `derived_from_temporal_aggregation`, `derived_from_previous_window`.
- **Scoring** — `boolean_match`, `normalized_match`, `numeric_tolerance`.

## Reproducibility

The **released VitalBench set is a frozen artifact** and should be used as-is
for any comparison against the paper. A rebuild reproduces the *methodology*,
but **not the exact released set**.

Selection is order-independent (candidates are sorted by
`sha256(seed:question_id)`, `seed=42`), so generation order never affects the
result. A rebuild differs from the released set because of **code drift in the
selection key**: `question_id` is the hash key, and state-builder / template
changes since the release (e.g. PPG-DaLiA now answers from wrist-BVP heart rate,
one template removed) change `question_id`s and therefore which items are
selected.

The shipped `manifest.json` records the original seed, caps, and selection
method, so the released selection remains auditable.
