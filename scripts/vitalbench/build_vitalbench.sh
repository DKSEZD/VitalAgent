#!/usr/bin/env bash
#
# Build VitalBench end to end: raw signals -> states -> QA -> release subset.
#
# Stage 1 generates per-source QA over every available patient/subject and
# concatenates them into a `full/` pool. Stage 2 selects the seeded, balanced
# release subset. Both stages are deterministic; no network or LLM calls.
#
# Dataset roots are read from the environment (see .env), with ./dataset
# defaults. Override any of them inline, e.g.
#   DATASET_WESAD_ROOT=/data/WESAD ./scripts/vitalbench/build_vitalbench.sh
#
set -euo pipefail

ICENTIA_ROOT="${DATASET_ICENTIA11K_ROOT:-./dataset/icentia11k_subset/1.0}"
AFPPGECG_ROOT="${DATASET_AFPPGECG_ROOT:-./dataset/zenodo_af_monitoring_v11242869/raw}"
PPG_DALIA_ROOT="${DATASET_PPG_DALIA_ROOT:-./dataset/ppg_dalia/PPG_FieldStudy}"
WESAD_ROOT="${DATASET_WESAD_ROOT:-./dataset/WESAD}"

OUT_ROOT="${VITALBENCH_OUT_ROOT:-./dataset/vitalbench}"
FULL="$OUT_ROOT/full"
# Build output. Final subset lands at $RELEASE/final. The bundled frozen
# release used by default config lives at ./dataset/vitalbench/final, so a
# rebuild does not overwrite it unless paths are changed explicitly.
RELEASE="$OUT_ROOT/eval_subsets"

# Repo-local scratch dir (avoids /tmp portability quirks).
TMP="$OUT_ROOT/.build_tmp"

GEN="uv run python -m scripts.vitalbench.generate_vitalbench"
mkdir -p "$FULL" "$TMP"
trap 'rm -rf "$TMP"' EXIT

# --- Stage 1: per-source generation -----------------------------------------

echo ">> icentia11k"
: > "$FULL/icentia11k_provisional_qa.jsonl"
: > "$FULL/icentia11k_provisional_states.jsonl"
for hea in "$ICENTIA_ROOT"/p*/p*/p*_s*.hea; do
  base=$(basename "$hea" .hea)              # e.g. p00012_s00
  patient=${base%_s*}; patient=${patient#p}
  segment=${base#*_s}
  $GEN --source icentia11k --local-root "$ICENTIA_ROOT" \
       --patient-id "$patient" --segment-id "$segment" \
       --output "$TMP/qa.jsonl" --states-output "$TMP/st.jsonl" --preview 0
  cat "$TMP/qa.jsonl" >> "$FULL/icentia11k_provisional_qa.jsonl"
  cat "$TMP/st.jsonl" >> "$FULL/icentia11k_provisional_states.jsonl"
done

echo ">> afppgecg"
: > "$FULL/afppgecg_candidate_qa.jsonl"
: > "$FULL/afppgecg_candidate_states.jsonl"
for ppg in "$AFPPGECG_ROOT"/*_PPG.mat; do
  patient=$(basename "$ppg" _PPG.mat)       # e.g. 001
  $GEN --source afppgecg --patient-id "$patient" --dataset-root "$AFPPGECG_ROOT" \
       --output "$TMP/qa.jsonl" --states-output "$TMP/st.jsonl" --preview 0
  cat "$TMP/qa.jsonl" >> "$FULL/afppgecg_candidate_qa.jsonl"
  cat "$TMP/st.jsonl" >> "$FULL/afppgecg_candidate_states.jsonl"
done

echo ">> ppg_dalia"
: > "$FULL/ppg_dalia_all_qa.jsonl"
: > "$FULL/ppg_dalia_all_states.jsonl"
for pkl in "$PPG_DALIA_ROOT"/S*/S*.pkl; do
  $GEN --source ppg_dalia --ppg-dalia-pickle "$pkl" \
       --output "$TMP/qa.jsonl" --states-output "$TMP/st.jsonl" --preview 0
  cat "$TMP/qa.jsonl" >> "$FULL/ppg_dalia_all_qa.jsonl"
  cat "$TMP/st.jsonl" >> "$FULL/ppg_dalia_all_states.jsonl"
done

echo ">> wesad"
: > "$FULL/wesad_all_qa.jsonl"
: > "$FULL/wesad_all_states.jsonl"
for pkl in "$WESAD_ROOT"/S*/S*.pkl; do
  $GEN --source wesad --wesad-pickle "$pkl" \
       --output "$TMP/qa.jsonl" --states-output "$TMP/st.jsonl" --preview 0
  cat "$TMP/qa.jsonl" >> "$FULL/wesad_all_qa.jsonl"
  cat "$TMP/st.jsonl" >> "$FULL/wesad_all_states.jsonl"
done

# --- Stage 2: select & assemble the release subset --------------------------

echo ">> assembling release subset -> $RELEASE"
uv run python -m scripts.vitalbench.build_vitalbench \
  --full-root "$FULL" --output-root "$RELEASE" --dev-ratio 0.30

echo ">> done. release at $RELEASE/final"

