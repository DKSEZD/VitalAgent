#!/usr/bin/env bash
#
# Download the raw data needed to run or rebuild VitalBench.
#
#   icentia11k  25 benchmark patients from PhysioNet   (open access)
#   afppgecg    Zenodo record 11242869                 (open access; see --cohort)
#   ppg_dalia   UCI ML Repository                      (manual - blocks scripted downloads)
#   wesad       UCI ML Repository                      (manual - blocks scripted downloads)
#
# Usage:
#   ./scripts/vitalbench/download_vitalbench_data.sh [--cohort qa|proactive|full]
#
#   --cohort qa          25 Icentia11k patients + 11 AFPPGECG QA subjects
#   --cohort proactive   25 Icentia11k patients + 44 AFPPGECG subjects
#   --cohort full        full Icentia11k + full AFPPGECG; prompts for confirmation
#
# Destination directories match the defaults in build_vitalbench.sh and accept
# the same environment overrides:
#   DATASET_ICENTIA11K_ROOT  DATASET_AFPPGECG_ROOT
#   DATASET_PPG_DALIA_ROOT   DATASET_WESAD_ROOT
#
# Requirements: wget (icentia11k), curl + python3 (afppgecg). macOS: brew install wget

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/vitalbench/download_vitalbench_data.sh [--cohort qa|proactive|full]

Options:
  --cohort qa          25 Icentia11k patients plus AFPPGECG subjects 001-011.
  --cohort proactive   25 Icentia11k patients plus AFPPGECG subjects 001-045 excluding 028.
  --cohort full        Complete Icentia11k tree plus every file in the AFPPGECG Zenodo record.
  -h, --help           Show this help text.
EOF
}


download_icentia_patient() {
  patient="$1"
  prefix="${patient:0:2}"
  patient_dir="$ICENTIA_ROOT/p${prefix}/p${patient}"
  mkdir -p "$patient_dir"

  segments=0
  for segment_index in $(seq 0 99); do
    segment=$(printf "%02d" "$segment_index")
    hea_url="$PHYSIONET_BASE/p${prefix}/p${patient}/p${patient}_s${segment}.hea"
    # wget exit 8 = server error (404 etc.) - no more segments for this patient
    if ! wget -q -N -c -P "$patient_dir" "$hea_url"; then
      break
    fi
    wget -q -N -c -P "$patient_dir" \
      "$PHYSIONET_BASE/p${prefix}/p${patient}/p${patient}_s${segment}.atr" || true
    if ! wget -q -N -c -P "$patient_dir" \
      "$PHYSIONET_BASE/p${prefix}/p${patient}/p${patient}_s${segment}.dat"; then
      echo "    ERROR: failed to download required signal file for patient $patient segment $segment" >&2
      exit 1
    fi
    segments=$((segments + 1))
  done

  echo "    patient $patient: $segments segment(s) downloaded"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
COHORT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cohort)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Missing value for --cohort." >&2
        usage >&2
        exit 2
      fi
      COHORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$COHORT" && "$COHORT" != "qa" && "$COHORT" != "proactive" && "$COHORT" != "full" ]]; then
  echo "Unknown cohort '$COHORT'. Use: qa, proactive, or full." >&2
  usage >&2
  exit 2
fi

ICENTIA_ROOT="${DATASET_ICENTIA11K_ROOT:-./dataset/icentia11k_subset/1.0}"
AFPPGECG_ROOT="${DATASET_AFPPGECG_ROOT:-./dataset/zenodo_af_monitoring_v11242869/raw}"
PPG_DALIA_ROOT="${DATASET_PPG_DALIA_ROOT:-./dataset/ppg_dalia/PPG_FieldStudy}"
WESAD_ROOT="${DATASET_WESAD_ROOT:-./dataset/WESAD}"

# 25 Icentia11k patients used in VitalBench
ICENTIA_PATIENTS=(
  00012 00069 00500 00992 01182 01500 01612 02500
  03123 03500 04309 04500 05413 05500 05526 05769
  05790 06011 06500 07500 08279 08500 08623 08947
  10500
)

PHYSIONET_BASE="https://physionet.org/files/icentia11k-continuous-ecg/1.0"
ZENODO_ID="11242869"

if [[ -z "$COHORT" ]]; then
  echo "Select a cohort:"
  echo "  qa          25 Icentia11k patients + 11 AFPPGECG subjects"
  echo "  proactive   25 Icentia11k patients + 44 AFPPGECG subjects"
  echo "  full        full Icentia11k + full AFPPGECG"
  read -rp "cohort [qa/proactive/full]: " COHORT
fi

ICENTIA_FULL=0
case "$COHORT" in
  qa)
    AFPPGECG_PATIENTS=(001 002 003 004 005 006 007 008 009 010 011)
    ;;
  proactive)
    AFPPGECG_PATIENTS=()
    for i in $(seq 1 45); do
      pid=$(printf "%03d" "$i")
      [[ "$pid" == "028" ]] && continue
      AFPPGECG_PATIENTS+=("$pid")
    done
    ;;
  full)
    echo "WARNING: full download fetches the complete Icentia11k PhysioNet tree"
    echo "and every file in the AFPPGECG Zenodo record (~77.6 GB for AFPPGECG alone)."
    read -rp "Confirm full download? [y/N]: " _confirm
    [[ "$_confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
    ICENTIA_FULL=1
    AFPPGECG_PATIENTS=()  # empty = no AFPPGECG filtering, download everything
    ;;
  *)
    echo "Unknown cohort '$COHORT'. Use: qa, proactive, or full." >&2
    usage >&2
    exit 2
    ;;
esac

# ---------------------------------------------------------------------------
# icentia11k
# ---------------------------------------------------------------------------
if [[ "$ICENTIA_FULL" -eq 1 ]]; then
  echo "==> icentia11k: full PhysioNet tree -> $ICENTIA_ROOT"
else
  echo "==> icentia11k: ${#ICENTIA_PATIENTS[@]} patients -> $ICENTIA_ROOT"
fi

if ! command -v wget &>/dev/null; then
  echo "    SKIP: wget not found (brew install wget on macOS)."
  echo "    The Icentia11kLoader falls back to PhysioNet on demand if no local"
  echo "    copy exists - slower but functional without downloading."
else
  echo "    Source: https://physionet.org/content/icentia11k-continuous-ecg/1.0/"
  echo "    This dataset is open access; comply with its license and citation terms."
  echo

  if [[ "$ICENTIA_FULL" -eq 1 ]]; then
    # -nH --cut-dirs=3 strips 'files/icentia11k-continuous-ecg/1.0/' so files
    # land at $ICENTIA_ROOT/p{prefix}/... for a full-tree download.
    echo "    downloading full Icentia11k tree ... "
    wget -q -r -N -c -np -nH --cut-dirs=3 \
         -P "$ICENTIA_ROOT" \
         "$PHYSIONET_BASE/"
    echo "done"
  else
    icentia_total="${#ICENTIA_PATIENTS[@]}"
    for index in "${!ICENTIA_PATIENTS[@]}"; do
      patient="${ICENTIA_PATIENTS[$index]}"
      printf "  [%02d/%02d]\n" "$((index + 1))" "$icentia_total"
      download_icentia_patient "$patient"
    done
  fi
  echo "    icentia11k done."
fi
echo

# ---------------------------------------------------------------------------
# afppgecg (Zenodo)
# ---------------------------------------------------------------------------
echo "==> afppgecg: Zenodo record $ZENODO_ID -> $AFPPGECG_ROOT"

mkdir -p "$AFPPGECG_ROOT"

FILE_ENTRIES=$(ZENODO_ID="$ZENODO_ID" python3 - <<'PYEOF'
import urllib.request, json, os, sys
record_id = os.environ["ZENODO_ID"]
url = f"https://zenodo.org/api/records/{record_id}"
try:
    with urllib.request.urlopen(url) as r:
        data = json.load(r)
    for f in data.get("files", []):
        filename = f.get("key", "")
        links = f.get("links", {})
        file_url = links.get("content") or links.get("self", "")
        if filename and file_url:
            print(f"{filename}\t{file_url}")
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
)

SELECTED_FILE_ENTRIES=()
while IFS=$'\t' read -r filename file_url; do
  [[ -z "${filename:-}" || -z "${file_url:-}" ]] && continue

  # Filter by cohort patient list when not full
  if [[ ${#AFPPGECG_PATIENTS[@]} -gt 0 ]]; then
    match=0
    for pid in "${AFPPGECG_PATIENTS[@]}"; do
      case "$filename" in
        "${pid}_ECG.mat"|"${pid}_PPG.mat")
          match=1
          break
          ;;
      esac
    done
    [[ "$match" -eq 0 ]] && continue
  fi

  SELECTED_FILE_ENTRIES+=("${filename}"$'\t'"${file_url}")
done <<< "$FILE_ENTRIES"

if [[ ${#SELECTED_FILE_ENTRIES[@]} -eq 0 ]]; then
  echo "    ERROR: no AFPPGECG files matched cohort '$COHORT'." >&2
  exit 1
fi

afppgecg_total="${#SELECTED_FILE_ENTRIES[@]}"
echo "    downloading $afppgecg_total files"

afppgecg_index=0
for entry in "${SELECTED_FILE_ENTRIES[@]}"; do
  IFS=$'\t' read -r filename file_url <<< "$entry"
  afppgecg_index=$((afppgecg_index + 1))

  dest="$AFPPGECG_ROOT/$filename"
  printf "    [%03d/%03d] %s\n" "$afppgecg_index" "$afppgecg_total" "$filename"
  curl --fail -L --retry 5 --retry-delay 5 -C - --progress-bar -o "$dest" "$file_url"

  if [[ "$filename" == *.zip ]]; then
    echo "    extracting $filename"
    unzip -q -o "$dest" -d "$AFPPGECG_ROOT"
  fi
done
echo "    afppgecg done."
echo

# ---------------------------------------------------------------------------
# ppg_dalia - manual
# ---------------------------------------------------------------------------
echo "==> ppg_dalia: manual download required"
echo "    URL : https://archive.ics.uci.edu/dataset/495/ppg+dalia"
echo "    Dest: $PPG_DALIA_ROOT"
echo "    After extracting, \$PPG_DALIA_ROOT/S1/S1.pkl should exist."
echo "    Override default dest with: export DATASET_PPG_DALIA_ROOT=/your/path"
echo

# ---------------------------------------------------------------------------
# wesad - manual
# ---------------------------------------------------------------------------
echo "==> wesad: manual download required"
echo "    URL : https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection"
echo "    Dest: $WESAD_ROOT"
echo "    After extracting, \$WESAD_ROOT/S2/S2.pkl should exist."
echo "    Override default dest with: export DATASET_WESAD_ROOT=/your/path"
echo

echo "==> All automatic downloads complete."
echo "    Set the four DATASET_* vars in .env and run scripts/vitalbench/build_vitalbench.sh."
