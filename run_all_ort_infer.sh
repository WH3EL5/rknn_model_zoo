#!/usr/bin/env bash

set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs/ort_batch"
mkdir -p "$LOG_DIR"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

if [[ -t 1 && "${NO_COLOR:-0}" != "1" ]]; then
  C_RESET='\033[0m'
  C_INFO='\033[1;34m'
  C_RUN='\033[1;36m'
  C_CMD='\033[0;35m'
  C_OK='\033[1;32m'
  C_WARN='\033[1;33m'
  C_ERR='\033[1;31m'
  C_LOG='\033[0;37m'
else
  C_RESET=''
  C_INFO=''
  C_RUN=''
  C_CMD=''
  C_OK=''
  C_WARN=''
  C_ERR=''
  C_LOG=''
fi

info() { echo -e "${C_INFO}[INFO]${C_RESET} $*"; }
run_msg() { echo -e "${C_RUN}[RUN]${C_RESET} $*"; }
cmd_msg() { echo -e "${C_CMD}[CMD]${C_RESET} $*"; }
ok_msg() { echo -e "${C_OK}[OK]${C_RESET}  $*"; }
warn_msg() { echo -e "${C_WARN}[WARN]${C_RESET} $*"; }
err_msg() { echo -e "${C_ERR}[ERR]${C_RESET} $*"; }
log_msg() { echo -e "${C_LOG}[LOG]${C_RESET} $*"; }

if command -v conda >/dev/null 2>&1 && [[ "${USE_CONDA:-1}" == "1" ]]; then
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-ort}"
  PYTHON_CMD=(conda run -n "$CONDA_ENV_NAME" python)
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
  PYTHON_CMD=("$PYTHON_BIN")
fi

has_flag() {
  local file="$1"
  local flag="$2"
  grep -Eq -- "\"--${flag}\"|'--${flag}'" "$file"
}

list_candidate_onnx() {
  local example_dir="$1"
  local model_dir="${example_dir}/model"

  {
    if [[ -d "$model_dir" ]]; then
      find "$model_dir" -type f -name '*.onnx' 2>/dev/null
    fi
    find "$example_dir" -maxdepth 1 -type f -name '*.onnx' 2>/dev/null
  } | awk 'NF' | sort -u
}

find_onnx_by_keyword_from_list() {
  local list_file="$1"
  local keyword="$2"
  grep -Ei "/[^/]*${keyword}[^/]*\.onnx$" "$list_file" | head -n 1
}

declare -a ORT_SCRIPTS
while IFS= read -r line; do
  ORT_SCRIPTS+=("$line")
done < <(find "$ROOT_DIR/examples" -type f -path '*/python/ort*.py' | sort)

if [[ ${#ORT_SCRIPTS[@]} -eq 0 ]]; then
  info "No ort*.py scripts found under examples."
  exit 0
fi

info "Root: $ROOT_DIR"
info "Python command: ${PYTHON_CMD[*]}"
info "Total scripts: ${#ORT_SCRIPTS[@]}"

declare -a FAILED
TOTAL_RUNS=0
SUCCESS=0

run_one() {
  local rel_script="$1"
  local script_dir="$2"
  local run_tag="$3"
  shift 3
  local -a cmd=("$@")
  local safe_tag="${run_tag//\//_}"
  safe_tag="${safe_tag// /_}"
  local log_file="$LOG_DIR/${rel_script//\//__}__${safe_tag}.log"

  TOTAL_RUNS=$((TOTAL_RUNS + 1))

  echo
  run_msg "$rel_script [$run_tag]"
  cmd_msg "(cd $script_dir && ${cmd[*]})"

  if [[ $DRY_RUN -eq 1 ]]; then
    return 0
  fi

  (
    cd "$script_dir" || exit 1
    "${cmd[@]}" 2>&1 | tee "$log_file"
  )
  local rc=$?

  if [[ $rc -eq 0 ]]; then
    SUCCESS=$((SUCCESS + 1))
    ok_msg "$rel_script [$run_tag]"
    log_msg "$log_file"
  else
    FAILED+=("$rel_script [$run_tag]")
    err_msg "$rel_script [$run_tag] (exit=$rc)"
    log_msg "$log_file"
  fi
}

for script in "${ORT_SCRIPTS[@]}"; do
  script_dir="$(cd "$(dirname "$script")" && pwd)"
  example_dir="$(cd "$script_dir/.." && pwd)"
  script_name="$(basename "$script")"
  rel_script="${script#"$ROOT_DIR/"}"

  onnx_list_file="$(mktemp)"
  list_candidate_onnx "$example_dir" > "$onnx_list_file"

  ONNX_FILES=()
  while IFS= read -r onnx; do
    ONNX_FILES+=("$onnx")
  done < "$onnx_list_file"

  base_cmd=("${PYTHON_CMD[@]}" "$script_name")

  if has_flag "$script" "encoder" && has_flag "$script" "decoder"; then
    encoder_model="$(find_onnx_by_keyword_from_list "$onnx_list_file" 'encoder')"
    decoder_model="$(find_onnx_by_keyword_from_list "$onnx_list_file" 'decoder')"

    if [[ -z "$encoder_model" || -z "$decoder_model" ]]; then
      encoder_model="${encoder_model:-${ONNX_FILES[0]:-}}"
      decoder_model="${decoder_model:-${ONNX_FILES[1]:-}}"
    fi

    if [[ -z "$encoder_model" || -z "$decoder_model" ]]; then
      warn_msg "$rel_script: cannot determine both --encoder and --decoder model files"
      FAILED+=("$rel_script [missing-encoder-decoder]")
      rm -f "$onnx_list_file"
      continue
    fi

    cmd=("${base_cmd[@]}" --encoder "$encoder_model" --decoder "$decoder_model")
    run_one "$rel_script" "$script_dir" "encoder_decoder" "${cmd[@]}"

  elif has_flag "$script" "models"; then
    if [[ ${#ONNX_FILES[@]} -gt 0 ]]; then
      cmd=("${base_cmd[@]}" --models)
      for m in "${ONNX_FILES[@]}"; do
        cmd+=("$m")
      done
      run_one "$rel_script" "$script_dir" "all_models_in_one_run" "${cmd[@]}"
    else
      warn_msg "$rel_script: no onnx file found, run with script defaults"
      run_one "$rel_script" "$script_dir" "default_args" "${base_cmd[@]}"
    fi

  elif has_flag "$script" "model" && grep -q 'required=True' "$script"; then
    if [[ ${#ONNX_FILES[@]} -eq 0 ]]; then
      warn_msg "$rel_script: --model is required but no onnx file found"
      FAILED+=("$rel_script [missing-model]")
      rm -f "$onnx_list_file"
      continue
    fi

    for model_path in "${ONNX_FILES[@]}"; do
      model_name="$(basename "$model_path")"
      cmd=("${base_cmd[@]}" --model "$model_path")
      run_one "$rel_script" "$script_dir" "$model_name" "${cmd[@]}"
    done

  else
    run_one "$rel_script" "$script_dir" "default_args" "${base_cmd[@]}"
  fi

  rm -f "$onnx_list_file"
done

failed_count=${#FAILED[@]}

echo
echo "========== SUMMARY =========="
echo "Scripts: ${#ORT_SCRIPTS[@]}"
echo "Runs:    $TOTAL_RUNS"
echo "Success: $SUCCESS"
echo "Failed:  $failed_count"

if [[ $failed_count -gt 0 ]]; then
  err_msg "Failed runs:"
  for item in "${FAILED[@]}"; do
    echo "  - $item"
  done
  exit 1
fi

ok_msg "All ort runs finished successfully."
