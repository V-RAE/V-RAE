#!/usr/bin/env bash

set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: scripts/train/vjepa2_1.sh <nnodes> <gpus_per_node> <node_rank> [train_args...]" >&2
  exit 2
fi

nnodes="$1"
gpus_per_node="$2"
node_rank="$3"
shift 3

if [[ ! "${nnodes}" =~ ^[1-9][0-9]*$ || ! "${gpus_per_node}" =~ ^[1-9][0-9]*$ ]]; then
  echo "nnodes and gpus_per_node must be positive integers" >&2
  exit 2
fi
if [[ ! "${node_rank}" =~ ^[0-9]+$ ]] || (( node_rank >= nnodes )); then
  echo "node_rank must be in [0, $((nnodes - 1))]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
config_path="${VJEPA_CONFIG:-${project_root}/configs/training/vjepa2_1_lerobot.yaml}"
paths_file="${VRAE_PATHS_FILE:-${project_root}/configs/paths.local.yaml}"

master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29500}"
if (( nnodes > 1 )) && [[ "${master_addr}" == "127.0.0.1" ]]; then
  echo "MASTER_ADDR must identify rank 0 when nnodes > 1" >&2
  exit 2
fi

python_bin="${VJEPA_PYTHON:-$(command -v python)}"
torchrun_bin="${VJEPA_TORCHRUN:-$(command -v torchrun)}"
if [[ ! -x "${python_bin}" || ! -x "${torchrun_bin}" ]]; then
  echo "python and torchrun must be available" >&2
  exit 2
fi

launch_args=(
  "${torchrun_bin}"
  --nnodes="${nnodes}"
  --nproc_per_node="${gpus_per_node}"
  --node_rank="${node_rank}"
  --master_addr="${master_addr}"
  --master_port="${master_port}"
  -m vrae.training.recon_training.train
  --config "${config_path}"
)
if [[ -f "${paths_file}" ]]; then
  launch_args+=(--paths "${paths_file}")
fi
launch_args+=("$@")

if [[ "${VJEPA_DRY_RUN:-0}" == "1" ]]; then
  "${python_bin}" - "${config_path}" <<'PY'
import json
import sys

from vrae.config import load_config

config = load_config(sys.argv[1])
if config["task"] != "recon_training":
    raise SystemExit("task must be recon_training")
if config["model"]["encoder"]["name"] != "vjepa2_1":
    raise SystemExit("only the V-JEPA 2.1 encoder is supported")
if config["data"].get("dataset") != "lerobot_reconstruction":
    raise SystemExit("data.dataset must be lerobot_reconstruction")
print(json.dumps({"task": config["task"], "encoder": "vjepa2_1", "dataset": "lerobot_reconstruction"}))
PY
  printf 'command:'
  printf ' %q' "${launch_args[@]}"
  printf '\n'
  exit 0
fi

cd -- "${project_root}"
exec "${launch_args[@]}"
