#!/usr/bin/env bash

set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: scripts/train/libero_fastwam.sh <nnodes> <gpus_per_node> <node_rank> [train_args...]" >&2
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
export PYTHONPATH="${project_root}/src${FASTWAM_ROOT:+:${FASTWAM_ROOT}/src}${PYTHONPATH:+:${PYTHONPATH}}"
config_path="${FASTWAM_CONFIG:-${project_root}/configs/training/generation/libero_ci2va_mot.yaml}"
paths_file="${VRAE_PATHS_FILE:-${project_root}/configs/paths.local.yaml}"
if [[ ! -f "${config_path}" ]]; then
  echo "FastWAM config does not exist: ${config_path}" >&2
  exit 2
fi

master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29501}"
if (( nnodes > 1 )) && [[ "${master_addr}" == "127.0.0.1" ]]; then
  echo "MASTER_ADDR must identify rank 0 when nnodes > 1" >&2
  exit 2
fi
torchrun_bin="${FASTWAM_TORCHRUN:-$(command -v torchrun)}"
if [[ ! -x "${torchrun_bin}" ]]; then
  echo "torchrun must be available" >&2
  exit 2
fi

launch_args=(
  "${torchrun_bin}"
  --nnodes="${nnodes}"
  --nproc_per_node="${gpus_per_node}"
  --node_rank="${node_rank}"
  --master_addr="${master_addr}"
  --master_port="${master_port}"
  -m vrae.training.fastwam.train
  --config "${config_path}"
)
if [[ -f "${paths_file}" ]]; then
  launch_args+=(--paths "${paths_file}")
fi
launch_args+=("$@")
cd -- "${project_root}"
exec "${launch_args[@]}"
