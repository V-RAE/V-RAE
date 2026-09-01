#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
cd -- "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
python -m vrae.evaluation.libero_rfvd.run \
  --config "${project_root}/configs/evaluation/libero_rfvd/config.yaml" "$@"
