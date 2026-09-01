#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
eval_dir="${project_root}/ckpts/eval_models"
perceptual_dir="${project_root}/ckpts/pretrained/perceptual"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to download evaluation weights" >&2
  exit 1
fi
if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
  echo "sha256sum or shasum is required to verify evaluation weights" >&2
  exit 1
fi

sha256_file() {
  local path="$1"
  local output
  if command -v sha256sum >/dev/null 2>&1; then
    output="$(sha256sum "${path}")"
  else
    output="$(shasum -a 256 "${path}")"
  fi
  printf '%s\n' "${output%% *}"
}

verify_file() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256_file "${path}")"
  if [[ "${actual:0:${#expected}}" != "${expected}" ]]; then
    echo "checksum mismatch for ${path}" >&2
    echo "  expected prefix: ${expected}" >&2
    echo "  actual:          ${actual}" >&2
    return 1
  fi
}

download_file() {
  local name="$1"
  local url="$2"
  local destination="$3"
  local expected_sha256="$4"
  local partial="${destination}.part"

  if [[ -f "${destination}" ]] && verify_file "${destination}" "${expected_sha256}"; then
    echo "Using existing ${name}: ${destination}"
    return
  fi

  echo "Downloading ${name}..."
  curl --fail --location \
    --retry 5 --retry-delay 2 \
    --continue-at - \
    --output "${partial}" \
    "${url}"

  if ! verify_file "${partial}" "${expected_sha256}"; then
    rm -f -- "${partial}"
    echo "Removed the invalid partial download; rerun the script to retry." >&2
    return 1
  fi
  mv -- "${partial}" "${destination}"
  echo "Saved ${name}: ${destination}"
}

mkdir -p "${eval_dir}" "${perceptual_dir}"

download_file \
  "FVD I3D TorchScript" \
  "https://huggingface.co/flateon/FVD-I3D-torchscript/resolve/1c2d61711c52571ad617a1063e6fa691b212e184/i3d_torchscript.pt" \
  "${eval_dir}/i3d_torchscript.pt" \
  "bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2"

download_file \
  "FID Inception v3" \
  "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth" \
  "${eval_dir}/inception_fid.pth" \
  "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"

download_file \
  "ImageNet VGG16" \
  "https://download.pytorch.org/models/vgg16-397923af.pth" \
  "${perceptual_dir}/vgg16.pt" \
  "397923af"

download_file \
  "LPIPS v0.1 VGG calibration" \
  "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-vgg16-lpips.pth" \
  "${perceptual_dir}/lpips_vgg.pt" \
  "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868"

echo "Evaluation weights are ready."
