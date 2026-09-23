#!/usr/bin/env bash
set -euo pipefail
: "${PLAYGROUND_PASSWORD:?Set PLAYGROUND_PASSWORD in the RunPod environment}"
# This image deliberately cannot select sd.cpp or a quantized profile.
export PLAYGROUND_BACKEND=diffusers
mkdir -p "${PLAYGROUND_MODEL_DIR:-/workspace/models}" "${PLAYGROUND_OUTPUT_DIR:-/workspace/outputs}"
if [[ "${PLAYGROUND_REQUIRE_CUDA:-1}" == 1 ]]; then
  python -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU attached"; assert torch.cuda.is_bf16_supported(including_emulation=False), "GPU must support native BF16"; print(torch.cuda.get_device_name(0))'
fi
if [[ "${PLAYGROUND_DOWNLOAD_MODELS:-1}" == 1 ]]; then
  args=()
  if [[ "${PLAYGROUND_REWRITER:-1}" == 1 ]]; then args+=(--rewriter); fi
  python /app/download_models.py "${args[@]}"
fi
exec python /app/app.py
