#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script builds the Metal backend on macOS. See README.md for other systems." >&2
  exit 1
fi

for command in git cmake curl python3; do
  if ! command -v "$command" >/dev/null; then
    echo "Missing $command. Install it before running this script." >&2
    exit 1
  fi
done

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ui.txt

if [[ ! -d stable-diffusion.cpp ]]; then
  git clone https://github.com/leejet/stable-diffusion.cpp.git
  git -C stable-diffusion.cpp checkout c92d73c408515c94beef32161bb5960764fde7a0
fi
git -C stable-diffusion.cpp submodule update --init --recursive
cmake -S stable-diffusion.cpp -B stable-diffusion.cpp/build -DSD_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build stable-diffusion.cpp/build --config Release --target sd-cli -j 4

mkdir -p models
download() {
  local filename="$1" expected_bytes="$2" url="$3" current_bytes=0
  if [[ -f "models/$filename" ]]; then
    current_bytes="$(stat -f %z "models/$filename")"
  fi
  if (( current_bytes == expected_bytes )); then
    echo "Already present: models/$filename"
    return
  fi
  echo "Downloading $filename"
  curl --fail --location --retry 3 --continue-at - --output "models/$filename" "$url"
  current_bytes="$(stat -f %z "models/$filename")"
  if (( current_bytes != expected_bytes )); then
    echo "Incomplete download: $filename ($current_bytes of $expected_bytes bytes). Rerun this script to resume." >&2
    exit 1
  fi
}

download qwen_image_2.1-Q4_K.gguf 4197494816 \
  https://huggingface.co/leejet/Qwen-Image-2.1-GGUF/resolve/main/qwen_image_2.1-Q4_K.gguf
download qwen3vl_8b_heretic-Q4_K_M.gguf 5027785376 \
  https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF/resolve/main/qwen3vl_8b_heretic-Q4_K_M.gguf
download mmproj-qwen3vl_8b_heretic-f16.gguf 1159030464 \
  https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF/resolve/main/mmproj-qwen3vl_8b_heretic-f16.gguf
download qwen_image_2.1_vae_bf16.safetensors 675509688 \
  https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors

echo "Setup complete. Run: .venv/bin/python app.py"
