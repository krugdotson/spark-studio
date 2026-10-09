#!/usr/bin/env bash
# Spark Studio installer for NVIDIA DGX Spark (GB10, ARM64 Linux).
#
#   ./install.sh                 install everything, then offer to download models
#   ./install.sh --all           hands-off: system packages, app, all model packs, then a self-test
#   COMFY_DIR=/path/to/ComfyUI ./install.sh   use a specific existing ComfyUI
#
# Options: --all (same as --yes --models all), --yes (no questions; installs missing system packages with sudo),
#          --models "image t2v ..." (packs to download, or all), --no-selftest, --no-services,
#          --skip-checks (no GPU checks; for testing)
# Versions: ComfyUI and PyTorch default to the versions this release was tested with.
#           COMFY_REF=latest and/or TORCH_PIN=latest use the newest instead.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
APP_PORT="${APP_PORT:-7860}"
COMFY_PORT="${COMFY_PORT:-8188}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
# Tested together on a DGX Spark (GB10). See README "Versions".
COMFY_REF="${COMFY_REF:-v0.39.2}"
TORCH_PIN="${TORCH_PIN:-torch==2.14.1 torchvision==0.29.1 torchaudio==2.11.0}"
NO_SERVICES=0; SKIP_CHECKS=0; YES=0; MODELS=""; SELFTEST=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) YES=1; MODELS="all" ;;
    --yes|-y) YES=1 ;;
    --models) shift; MODELS="${1:-}" ;;
    --models=*) MODELS="${1#*=}" ;;
    --no-selftest) SELFTEST=0 ;;
    --no-services) NO_SERVICES=1 ;;
    --skip-checks) SKIP_CHECKS=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done
START=$SECONDS

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
die()  { printf '\n\033[31mError:\033[0m %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ checks
bold "Checking this machine"
if [[ $SKIP_CHECKS == 0 ]]; then
  [[ "$(uname -m)" == "aarch64" ]] || info "Note: this isn't an ARM64 machine ($(uname -m)). It's built for the DGX Spark but should work on any Linux box with an NVIDIA GPU."
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found. Is the NVIDIA driver installed?"
  info "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi

# System packages. python3-dev and gcc are needed because PyTorch's Triton compiles a small GPU helper
# the first time it runs; without them the first render fails.
has_python_h() { python3 -c 'import os, sys, sysconfig; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)' 2>/dev/null; }
missing=()
command -v git >/dev/null || missing+=(git)
command -v curl >/dev/null || missing+=(curl)
command -v gcc >/dev/null || missing+=(build-essential)
command -v python3 >/dev/null || missing+=(python3)
python3 -c 'import ensurepip, venv' 2>/dev/null || missing+=(python3-venv)
has_python_h || missing+=(python3-dev)
if [[ ${#missing[@]} -gt 0 ]]; then
  info "These system packages are needed: ${missing[*]}"
  command -v apt-get >/dev/null || die "Install them with your package manager, then re-run ./install.sh"
  if [[ $YES == 0 ]]; then
    [[ -t 0 ]] || die "Run: sudo apt install -y ${missing[*]}   (or re-run with --yes to have the installer do it)"
    read -r -p "  Install them now with sudo? [Y/n] " ans
    [[ -z "$ans" || "$ans" =~ ^[Yy] ]] || die "Run: sudo apt install -y ${missing[*]}, then re-run ./install.sh"
  fi
  info "Installing them (sudo will ask for your password once)"
  sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}" >/dev/null \
    || die "Couldn't install ${missing[*]}. Run: sudo apt install -y ${missing[*]}"
fi
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "Python 3.10 or newer is needed (found $(python3 --version))."
has_python_h || die "Python developer files are still missing. Run: sudo apt install -y python3-dev"
info "Python: $(python3 --version | cut -d' ' -f2)"

cuda_ok() {  # $1 = python; true if PyTorch there can use the GPU
  [[ $SKIP_CHECKS == 1 ]] && { "$1" -c 'import torch' 2>/dev/null; return; }
  "$1" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null
}

comfy_checkout() {  # put $COMFY_DIR on COMFY_REF (a tag/branch/commit, or "latest" = newest release tag)
  git -C "$COMFY_DIR" fetch -q --tags origin 2>/dev/null || true
  local ref="$COMFY_REF"
  [[ "$ref" == latest ]] && ref="$(git -C "$COMFY_DIR" tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1)"
  git -C "$COMFY_DIR" -c advice.detachedHead=false checkout -q "$ref" || die "Couldn't switch ComfyUI to $ref"
  info "ComfyUI $ref"
}

# ------------------------------------------------------------------ ComfyUI
bold "Finding ComfyUI"
if [[ -z "${COMFY_DIR:-}" ]]; then
  for c in "$HERE/ComfyUI" "$HOME/ComfyUI" "$HOME/comfyui/ComfyUI" "$HOME/comfy-ui/ComfyUI"; do
    [[ -f "$c/main.py" ]] && { COMFY_DIR="$c"; break; }
  done
  if [[ -z "${COMFY_DIR:-}" ]]; then
    found="$(find "$HOME" -maxdepth 4 -type f -path '*ComfyUI/main.py' 2>/dev/null | head -1 || true)"
    [[ -n "$found" ]] && COMFY_DIR="$(dirname "$found")"
  fi
fi

COMFY_PY=""
if [[ -n "${COMFY_DIR:-}" && -f "$COMFY_DIR/main.py" ]]; then
  COMFY_DIR="$(cd "$COMFY_DIR" && pwd)"
  info "Using your existing ComfyUI at $COMFY_DIR"
  for v in "$HERE/comfy-venv" "$COMFY_DIR/venv" "$COMFY_DIR/.venv" "$COMFY_DIR/../comfyui-env" "$COMFY_DIR/../venv" "$COMFY_DIR/../.venv"; do
    if [[ -x "$v/bin/python" ]] && cuda_ok "$v/bin/python"; then COMFY_PY="$(cd "$v" && pwd)/bin/python"; break; fi
  done
  [[ -n "$COMFY_PY" ]] && info "Using its Python environment: $(dirname "$(dirname "$COMFY_PY")")"
  if ! git -C "$COMFY_DIR" diff --quiet 2>/dev/null; then
    info "It has local changes, so I left it as is."
  elif [[ "$COMFY_DIR" == "$HERE/ComfyUI" ]]; then
    comfy_checkout  # our own copy: keep it on the tested version
  else
    info "Updating it so the newest model nodes are available"
    git -C "$COMFY_DIR" pull --ff-only -q || info "Couldn't update automatically (that's OK if it's recent)."
  fi
else
  COMFY_DIR="$HERE/ComfyUI"
  info "No ComfyUI found. Installing a fresh copy to $COMFY_DIR"
  git clone -q https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
  comfy_checkout
fi

if [[ -z "$COMFY_PY" ]]; then
  bold "Setting up PyTorch for the GPU (CUDA 13) — this takes a few minutes"
  python3 -m venv "$HERE/comfy-venv"
  COMFY_PY="$HERE/comfy-venv/bin/python"
  "$COMFY_PY" -m pip install -q --upgrade pip wheel
  if [[ $SKIP_CHECKS == 0 ]]; then
    pkgs="$TORCH_PIN"; [[ "$pkgs" == latest ]] && pkgs="torch torchvision torchaudio"
    # shellcheck disable=SC2086
    "$COMFY_PY" -m pip install -q $pkgs --index-url "$TORCH_INDEX"
  fi
fi
bold "Installing ComfyUI's requirements"
"$COMFY_PY" -m pip install -q -r "$COMFY_DIR/requirements.txt"
if [[ $SKIP_CHECKS == 0 ]]; then
  "$COMFY_PY" - <<'PY' || die "PyTorch can't see the GPU. Check nvidia-smi, then re-run ./install.sh"
import torch
assert torch.cuda.is_available()
p = torch.cuda.get_device_properties(0)
print(f"  PyTorch {torch.__version__} sees {p.name} (compute {p.major}.{p.minor}, CUDA {torch.version.cuda})")
PY
fi

# ------------------------------------------------------------------ app
bold "Installing Spark Studio"
[[ -x "$HERE/app-venv/bin/python" ]] || python3 -m venv "$HERE/app-venv"
"$HERE/app-venv/bin/python" -m pip install -q --upgrade pip
"$HERE/app-venv/bin/python" -m pip install -q -r "$HERE/app/requirements.txt"
chmod +x "$HERE/download_models.py" "$HERE/spark-studio" 2>/dev/null || true

if [[ -f "$HERE/config.env" ]] && grep -q '^APP_PASSWORD=' "$HERE/config.env"; then
  APP_PASSWORD="$(grep '^APP_PASSWORD=' "$HERE/config.env" | cut -d= -f2- | tr -d '"')"
fi
cat > "$HERE/config.env" <<EOF
# Spark Studio settings. Edit, then run: ./spark-studio restart
COMFY_DIR="$COMFY_DIR"
COMFY_PY="$COMFY_PY"
COMFY_PORT=$COMFY_PORT
COMFY_URL="http://127.0.0.1:$COMFY_PORT"
COMFY_ARGS=""
APP_PORT=$APP_PORT
APP_HOST="0.0.0.0"
# Set a password to require a login in the browser (recommended on shared networks)
APP_PASSWORD="${APP_PASSWORD:-}"
EOF
info "Settings saved to config.env"

# ------------------------------------------------------------------ services
if [[ $NO_SERVICES == 0 ]] && command -v systemctl >/dev/null && systemctl --user show-environment >/dev/null 2>&1; then
  bold "Setting Spark Studio to run in the background"
  UNIT_DIR="$HOME/.config/systemd/user"; mkdir -p "$UNIT_DIR"
  external_comfy=0
  if curl -fs "http://127.0.0.1:$COMFY_PORT/system_stats" >/dev/null 2>&1 && \
     ! systemctl --user is-active --quiet spark-studio-comfy 2>/dev/null; then
    external_comfy=1
    info "ComfyUI is already running on port $COMFY_PORT, so Spark Studio will use that one."
    info "(Stop it and re-run ./install.sh if you'd rather Spark Studio start ComfyUI itself.)"
  fi
  if [[ $external_comfy == 0 ]]; then
    cat > "$UNIT_DIR/spark-studio-comfy.service" <<EOF
[Unit]
Description=ComfyUI engine for Spark Studio
After=network-online.target

[Service]
EnvironmentFile=$HERE/config.env
WorkingDirectory=$COMFY_DIR
ExecStart=/bin/bash -c 'exec "\$\${COMFY_PY}" main.py --listen 127.0.0.1 --port \$\${COMFY_PORT} \$\${COMFY_ARGS}'
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
  fi
  cat > "$UNIT_DIR/spark-studio.service" <<EOF
[Unit]
Description=Spark Studio web app
After=network-online.target spark-studio-comfy.service

[Service]
EnvironmentFile=$HERE/config.env
Environment=APP_DATA=$HERE/data
WorkingDirectory=$HERE/app
ExecStart=/bin/bash -c 'exec "$HERE/app-venv/bin/uvicorn" server:app --host \$\${APP_HOST} --port \$\${APP_PORT}'
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  [[ $external_comfy == 0 ]] && systemctl --user enable --now spark-studio-comfy.service >/dev/null 2>&1 \
    && systemctl --user restart spark-studio-comfy.service
  systemctl --user enable --now spark-studio.service >/dev/null 2>&1 && systemctl --user restart spark-studio.service
  if ! loginctl show-user "$USER" -p Linger 2>/dev/null | grep -q yes; then
    if loginctl enable-linger "$USER" 2>/dev/null || sudo -n loginctl enable-linger "$USER" 2>/dev/null; then
      info "Spark Studio will keep running after you log out and start on boot."
    else
      info "To keep it running after you log out, run once: sudo loginctl enable-linger $USER"
    fi
  fi
  info "Waiting for ComfyUI to start…"
  for _ in $(seq 1 90); do curl -fs "http://127.0.0.1:$COMFY_PORT/system_stats" >/dev/null 2>&1 && break; sleep 2; done
  curl -fs "http://127.0.0.1:$COMFY_PORT/system_stats" >/dev/null 2>&1 && info "ComfyUI is up." \
    || info "ComfyUI hasn't answered yet. Check with: ./spark-studio logs comfy"
else
  info "Skipping background services. Start manually with: ./spark-studio run"
fi

if command -v ufw >/dev/null && sudo -n ufw status 2>/dev/null | grep -q "Status: active"; then
  info "Your firewall is on. To reach Spark Studio from other computers: sudo ufw allow $APP_PORT/tcp"
fi

# ------------------------------------------------------------------ models
bold "Models"
"$HERE/app-venv/bin/python" "$HERE/download_models.py" --list
packs="$MODELS"
if [[ -z "$packs" && $YES == 0 && -t 0 ]]; then
  echo
  echo "  Which packs should I download now? (image, edit, t2v, i2v, minimax, ltx, all, or press Enter to skip)"
  read -r -p "  > " packs || packs=""
fi
if [[ -n "$packs" ]]; then
  # Downloads resume, so retry a couple of times in case the network drops mid-way.
  ok=0
  for attempt in 1 2 3; do
    # shellcheck disable=SC2086
    if "$HERE/app-venv/bin/python" "$HERE/download_models.py" $packs -y; then ok=1; break; fi
    info "Download interrupted (attempt $attempt of 3). Resuming in 10 seconds…"; sleep 10
  done
  [[ $ok == 1 ]] || die "Model download didn't finish. Re-run: ./spark-studio models $packs"
fi

# ------------------------------------------------------------------ self-test
TEST_RESULT=""
if [[ -n "$packs" && $SELFTEST == 1 && $NO_SERVICES == 0 ]]; then
  bold "Self-test: rendering one of each thing Spark Studio can make"
  for _ in $(seq 1 30); do curl -fs "http://127.0.0.1:$APP_PORT/api/health" >/dev/null 2>&1 && break; sleep 2; done
  if "$HERE/spark-studio" selftest; then TEST_RESULT="passed"; else TEST_RESULT="failed"; fi
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
mins=$(( (SECONDS - START) / 60 ))
if [[ "$TEST_RESULT" == failed ]]; then
  bold "Installed, but the self-test found a problem (${mins} min)"
  info "See the table above, then check: ./spark-studio logs comfy"
else
  bold "All set${TEST_RESULT:+ — self-test passed} (${mins} min)"
fi
info "Open Spark Studio in a browser:"
info "  On the Spark:            http://localhost:$APP_PORT"
[[ -n "$IP" ]] && info "  From another computer:  http://$IP:$APP_PORT"
info "Download more models any time:  ./spark-studio models all   (or the Models button in the page)"
info "Start, stop, logs, test:        ./spark-studio status | restart | logs | selftest"
echo
[[ "$TEST_RESULT" != failed ]]
