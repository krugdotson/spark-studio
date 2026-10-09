#!/usr/bin/env bash
# Spark Studio installer for NVIDIA DGX Spark (GB10, ARM64 Linux).
#
#   ./install.sh                 install everything, then offer to download models
#   COMFY_DIR=/path/to/ComfyUI ./install.sh   use a specific existing ComfyUI
#
# Options: --no-services (don't create background services), --skip-checks (no GPU checks; for testing)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
APP_PORT="${APP_PORT:-7860}"
COMFY_PORT="${COMFY_PORT:-8188}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
NO_SERVICES=0; SKIP_CHECKS=0
for a in "$@"; do
  case "$a" in
    --no-services) NO_SERVICES=1 ;;
    --skip-checks) SKIP_CHECKS=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "Unknown option: $a"; exit 1 ;;
  esac
done

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
command -v git >/dev/null || die "git is missing. Run: sudo apt install -y git"
command -v python3 >/dev/null || die "python3 is missing. Run: sudo apt install -y python3 python3-venv python3-pip"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "Python 3.10 or newer is needed (found $(python3 --version))."
python3 -c 'import ensurepip, venv' 2>/dev/null || die "Python venv support is missing. Run: sudo apt install -y python3-venv"
info "Python: $(python3 --version | cut -d' ' -f2)"

cuda_ok() {  # $1 = python; true if PyTorch there can use the GPU
  [[ $SKIP_CHECKS == 1 ]] && { "$1" -c 'import torch' 2>/dev/null; return; }
  "$1" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null
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
  for v in "$COMFY_DIR/venv" "$COMFY_DIR/.venv" "$COMFY_DIR/../comfyui-env" "$COMFY_DIR/../venv" "$COMFY_DIR/../.venv"; do
    if [[ -x "$v/bin/python" ]] && cuda_ok "$v/bin/python"; then COMFY_PY="$(cd "$v" && pwd)/bin/python"; break; fi
  done
  [[ -n "$COMFY_PY" ]] && info "Using its Python environment: $(dirname "$(dirname "$COMFY_PY")")"
  info "Updating it so the newest model nodes are available"
  if git -C "$COMFY_DIR" diff --quiet 2>/dev/null; then
    git -C "$COMFY_DIR" pull --ff-only -q || info "Couldn't update automatically (that's OK if it's recent)."
  else
    info "It has local changes, so I left it as is."
  fi
else
  COMFY_DIR="$HERE/ComfyUI"
  info "No ComfyUI found. Installing a fresh copy to $COMFY_DIR"
  git clone -q https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
  tag="$(git -C "$COMFY_DIR" describe --tags --abbrev=0 2>/dev/null || true)"
  [[ -n "$tag" ]] && git -C "$COMFY_DIR" -c advice.detachedHead=false checkout -q "$tag" && info "ComfyUI $tag"
fi

if [[ -z "$COMFY_PY" ]]; then
  bold "Setting up PyTorch for the GPU (CUDA 13) — this takes a few minutes"
  python3 -m venv "$HERE/comfy-venv"
  COMFY_PY="$HERE/comfy-venv/bin/python"
  "$COMFY_PY" -m pip install -q --upgrade pip wheel
  if [[ $SKIP_CHECKS == 0 ]]; then
    "$COMFY_PY" -m pip install -q torch torchvision torchaudio --index-url "$TORCH_INDEX"
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
if [[ -t 0 ]]; then
  echo
  echo "  Which packs should I download now? (image, edit, t2v, i2v, all, or press Enter to skip)"
  read -r -p "  > " packs || packs=""
  if [[ -n "$packs" ]]; then
    "$HERE/app-venv/bin/python" "$HERE/download_models.py" $packs -y
  fi
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
bold "All set"
info "Open Spark Studio in a browser:"
info "  On the Spark:            http://localhost:$APP_PORT"
[[ -n "$IP" ]] && info "  From another computer:  http://$IP:$APP_PORT"
info "Download more models any time:  ./spark-studio models all"
info "Start, stop, logs:              ./spark-studio status | restart | logs"
echo
