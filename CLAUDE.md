# Instructions for Claude Code

This folder is Spark Studio, a three-step web app for AI image and video generation on top of ComfyUI.
It is meant to be installed on an NVIDIA DGX Spark (GB10, ARM64 Linux, CUDA 13, 128 GB unified memory).
README.md has the full details.

When the user asks to install or set it up:

1. Check the machine: `uname -m`, `nvidia-smi`, `python3 --version`, free disk space (`df -h ~`). About 200 GB is needed for all models (about 42 GB each for the MiniMax H3 and LTX 2.3 packs).
   Also check `ls /usr/include/python3*/Python.h`: if it's missing, ask the user to run `sudo apt install -y python3-dev`
   in a normal terminal (PyTorch compiles a GPU helper at first use and fails without it).
2. Run `./install.sh`. It reuses an existing ComfyUI (or installs one), installs PyTorch with CUDA 13 if needed,
   installs the app, and creates two systemd user services: `spark-studio-comfy` and `spark-studio`.
   It prompts for model packs at the end; answer with nothing (press Enter) and download separately in step 3,
   because downloads are long.
3. Download models: `./spark-studio models all -y` (or the packs the user wants: image, edit, t2v, i2v, minimax, ltx).
   The web page can also do this: the **Models** button, backed by `GET /api/models`, `POST /api/models/download {"pack": ...}`
   and `POST /api/models/cancel` (runs as the transient user unit `spark-studio-models`; log in `data/model-download.log`).
   This takes a long time; run it in the background or let the user know.
4. Verify: `./spark-studio status`, then `curl -s localhost:7860/api/config` and confirm `online` is true and the
   wanted modes show `ready: true`.
5. Test a real render through the API, for example:
   `curl -s -X POST localhost:7860/api/generate -H 'Content-Type: application/json' -d '{"mode":"image","style":"photo","prompt":"a red apple on a wooden table","fast":true}'`
   then poll `curl -s localhost:7860/api/jobs` until that job is `done` or `error`.
   Do the same with a short video: `{"mode":"t2v","style":"none","prompt":"waves rolling onto a beach","fast":true,"seconds":3}`.
   If the `minimax` pack is downloaded, also test MiniMax H3 (video with sound):
   `{"mode":"t2v","engine":"minimax","style":"none","prompt":"Waves rolling onto a beach at sunset. Sound: surf, gulls, light wind.","fast":true,"seconds":5}`
   If the `ltx` pack is downloaded, also test LTX 2.3 (video with sound; Fast only):
   `{"mode":"t2v","engine":"ltx","style":"none","prompt":"Waves rolling onto a beach at sunset. Sound: surf, gulls, light wind.","fast":true,"seconds":5,"resolution":"720p"}`
   For both, check the saved .mp4 has an audio track: `ffprobe -v error -show_streams <file> | grep codec_type` (or use PyAV in the ComfyUI venv).
6. Tell the user the address to open: `http://<first IP from hostname -I>:7860`.

If something fails:
- ComfyUI log: `journalctl --user -u spark-studio-comfy -n 200`; app log: `journalctl --user -u spark-studio -n 200`.
- `torch.cuda.is_available()` False: check the PyTorch build matches CUDA 13 (`--index-url https://download.pytorch.org/whl/cu130`).
- The GB10 is compute capability 12.1 (sm_121); some optional speed-up packages don't support it yet. Prefer leaving them out over forcing them.
- Out-of-memory on unified memory: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`, then retry.
- Workflow graphs live in `app/workflows.py`; if a ComfyUI node rejects an input, check the node's current inputs at
  `curl -s localhost:8188/object_info/<NodeName>` and adjust the graph.
- Ask before using sudo, deleting files, or changing anything outside this folder and the ComfyUI folder.
