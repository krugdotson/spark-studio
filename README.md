# Spark Studio

A three-step AI image and video studio for the NVIDIA DGX Spark (GB10). Pick what you're making, pick a style, describe it. It runs ComfyUI underneath and everything stays on your Spark.

It makes four things:

| Mode | Model | Download |
|---|---|---|
| Image | Qwen-Image 2512 (fp8) | ~31 GB |
| Edit a photo | Qwen-Image-Edit 2511 (fp8) | ~21 GB extra (shares the text encoder with Image) |
| Text to video | WAN 2.2 14B T2V (fp8) | ~38 GB |
| Photo to video | WAN 2.2 14B I2V (fp8) | ~31 GB extra (shares the encoder with Text to video) |
| Text or photo to video, **with sound** | MiniMax H3 (int8 + NVFP4 text encoder) | ~42 GB (pack name: `minimax`) |
| Text or photo to video, **with sound**, up to 1080p | LTX 2.3 22B (fp8 + fp4 Gemma 3 text encoder) | ~42 GB (pack name: `ltx`) |

Both video modes have a **Model** switch: WAN 2.2 makes silent clips; MiniMax H3 generates the picture and a
matching stereo soundtrack (voices, sound effects, music) together, at 24 fps, 5 to 15 seconds. LTX 2.3 also makes
video with sound, at 25 fps, 3 to 10 seconds, up to 1080p; it renders a half-size draft, upscales it 2x and refines it,
and only has a Fast mode. Describe the sound in your prompt. Everything together comes to about 200 GB of disk;
download only the packs you want.

**Downloading packs from the browser:** the **Models** button (top right of the page) lists every pack, shows what's
installed, and has Download, Stop and Resume buttons with a progress bar. Downloads keep going if you close the page.

## Demo install (hands-off)

On a fresh DGX Spark, two commands take it from nothing to a tested, working studio:

```bash
git clone https://github.com/krugdotson/spark-studio ~/spark-studio
~/spark-studio/install.sh --all
```

Type your password once at the start (it installs a few Ubuntu packages such as `python3-dev`). After that it runs
on its own: ComfyUI, PyTorch, the web app and background services, all six model packs (about 200 GB; roughly
1½–2 hours on a fast connection, with visible progress and automatic resume), and finally a self-test that renders
one of each thing it can make and prints a timing table. It ends by printing the address to open.

## Install

Copy this folder to the Spark (for example to `~/spark-studio`), then on the Spark:

```bash
cd ~/spark-studio
./install.sh
```

The installer:

1. Finds an existing ComfyUI (for example one set up with NVIDIA's ComfyUI playbook) and reuses it, or installs a fresh copy next to this folder.
2. Sets up PyTorch with CUDA 13 for the GB10 if needed and checks that it can see the GPU.
3. Installs the Spark Studio web app and sets both to run in the background and start on boot.
4. Asks which model packs to download. Type `all`, or a few of `image edit t2v i2v minimax ltx`, or press Enter and do it later
   (you can also download them from the **Models** button in the web page).

If your ComfyUI lives somewhere unusual: `COMFY_DIR=/path/to/ComfyUI ./install.sh`

Then open **http://<spark-ip>:7860** from any computer on your network (or http://localhost:7860 on the Spark itself).

## Everyday commands

```bash
./spark-studio status          # is it running? prints the address
./spark-studio restart
./spark-studio logs            # app log      (logs comfy = ComfyUI log)
./spark-studio models          # see which model packs are downloaded
./spark-studio models t2v i2v  # download packs (resumes if interrupted)
./spark-studio selftest        # render one of each thing it can make and time it (--quick: one image)
./spark-studio free            # unload models from memory
./spark-studio update          # update ComfyUI and the app's packages
```

## Versions

The installer pins what this release was tested with, so a rebuild months later behaves the same:
ComfyUI **v0.39.2** and PyTorch **2.14.1** (torchvision 0.29.1, torchaudio 2.11.0) for CUDA 13.
To try newer ones: `COMFY_REF=latest TORCH_PIN=latest ./install.sh` (or a specific tag, e.g. `COMFY_REF=v0.40.0`),
then run `./spark-studio selftest`.

Tested on a DGX Spark (GB10), Fast mode, `./spark-studio selftest` (each test includes loading its model):

| Test | Time |
|---|---|
| Image | 28 s |
| Edit a photo | 30 s |
| Video: WAN 2.2, 3 s 480p | 61 s |
| Video: MiniMax H3, 5 s 480p, with sound | 137 s |
| Video: LTX 2.3, 5 s 720p, with sound | 87 s |
| Photo to video: WAN 2.2, 3 s | 67 s |
| Photo to video: MiniMax H3, 5 s, with sound | 139 s |
| Photo to video: LTX 2.3, 3 s 480p, with sound | 60 s |

## Settings

`config.env` holds the settings. Edit it, then `./spark-studio restart`.

- `APP_PASSWORD="something"` turns on a browser login. Recommended if your network isn't just you.
- `APP_PORT` changes the web port (default 7860).
- `COMFY_ARGS` passes extra flags to ComfyUI.

ComfyUI itself only listens on the Spark (127.0.0.1:8188). To use its node editor from another computer, use an SSH tunnel: `ssh -L 8188:localhost:8188 you@spark`, then open http://localhost:8188.

## Fast vs Best

Every mode has a speed switch.

- **Fast** uses 4-step "Lightning" LoRAs. Much quicker; video motion can be a little less dynamic.
- **Best** runs the full model (30 steps for images, 20 for video). Several times slower.

As a rough estimate (not benchmarked) for a 5-second 480p clip: a few minutes on Fast, 15 to 25 on Best. The first job after a restart is slower while models load.

## Troubleshooting

- **"Models aren't downloaded yet"**: run the `./spark-studio models <mode>` command the page shows, then refresh.
- **A job fails with `#include <Python.h>` / `gcc ... returned non-zero exit status`**: Ubuntu's Python developer files are
  missing (PyTorch compiles a small GPU helper the first time). Run `sudo apt install -y python3-dev`, then
  `systemctl --user restart spark-studio-comfy`.
- **Out of memory**: unified memory can get fragmented by the page cache. Run `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` (NVIDIA's recommended fix) or `./spark-studio free`.
- **Can't open the page from another computer**: if the firewall is on, `sudo ufw allow 7860/tcp`.
- **Keeps stopping when you log out**: `sudo loginctl enable-linger $USER`.

## What's where

```
install.sh          installer
spark-studio        helper commands
selftest.py         end-to-end test (used by `spark-studio selftest` and `install.sh --all`)
download_models.py  model downloader (used by `spark-studio models`)
app/server.py       web backend (talks to ComfyUI's API)
app/workflows.py    the ComfyUI graphs for each mode
app/presets.json    styles: edit or add your own, then restart
app/static/         the web page
data/history.json   your generation history (files live in ComfyUI/output/spark-studio)
```

### Adding a style

Add an entry to the right list in `app/presets.json`. `{prompt}` is replaced with what you type:

```json
{"id": "noir", "name": "Film noir", "blurb": "Black and white, hard shadows.",
 "template": "{prompt}. 1940s film noir, black and white, hard venetian-blind shadows, smoky atmosphere.",
 "aspect": "16:9", "hue": 0}
```

## Model licenses

Qwen-Image, Qwen-Image-Edit and WAN 2.2 are released under Apache 2.0. The Lightning LoRAs are from lightx2v. Files are downloaded from Hugging Face (Comfy-Org repackages and lightx2v).

LTX 2.3 is **not** Apache 2.0: it uses Lightricks' [LTX-2 Community License](https://huggingface.co/Lightricks/LTX-2.3), which has conditions for commercial use by larger companies. Read it before using LTX output commercially. Files come from Lightricks and Comfy-Org on Hugging Face.
