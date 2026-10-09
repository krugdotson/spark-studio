# Spark Studio

A three-step AI image and video studio for the NVIDIA DGX Spark (GB10). Pick what you're making, pick a style, describe it. It runs ComfyUI underneath and everything stays on your Spark.

It makes four things:

| Mode | Model | Download |
|---|---|---|
| Image | Qwen-Image 2512 (fp8) | ~31 GB |
| Edit a photo | Qwen-Image-Edit 2511 (fp8) | ~21 GB extra (shares the text encoder with Image) |
| Text to video | WAN 2.2 14B T2V (fp8) | ~38 GB |
| Photo to video | WAN 2.2 14B I2V (fp8) | ~31 GB extra (shares the encoder with Text to video) |

All four together come to about 120 GB of disk.

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
4. Asks which model packs to download. Type `all`, or a few of `image edit t2v i2v`, or press Enter and do it later.

If your ComfyUI lives somewhere unusual: `COMFY_DIR=/path/to/ComfyUI ./install.sh`

Then open **http://<spark-ip>:7860** from any computer on your network (or http://localhost:7860 on the Spark itself).

## Everyday commands

```bash
./spark-studio status          # is it running? prints the address
./spark-studio restart
./spark-studio logs            # app log      (logs comfy = ComfyUI log)
./spark-studio models          # see which model packs are downloaded
./spark-studio models t2v i2v  # download packs (resumes if interrupted)
./spark-studio free            # unload models from memory
./spark-studio update          # update ComfyUI and the app's packages
```

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
- **Out of memory**: unified memory can get fragmented by the page cache. Run `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` (NVIDIA's recommended fix) or `./spark-studio free`.
- **Can't open the page from another computer**: if the firewall is on, `sudo ufw allow 7860/tcp`.
- **Keeps stopping when you log out**: `sudo loginctl enable-linger $USER`.

## What's where

```
install.sh          installer
spark-studio        helper commands
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
