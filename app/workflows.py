"""ComfyUI API-format workflow builders.

Graphs follow the official ComfyUI templates for Qwen-Image 2512, Qwen-Image-Edit 2511
and WAN 2.2 14B, with an optional 4-step "Lightning" LoRA path for fast mode.
Each builder returns (graph, sampler_node_ids) so progress can be tracked across samplers.
"""

QWEN_NEG = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲"
WAN_NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
           "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
           "杂乱的背景，三条腿，背景人很多，倒着走，裸露，NSFW")

# Model files each mode needs, keyed by the ComfyUI loader folder.
REQUIRED = {
    "image": {
        "diffusion_models": ["qwen_image_2512_fp8_e4m3fn.safetensors"],
        "text_encoders": ["qwen_2.5_vl_7b_fp8_scaled.safetensors"],
        "vae": ["qwen_image_vae.safetensors"],
    },
    "edit": {
        "diffusion_models": ["qwen_image_edit_2511_fp8mixed.safetensors"],
        "text_encoders": ["qwen_2.5_vl_7b_fp8_scaled.safetensors"],
        "vae": ["qwen_image_vae.safetensors"],
    },
    "t2v": {
        "diffusion_models": ["wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
                             "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"],
        "text_encoders": ["umt5_xxl_fp8_e4m3fn_scaled.safetensors"],
        "vae": ["wan_2.1_vae.safetensors"],
    },
    "i2v": {
        "diffusion_models": ["wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                             "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"],
        "text_encoders": ["umt5_xxl_fp8_e4m3fn_scaled.safetensors"],
        "vae": ["wan_2.1_vae.safetensors"],
    },
}
FAST_LORAS = {
    "image": ["Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"],
    "edit": ["Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"],
    "t2v": ["wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
            "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"],
    "i2v": ["wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
            "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"],
}

IMAGE_SIZES = {  # Qwen-Image recommended resolutions
    "1:1": (1328, 1328), "16:9": (1664, 928), "9:16": (928, 1664),
    "4:3": (1472, 1104), "3:4": (1104, 1472), "3:2": (1584, 1056), "2:3": (1056, 1584),
}
VIDEO_SIZES = {
    "480p": {"16:9": (832, 480), "9:16": (480, 832), "1:1": (640, 640), "4:3": (736, 560), "3:4": (560, 736)},
    "720p": {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (960, 960), "4:3": (1104, 832), "3:4": (832, 1104)},
}
VIDEO_FPS = 16

# MiniMax H3: one model for text-to-video and photo-to-video, with native stereo audio.
# Wiring follows ComfyUI's official "MiniMax H3" templates (video_minimax_h3_t2v / _i2v).
MINIMAX = {
    "diffusion_models": ["minimax_h3_fl2va_pruned_int8_convrot.safetensors"],
    "text_encoders": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
    "vae": ["minimax_h3_video_vae_int8_convrot.safetensors", "minimax_h3_audio_vae_fp32.safetensors"],
}
MINIMAX_TURBO = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
MINIMAX_FPS = 24
# Sizes are multiples of 32. "480p" ~0.4 MP (template default); "768p" is H3's native canvas (768 short edge).
MINIMAX_SIZES = {
    "480p": {"16:9": (864, 480), "9:16": (480, 864), "1:1": (640, 640), "4:3": (736, 544), "3:4": (544, 736)},
    "768p": {"16:9": (1344, 768), "9:16": (768, 1344), "1:1": (1024, 1024), "4:3": (1152, 864), "3:4": (864, 1152)},
}
ENGINES = ("wan", "minimax")


def sizes_for(engine):
    return MINIMAX_SIZES if engine == "minimax" else VIDEO_SIZES


def minimax_frames(seconds):
    # 24 fps, snapped up to H3's 17k+5 frame grid (same formula as the official template)
    f = max(5, round(seconds * MINIMAX_FPS))
    return f + (5 - f % 17) % 17


def nearest_aspect(w, h, choices):
    r = w / h
    def ratio(k):
        a, b = k.split(":")
        return int(a) / int(b)
    return min(choices, key=lambda k: abs(ratio(k) - r))


def frames_for(seconds):
    # WAN wants 4n+1 frames
    n = max(1, round(seconds * VIDEO_FPS / 4))
    return 4 * n + 1


def build_image(p):
    w, h = IMAGE_SIZES.get(p.get("aspect", "1:1"), IMAGE_SIZES["1:1"])
    fast = p.get("fast", True)
    model = ["unet", 0]
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": REQUIRED["image"]["diffusion_models"][0], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": REQUIRED["image"]["text_encoders"][0], "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": REQUIRED["image"]["vae"][0]}},
    }
    if fast:
        g["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": FAST_LORAS["image"][0], "strength_model": 1.0}}
        model = ["lora", 0]
    g.update({
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": 3.1}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": p["prompt"], "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": p.get("negative") or QWEN_NEG, "clip": ["clip", 0]}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": int(p.get("count", 1))}},
        "sampler": {"class_type": "KSampler", "inputs": {
            "model": ["shift", 0], "seed": p["seed"], "steps": 4 if fast else 30, "cfg": 1.0 if fast else 4.0,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["pos", 0], "negative": ["neg", 0],
            "latent_image": ["latent", 0], "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "spark-studio/image"}},
    })
    return g, ["sampler"]


def build_edit(p):
    fast = p.get("fast", True)
    model = ["unet", 0]
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": REQUIRED["edit"]["diffusion_models"][0], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": REQUIRED["edit"]["text_encoders"][0], "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": REQUIRED["edit"]["vae"][0]}},
        "load": {"class_type": "LoadImage", "inputs": {"image": p["image"]}},
        "scale": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["load", 0]}},
    }
    if fast:
        g["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": FAST_LORAS["edit"][0], "strength_model": 1.0}}
        model = ["lora", 0]
    enc = lambda text: {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
        "clip": ["clip", 0], "prompt": text, "vae": ["vae", 0], "image1": ["scale", 0]}}
    g.update({
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": 3.1}},
        "norm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
        "pos_raw": enc(p["prompt"]),
        "neg_raw": enc(""),
        "pos": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["pos_raw", 0], "reference_latents_method": "index_timestep_zero"}},
        "neg": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["neg_raw", 0], "reference_latents_method": "index_timestep_zero"}},
        "encode": {"class_type": "VAEEncode", "inputs": {"pixels": ["scale", 0], "vae": ["vae", 0]}},
        "sampler": {"class_type": "KSampler", "inputs": {
            "model": ["norm", 0], "seed": p["seed"], "steps": 4 if fast else 30, "cfg": 1.0 if fast else 4.0,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["pos", 0], "negative": ["neg", 0],
            "latent_image": ["encode", 0], "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "spark-studio/edit"}},
    })
    return g, ["sampler"]


def _video(p, mode):
    fast = p.get("fast", True)
    res = p.get("resolution", "480p")
    w, h = p.get("size") or VIDEO_SIZES.get(res, VIDEO_SIZES["480p"]).get(p.get("aspect", "16:9"), (832, 480))
    length = frames_for(float(p.get("seconds", 5)))
    req = REQUIRED[mode]
    g = {
        "high": {"class_type": "UNETLoader", "inputs": {"unet_name": req["diffusion_models"][0], "weight_dtype": "default"}},
        "low": {"class_type": "UNETLoader", "inputs": {"unet_name": req["diffusion_models"][1], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": req["text_encoders"][0], "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": req["vae"][0]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": p["prompt"], "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": p.get("negative") or WAN_NEG, "clip": ["clip", 0]}},
    }
    high, low = ["high", 0], ["low", 0]
    if fast:
        g["lora_high"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": high, "lora_name": FAST_LORAS[mode][0], "strength_model": 1.0}}
        g["lora_low"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": low, "lora_name": FAST_LORAS[mode][1], "strength_model": 1.0}}
        high, low = ["lora_high", 0], ["lora_low", 0]
    g["shift_high"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": high, "shift": 5.0 if fast else 8.0}}
    g["shift_low"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": low, "shift": 5.0 if fast else 8.0}}

    if mode == "i2v":
        g["load"] = {"class_type": "LoadImage", "inputs": {"image": p["image"]}}
        g["i2v"] = {"class_type": "WanImageToVideo", "inputs": {
            "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
            "width": w, "height": h, "length": length, "batch_size": 1, "start_image": ["load", 0]}}
        pos, neg, latent = ["i2v", 0], ["i2v", 1], ["i2v", 2]
    else:
        g["latent"] = {"class_type": "EmptyHunyuanLatentVideo", "inputs": {"width": w, "height": h, "length": length, "batch_size": 1}}
        pos, neg, latent = ["pos", 0], ["neg", 0], ["latent", 0]

    steps, split, cfg = (4, 2, 1.0) if fast else (20, 10, 3.5)
    common = {"sampler_name": "euler", "scheduler": "simple", "steps": steps, "cfg": cfg, "positive": pos, "negative": neg}
    g["sample_high"] = {"class_type": "KSamplerAdvanced", "inputs": {
        **common, "model": ["shift_high", 0], "add_noise": "enable", "noise_seed": p["seed"],
        "latent_image": latent, "start_at_step": 0, "end_at_step": split, "return_with_leftover_noise": "enable"}}
    g["sample_low"] = {"class_type": "KSamplerAdvanced", "inputs": {
        **common, "model": ["shift_low", 0], "add_noise": "disable", "noise_seed": p["seed"],
        "latent_image": ["sample_high", 0], "start_at_step": split, "end_at_step": 10000, "return_with_leftover_noise": "disable"}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample_low", 0], "vae": ["vae", 0]}}
    g["video"] = {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "fps": VIDEO_FPS}}
    g["save"] = {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": f"spark-studio/{mode}", "format": "auto"}}
    return g, ["sample_high", "sample_low"]


def _minimax(p, mode):
    fast = p.get("fast", True)
    res = p.get("resolution", "480p")
    w, h = p.get("size") or MINIMAX_SIZES.get(res, MINIMAX_SIZES["480p"]).get(p.get("aspect", "16:9"), (864, 480))
    model = ["unet", 0]
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": MINIMAX["diffusion_models"][0], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": MINIMAX["text_encoders"][0], "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": MINIMAX["vae"][0]}},
        "audio_vae": {"class_type": "VAELoader", "inputs": {"vae_name": MINIMAX["vae"][1]}},
    }
    if fast:
        g["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": MINIMAX_TURBO, "strength_model": 1.0}}
        model = ["lora", 0]
    cond = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": p["prompt"],
            "width": w, "height": h, "length": minimax_frames(float(p.get("seconds", 5)))}
    if mode == "i2v":
        g["load"] = {"class_type": "LoadImage", "inputs": {"image": p["image"]}}
        g["fit"] = {"class_type": "ImageScale", "inputs": {"image": ["load", 0], "upscale_method": "lanczos",
                                                           "width": w, "height": h, "crop": "center"}}
        cond["first_frame"] = ["fit", 0]
    g.update({
        "cond": {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": p["seed"]}},
        "sampler_select": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "sigmas": {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple",
                                                              "steps": 8 if fast else 20, "denoise": 1.0}},
        "guider": {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["cond", 0]}},
        "sampler": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler_select", 0],
            "sigmas": ["sigmas", 0], "latent_image": ["cond", 1]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "decode_audio": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sampler", 0], "vae": ["audio_vae", 0]}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "audio": ["decode_audio", 0], "fps": MINIMAX_FPS}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": f"spark-studio/minimax-{mode}", "format": "auto"}},
    })
    return g, ["sampler"]


BUILDERS = {
    "image": build_image,
    "edit": build_edit,
    "t2v": lambda p: _minimax(p, "t2v") if p.get("engine") == "minimax" else _video(p, "t2v"),
    "i2v": lambda p: _minimax(p, "i2v") if p.get("engine") == "minimax" else _video(p, "i2v"),
}


def build(mode, params):
    return BUILDERS[mode](params)


def required_files(mode, fast, engine="wan"):
    if engine == "minimax" and mode in ("t2v", "i2v"):
        req = {k: list(v) for k, v in MINIMAX.items()}
        if fast:
            req["loras"] = [MINIMAX_TURBO]
        return req
    req = {k: list(v) for k, v in REQUIRED[mode].items()}
    if fast:
        req.setdefault("loras", []).extend(FAST_LORAS[mode])
    return req
