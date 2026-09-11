#!/usr/bin/env python3
"""
Standalone MiniMax-H3 CPU/GPU VAE Decoder for GitHub Actions & Local Execution (v3).

Features:
- Auto-detects CPU / CUDA.
- Precision: bfloat16 (hardware AVX-512 accelerated on Zen 4), float16, or float32.
- Granular Progress Tracking: PyTorch forward hooks on all 36 ViT decoder blocks
  reporting step count, block index, speed (s/block), and ETA.
- Configurable Spatial Tiling: supports custom tile sizes (e.g. 512x512 or 544x960)
  to avoid redundant sub-tile passes.
- Background Memory Heartbeat thread for system health monitoring.
- PyAV audio/video muxing into H.264 MP4.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
import psutil
import torch
from safetensors import safe_open
from diffusers.models import AutoencoderKLMiniMaxH3
from diffusers.video_processor import VideoProcessor
from diffusers.utils import encode_video


class MemoryHeartbeat(threading.Thread):
    def __init__(self, interval_sec: float = 30.0):
        super().__init__(daemon=True)
        self.interval = interval_sec
        self.stop_event = threading.Event()
        self.start_time = time.time()
        self.proc = psutil.Process()

    def run(self):
        while not self.stop_event.is_set():
            self.stop_event.wait(self.interval)
            if self.stop_event.is_set():
                break
            elapsed = time.time() - self.start_time
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            rss_mb = self.proc.memory_info().rss / (1024 * 1024)
            used_gb = vm.used / (1024**3)
            total_gb = vm.total / (1024**3)
            swap_mb = swap.used / (1024 * 1024)
            print(
                f"[heartbeat] Elapsed: {elapsed:5.1f}s | Process RSS: {rss_mb:6.1f} MB | "
                f"RAM: {used_gb:4.1f}/{total_gb:.1f} GB ({vm.percent}%) | Swap: {swap_mb:5.1f} MB",
                flush=True,
            )

    def stop(self):
        self.stop_event.set()


class VAEProgressTracker:
    def __init__(self, vae, num_blocks: int = 36, expected_tiles: int = 1):
        self.num_blocks = num_blocks
        self.expected_tiles = expected_tiles
        self.total_expected_steps = num_blocks * expected_tiles
        self.current_step = 0
        self.start_time = None
        self.hooks = []
        self.proc = psutil.Process()

        # Register forward hook on each transformer block of the ViT decoder
        for idx, block in enumerate(vae.decoder.transformer_blocks):
            hook = block.register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, block_idx: int):
        def hook_fn(module, input, output):
            if self.start_time is None:
                self.start_time = time.time()
            self.current_step += 1
            elapsed = time.time() - self.start_time
            rate = elapsed / self.current_step
            
            # If total_expected_steps was an underestimate due to tiling, update dynamically
            if self.current_step > self.total_expected_steps:
                self.total_expected_steps = self.current_step + self.num_blocks

            pct = (self.current_step / self.total_expected_steps) * 100.0
            eta_sec = (self.total_expected_steps - self.current_step) * rate
            eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s" if eta_sec >= 0 else "finishing"
            rss_mb = self.proc.memory_info().rss / (1024 * 1024)

            print(
                f"[PROGRESS] Step {self.current_step:3d}/{self.total_expected_steps:3d} "
                f"({pct:5.1f}%) | Layer {block_idx+1:2d}/36 | "
                f"Elapsed: {int(elapsed//60)}m {int(elapsed%60):02d}s | ETA: {eta_str} "
                f"({rate:4.2f}s/layer) | RSS: {rss_mb:.0f} MB",
                flush=True,
            )
        return hook_fn

    def remove(self):
        for h in self.hooks:
            h.remove()


def print_memory_diagnostics(stage: str):
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    total_gb = vm.total / (1024**3)
    avail_gb = vm.available / (1024**3)
    swap = psutil.swap_memory()
    swap_used_mb = swap.used / (1024 * 1024)
    print(
        f"[diag] [{stage}] Process RSS: {rss_mb:.1f} MB | System RAM: {vm.used / (1024**3):.2f}/{total_gb:.2f} GB ({vm.percent}%) | Avail: {avail_gb:.2f} GB | Swap: {swap_used_mb:.1f} MB",
        flush=True,
    )


def decode_latents(
    latent_path: str,
    vae_path: str = "MiniMaxAI/MiniMax-H3",
    output_path: str | None = None,
    device: str | None = None,
    dtype: str = "bfloat16",
    tile: bool = True,
    tile_size: tuple[int, int] | None = None,
) -> str:
    start_total = time.time()
    print_memory_diagnostics("start")

    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Latent file not found: {latent_path}")

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    print(f"[decoder] Target device: {torch_device}", flush=True)

    # Determine dtype
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    print(f"[decoder] Target precision: {torch_dtype}", flush=True)

    # Load latent file
    print(f"[decoder] Reading latents from {latent_path} ...", flush=True)
    with safe_open(latent_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        latents = f.get_tensor("latents")
        audio = f.get_tensor("audio") if "audio" in f.keys() else None

    height = int(metadata.get("height", latents.shape[-2] * 16))
    width = int(metadata.get("width", latents.shape[-1] * 16))
    fps = int(metadata.get("fps", 24))
    sampling_rate = int(metadata.get("sampling_rate", 24000))
    prompt = metadata.get("prompt", "")

    print(f"[decoder] Latents shape: {latents.shape}, dtype: {latents.dtype}", flush=True)
    print(f"[decoder] Target resolution: {width}x{height}, latent frames: {latents.shape[2]}, fps: {fps}", flush=True)
    if prompt:
        print(f"[decoder] Prompt: {prompt}", flush=True)

    # Load VAE
    print(f"[decoder] Loading AutoencoderKLMiniMaxH3 from {vae_path} ...", flush=True)
    t0 = time.time()
    load_kwargs = {}
    if os.path.isdir(vae_path):
        target_model = vae_path
    else:
        target_model = vae_path
        load_kwargs["subfolder"] = "vae"

    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        target_model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **load_kwargs,
    ).to(torch_device, dtype=torch_dtype)
    vae.eval()
    print(f"[decoder] VAE loaded in {time.time() - t0:.2f}s", flush=True)
    print_memory_diagnostics("vae_loaded")

    # Configure tiling
    expected_tiles = 1
    if not tile:
        print("[decoder] Spatial tiling DISABLED (single monolithic pass)", flush=True)
        vae.disable_tiling()
    elif tile_size is not None:
        th, tw = tile_size
        print(f"[decoder] Configuring spatial tiling with custom tile size {tw}x{th} ...", flush=True)
        vae.enable_tiling(tile_sample_min_height=th, tile_sample_min_width=tw)
        # Estimate tile count
        ny = max(1, (height + th - 1) // th)
        nx = max(1, (width + tw - 1) // tw)
        expected_tiles = ny * nx
        print(f"[decoder] Estimated spatial tiles: {expected_tiles} ({nx} horizontal x {ny} vertical)", flush=True)
    else:
        print("[decoder] Spatial tiling ENABLED (default diffusers 256x256 tile size)", flush=True)
        vae.enable_tiling()
        ny = max(1, (height + 256 - 64 - 1) // (256 - 64))
        nx = max(1, (width + 256 - 64 - 1) // (256 - 64))
        expected_tiles = ny * nx
        print(f"[decoder] Estimated spatial tiles: {expected_tiles} ({nx} horizontal x {ny} vertical)", flush=True)

    # Register real-time forward hook tracker
    tracker = VAEProgressTracker(vae, num_blocks=len(vae.decoder.transformer_blocks), expected_tiles=expected_tiles)
    heartbeat = MemoryHeartbeat(interval_sec=30.0)
    heartbeat.start()

    # Denormalize latents
    print("=" * 65, flush=True)
    print(f"[decoder] Starting decode: 36 layers x {expected_tiles} tiles = ~{tracker.total_expected_steps} total layer evaluations", flush=True)
    print("=" * 65, flush=True)
    t_decode = time.time()

    try:
        with torch.no_grad():
            latents_mean = torch.tensor(vae.config.latents_mean, device=torch_device, dtype=torch_dtype).view(1, -1, 1, 1, 1)
            latents_std = torch.tensor(vae.config.latents_std, device=torch_device, dtype=torch_dtype).view(1, -1, 1, 1, 1)
            latents_norm = latents.to(device=torch_device, dtype=torch_dtype) * latents_std + latents_mean

            autocast_enabled = (torch_device.type == "cuda" and torch_dtype in (torch.float16, torch.bfloat16))
            with torch.autocast(device_type=torch_device.type, dtype=torch_dtype, enabled=autocast_enabled):
                video = vae.decode(latents_norm, return_dict=False)[0]

            pixel_mean = torch.tensor((0.485, 0.456, 0.406), device=torch_device, dtype=torch.float32).view(1, -1, 1, 1, 1)
            pixel_std = torch.tensor((0.229, 0.224, 0.225), device=torch_device, dtype=torch.float32).view(1, -1, 1, 1, 1)
            video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
    finally:
        tracker.remove()
        heartbeat.stop()

    decode_duration = time.time() - t_decode
    print("=" * 65, flush=True)
    print(f"[decoder] Decoded video tensor: {video.shape} in {decode_duration:.2f}s ({decode_duration/60:.2f} min)", flush=True)
    print_memory_diagnostics("decoded")

    video_np = (video[0].permute(1, 2, 3, 0).float() * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    video_bytes = video_np.tobytes()

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(latent_path))[0]
        output_path = os.path.join(os.path.dirname(latent_path) or ".", f"{base_name}.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    base_name = os.path.splitext(output_path)[0]
    raw_output = f"{base_name}_raw.avi"
    av1_output = output_path if output_path.endswith(".mp4") else f"{output_path}.mp4"

    temp_wav = "/tmp/temp_audio.wav"
    has_audio = False
    if audio is not None:
        from scipy.io import wavfile
        import numpy as np
        audio_np = audio.float().cpu().numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np.T
        wavfile.write(temp_wav, sampling_rate, (audio_np * 32767).astype(np.int16))
        has_audio = True

    # Stage 2: Raw Uncompressed Video Export (timed)
    print("=" * 65, flush=True)
    print(f"[decoder] Writing Raw Uncompressed Video to {raw_output} ...", flush=True)
    t_raw_start = time.time()
    raw_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
    ]
    if has_audio:
        raw_cmd.extend(["-i", temp_wav, "-c:a", "pcm_s16le"])
    raw_cmd.extend([
        "-c:v", "rawvideo",
        "-pix_fmt", "bgr24",
        "-shortest",
        raw_output,
    ])
    p_raw = subprocess.Popen(raw_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err_raw = p_raw.communicate(input=video_bytes)
    if p_raw.returncode != 0:
        print(f"[raw-error] {err_raw.decode('utf-8', errors='ignore')}", flush=True)
    t_raw_export = time.time() - t_raw_start
    raw_size_mb = os.path.getsize(raw_output) / (1024 * 1024) if os.path.exists(raw_output) else 0
    print(f"[decoder] Raw Video export completed in {t_raw_export:.2f}s ({raw_size_mb:.2f} MB)", flush=True)

    # Stage 3: Lossless AV1 Video Encoding (timed)
    print(f"[decoder] Encoding Lossless AV1 Video to {av1_output} ...", flush=True)
    t_av1_start = time.time()
    av1_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
    ]
    if has_audio:
        av1_cmd.extend(["-i", temp_wav, "-c:a", "aac", "-b:a", "192k"])
    av1_cmd.extend([
        "-c:v", "libaom-av1",
        "-crf", "0",
        "-cpu-used", "8",
        "-pix_fmt", "yuv420p",
        "-shortest",
        av1_output,
    ])
    p_av1 = subprocess.Popen(av1_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err_av1 = p_av1.communicate(input=video_bytes)
    if p_av1.returncode != 0:
        print(f"[av1-error] {err_av1.decode('utf-8', errors='ignore')}", flush=True)
    t_av1_encode = time.time() - t_av1_start
    av1_size_mb = os.path.getsize(av1_output) / (1024 * 1024) if os.path.exists(av1_output) else 0
    print(f"[decoder] Lossless AV1 encode completed in {t_av1_encode:.2f}s ({av1_size_mb:.2f} MB)", flush=True)

    total_duration = time.time() - start_total
    print_memory_diagnostics("finished")
    print("=" * 65, flush=True)
    print("TIMING & PERFORMANCE SUMMARY")
    print(f"1. VAE Decode Time:           {decode_duration:.2f}s ({decode_duration/60:.2f} min)")
    print(f"2. Raw Video Export Time:      {t_raw_export:.2f}s ({raw_size_mb:.2f} MB)")
    print(f"3. Lossless AV1 Encode Time:   {t_av1_encode:.2f}s ({av1_size_mb:.2f} MB)")
    print(f"Total Execution Time:          {total_duration:.2f}s ({total_duration/60:.2f} min)")
    print("=" * 65, flush=True)
    return av1_output


def main():
    parser = argparse.ArgumentParser(description="Decode MiniMax-H3 latents to MP4")
    parser.add_argument("latent_path", help="Path to the .safetensors file containing latents and audio")
    parser.add_argument("--vae_path", default="MiniMaxAI/MiniMax-H3", help="Hub repo or local dir of VAE")
    parser.add_argument("--output", "-o", default=None, help="Output MP4 file path")
    parser.add_argument("--device", default=None, help="Device to use (cuda or cpu)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Precision")
    parser.add_argument("--no-tile", action="store_true", help="Disable spatial tiling")
    parser.add_argument("--tile-size", nargs=2, type=int, default=None, metavar=("HEIGHT", "WIDTH"), help="Custom tile size (e.g. 512 512 or 544 960)")

    args = parser.parse_args()
    tile_size = tuple(args.tile_size) if args.tile_size is not None else None
    decode_latents(
        latent_path=args.latent_path,
        vae_path=args.vae_path,
        output_path=args.output,
        device=args.device,
        dtype=args.dtype,
        tile=not args.no_tile,
        tile_size=tile_size,
    )


if __name__ == "__main__":
    main()
