#!/usr/bin/env python3
"""
Standalone MiniMax-H3 CPU/GPU VAE Decoder for GitHub Actions & Local Execution.

Decodes raw 5D latents and audio from a .safetensors file into a high-definition MP4.
Features:
- Auto-detects CPU / CUDA.
- Diagnostics: logs RAM usage (resident memory + available) at each stage.
- Supports spatial tiling via --tile to fit within free runner RAM ceilings.
- Combines decoded video frames with stereo audio into a final H.264 MP4.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import psutil
import torch
from safetensors import safe_open
from diffusers.models import AutoencoderKLMiniMaxH3
from diffusers.video_processor import VideoProcessor
from diffusers.utils import encode_video


def print_memory_diagnostics(stage: str):
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    total_gb = vm.total / (1024**3)
    avail_gb = vm.available / (1024**3)
    used_pct = vm.percent
    swap = psutil.swap_memory()
    swap_used_mb = swap.used / (1024 * 1024)
    print(f"[diag] [{stage}] Process RSS: {rss_mb:.1f} MB | System RAM: {vm.used / (1024**3):.2f}/{total_gb:.2f} GB ({used_pct}%) | Avail: {avail_gb:.2f} GB | Swap Used: {swap_used_mb:.1f} MB")


def decode_latents(
    latent_path: str,
    vae_path: str = "MiniMaxAI/MiniMax-H3",
    output_path: str | None = None,
    device: str | None = None,
    dtype: str = "float32",
    tile: bool = True,
) -> str:
    start_total = time.time()
    print_memory_diagnostics("start")

    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Latent file not found: {latent_path}")

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    print(f"[decoder] Using device: {torch_device}")

    # Determine dtype
    if torch_device.type == "cpu":
        torch_dtype = torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    print(f"[decoder] Using dtype: {torch_dtype}")

    # Load latent file
    print(f"[decoder] Reading latents from {latent_path} ...")
    with safe_open(latent_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        latents = f.get_tensor("latents")
        audio = f.get_tensor("audio") if "audio" in f.keys() else None

    height = int(metadata.get("height", latents.shape[-2] * 16))
    width = int(metadata.get("width", latents.shape[-1] * 16))
    fps = int(metadata.get("fps", 24))
    sampling_rate = int(metadata.get("sampling_rate", 24000))
    prompt = metadata.get("prompt", "")

    print(f"[decoder] Latents shape: {latents.shape}, dtype: {latents.dtype}")
    print(f"[decoder] Target resolution: {width}x{height}, latent frames: {latents.shape[2]}, fps: {fps}")
    if prompt:
        print(f"[decoder] Prompt: {prompt}")

    # Load VAE
    print(f"[decoder] Loading AutoencoderKLMiniMaxH3 from {vae_path} ...")
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
    print(f"[decoder] VAE loaded in {time.time() - t0:.2f}s")
    print_memory_diagnostics("vae_loaded")

    if tile:
        print("[decoder] Spatial tiling ENABLED (low memory consumption)")
        vae.enable_tiling()

    # Denormalize latents
    print("[decoder] Denormalizing latents and decoding video ...")
    t_decode = time.time()
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

    decode_duration = time.time() - t_decode
    print(f"[decoder] Decoded video tensor shape: {video.shape} in {decode_duration:.2f}s")
    print_memory_diagnostics("decoded")

    print("[decoder] Postprocessing video into PIL frames ...")
    vp = VideoProcessor(vae_scale_factor=16, do_normalize=False)
    frames = vp.postprocess_video(video, output_type="pil")[0]
    num_frames = len(frames)
    print(f"[decoder] Total frames generated: {num_frames} ({num_frames / fps:.2f}s)")

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(latent_path))[0]
        output_path = os.path.join(os.path.dirname(latent_path) or ".", f"{base_name}.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"[decoder] Muxing video and audio into {output_path} ...")
    if audio is not None:
        audio = audio.float()
        print(f"[decoder] Audio shape: {audio.shape}, sample rate: {sampling_rate} Hz")

    encode_video(
        frames,
        fps=fps,
        output_path=output_path,
        audio=audio,
        audio_sample_rate=sampling_rate if audio is not None else None,
    )

    total_duration = time.time() - start_total
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print_memory_diagnostics("finished")
    print("=" * 60)
    print(f"[decoder] SUCCESS! Video saved to: {output_path} ({file_size_mb:.2f} MB)")
    print(f"[decoder] Decode Time: {decode_duration:.2f}s | Total Elapsed: {total_duration:.2f}s")
    print("=" * 60)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Decode MiniMax-H3 latents to MP4")
    parser.add_argument("latent_path", help="Path to the .safetensors file containing latents and audio")
    parser.add_argument("--vae_path", default="MiniMaxAI/MiniMax-H3", help="Hub repo or local dir of VAE")
    parser.add_argument("--output", "-o", default=None, help="Output MP4 file path")
    parser.add_argument("--device", default=None, help="Device to use (cuda or cpu)")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"], help="Precision")
    parser.add_argument("--no-tile", action="store_true", help="Disable spatial tiling")

    args = parser.parse_args()
    decode_latents(
        latent_path=args.latent_path,
        vae_path=args.vae_path,
        output_path=args.output,
        device=args.device,
        dtype=args.dtype,
        tile=not args.no_tile,
    )


if __name__ == "__main__":
    main()
