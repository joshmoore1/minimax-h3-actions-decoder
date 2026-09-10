# MiniMax-H3 GitHub Actions VAE Decoder

A serverless, 100% free CPU VAE decoder for [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) video generation powered by GitHub Actions.

Decodes raw 5D `.safetensors` video latents into high-definition MP4 videos with synchronized stereo audio without consuming Hugging Face ZeroGPU quota.

## Features
- **Zero GPU Quota Consumed**: Runs on GitHub Actions free `ubuntu-latest` (4 vCPU, 16 GB RAM + 10 GB swap).
- **Spatial Tiling**: Memory-safe causal 3D decoding via `diffusers.models.AutoencoderKLMiniMaxH3`.
- **Automated Artifacts**: The decoded MP4 is saved directly as an artifact downloadable from the Actions run.

## Running the Workflow

### Via GitHub Web UI
1. Go to the **Actions** tab in this repository.
2. Select **MiniMax-H3 VAE Decoder** on the left.
3. Click **Run workflow**, choose your latent file (default: `real_fox_latents.safetensors`), and click **Run workflow**.
4. Once completed, download the `.mp4` video from the **Artifacts** section at the bottom of the run page.

### Via GitHub CLI (`gh`)
```bash
gh workflow run decode_vae.yml -f latent_file=real_fox_latents.safetensors -f output_name=fox_decoded.mp4
gh run watch
```
