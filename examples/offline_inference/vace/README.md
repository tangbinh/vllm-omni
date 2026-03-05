# VACE Video Generation Example

VACE (Video Creation and Editing) is an all-in-one model for video creation and editing supporting multiple tasks:

- **T2V**: Text-to-Video generation (prompt only)
- **R2V**: Reference-to-Video generation (prompt + reference image)
- **V2V**: Video-to-Video editing (prompt + source video)
- **MV2V**: Masked Video-to-Video editing (prompt + source video + mask)

## Supported Models

This example uses official HuggingFace VACE models:
- `Wan-AI/Wan2.1-VACE-14B-diffusers` (14B parameters)
- `Wan-AI/Wan2.1-VACE-1.3B-diffusers` (1.3B parameters)

## Quick Start

### Text-to-Video (T2V)

```bash
python vace_video_generation.py \
    --model Wan-AI/Wan2.1-VACE-14B-diffusers \
    --prompt "A serene natural landscape with gentle motion"
```

### Reference-to-Video (R2V)

Use a reference image to guide video generation:

```bash
python vace_video_generation.py \
    --model Wan-AI/Wan2.1-VACE-14B-diffusers \
    --reference-image reference.png \
    --prompt "Camera slowly pans across the scene"
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `Wan-AI/Wan2.1-VACE-14B-diffusers` | HuggingFace model ID or local path |
| `--reference-image` | None | Reference image for R2V mode |
| `--prompt` | `"A beautiful natural scene..."` | Text prompt |
| `--negative-prompt` | `""` | Negative prompt for CFG |
| `--height` | 480 | Video height |
| `--width` | 832 | Video width |
| `--num-frames` | 81 | Number of output frames |
| `--num-inference-steps` | 50 | Number of denoising steps |
| `--guidance-scale` | 5.0 | CFG guidance scale |
| `--seed` | 42 | Random seed |
| `--output` | `vace_output.mp4` | Output file path |
| `--fps` | 24 | Output video FPS |
| `--ulysses-degree` | 1 | Ulysses sequence parallelism degree for multi-GPU |

## Memory Optimization

For large models (14B), use these flags to reduce memory usage:

```bash
python vace_video_generation.py \
    --model Wan-AI/Wan2.1-VACE-14B-diffusers \
    --prompt "Your prompt" \
    --vae-use-tiling \
    --enable-cpu-offload \
    --height 480 \
    --width 832
```

## Multi-GPU Support

Use Ulysses sequence parallelism for faster inference:

```bash
python vace_video_generation.py \
    --model Wan-AI/Wan2.1-VACE-14B-diffusers \
    --prompt "Your prompt" \
    --ulysses-degree 8
```
