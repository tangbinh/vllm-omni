# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
VACE (Video Creation and Editing) video generation example.

VACE is an all-in-one model supporting multiple video generation tasks:
- T2V: Text-to-Video generation (prompt only)
- R2V: Reference-to-Video generation (prompt + reference image)
- V2V: Video-to-Video editing (prompt + source video) [not yet implemented]
- MV2V: Masked Video-to-Video editing (prompt + source video + mask) [not yet implemented]

This example uses official HuggingFace VACE models:
- Wan-AI/Wan2.1-VACE-14B-diffusers
- Wan-AI/Wan2.1-VACE-1.3B-diffusers

Usage:
    # T2V generation (text-to-video)
    python vace_video_generation.py \\
        --model Wan-AI/Wan2.1-VACE-14B-diffusers \\
        --prompt "A beautiful sunset over mountains" \\
        --ulysses-degree 8

    # R2V generation (reference image to video)
    python vace_video_generation.py \\
        --model Wan-AI/Wan2.1-VACE-14B-diffusers \\
        --reference-image reference.jpg \\
        --prompt "Camera panning across the scene" \\
        --ulysses-degree 8
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate video using Wan2.1 VACE model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.1-VACE-14B-diffusers",
        help="HuggingFace VACE model ID or local path. "
        "Options: Wan-AI/Wan2.1-VACE-14B-diffusers, Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    )

    # Reference image for R2V mode
    parser.add_argument(
        "--reference-image",
        type=str,
        default=None,
        help="Path to reference image for R2V (Reference-to-Video) mode. "
        "The reference image will be used to condition video generation.",
    )

    # Generation parameters
    parser.add_argument(
        "--prompt",
        default="A beautiful natural scene with smooth motion.",
        help="Text prompt describing the video.",
    )
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG scale.")
    parser.add_argument("--height", type=int, default=480, help="Video height.")
    parser.add_argument("--width", type=int, default=832, help="Video width.")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Sampling steps.")
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=None,
        help="Scheduler flow_shift. Defaults to model config.",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="vace_output.mp4",
        help="Path to save the video.",
    )
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS.")

    # Memory optimization
    parser.add_argument(
        "--vae-use-tiling",
        action="store_true",
        help="Enable VAE tiling for memory optimization.",
    )
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable CPU offloading.",
    )

    # Parallelism
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism size.",
    )
    parser.add_argument(
        "--ulysses-degree",
        type=int,
        default=1,
        help="Ulysses sequence parallelism degree. Faster than TP for video generation.",
    )

    return parser.parse_args()


def save_video(frames: np.ndarray, output_path: str, fps: int = 24) -> None:
    """Save video frames to file.

    Args:
        frames: Video frames array [T, H, W, C]
        output_path: Output file path
        fps: Frames per second
    """
    try:
        import imageio.v3 as iio

        iio.imwrite(output_path, frames, fps=fps, codec="libx264")
        print(f"Video saved to: {output_path}")
    except ImportError:
        # Fallback to saving individual frames
        output_dir = Path(output_path).with_suffix("")
        output_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(frames):
            frame_path = output_dir / f"frame_{i:04d}.png"
            PIL.Image.fromarray(frame).save(frame_path)
        print(f"Frames saved to: {output_dir}/")


def main():
    args = parse_args()

    # Validate reference image if provided
    if args.reference_image and not os.path.exists(args.reference_image):
        raise FileNotFoundError(f"Reference image not found: {args.reference_image}")

    device = current_omni_platform.device_type
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Configure parallelism
    parallel_config = DiffusionParallelConfig(
        tensor_parallel_size=args.tensor_parallel_size,
        ulysses_degree=args.ulysses_degree,
    )

    # Determine mode
    mode = "R2V" if args.reference_image else "T2V"

    print(f"\n{'=' * 60}")
    print(f"VACE Video Generation ({mode} mode)")
    print(f"  Model: {args.model}")
    if args.reference_image:
        print(f"  Reference image: {args.reference_image}")
    print(f"  Video size: {args.width}x{args.height}")
    print(f"  Frames: {args.num_frames}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Parallelism: TP={args.tensor_parallel_size}, Ulysses={args.ulysses_degree}")
    print(f"{'=' * 60}\n")

    # Build Omni config
    omni_kwargs = {
        "model": args.model,
        "vae_use_tiling": args.vae_use_tiling,
        "enable_cpu_offload": args.enable_cpu_offload,
        "parallel_config": parallel_config,
    }
    if args.flow_shift is not None:
        omni_kwargs["flow_shift"] = args.flow_shift

    # Initialize Omni
    omni = Omni(**omni_kwargs)

    # Build prompt with optional reference image
    prompt_data = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
    }

    # Add reference image for R2V mode
    if args.reference_image:
        prompt_data["multi_modal_data"] = {
            "image": args.reference_image,
        }

    # Generate video
    print("Generating video...")
    generation_start = time.perf_counter()

    frames = omni.generate(
        prompt_data,
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
        ),
    )

    generation_time = time.perf_counter() - generation_start
    print(f"Generation completed in {generation_time:.2f} seconds")

    # Extract and save video
    if not frames or len(frames) == 0:
        print("No output generated")
        return

    output = frames[0]
    if not hasattr(output, "request_output") or not output.request_output:
        print(f"Unexpected output format: {type(output)}")
        return

    req_output = output.request_output
    if isinstance(req_output, list):
        req_output = req_output[0]

    if not hasattr(req_output, "images") or not req_output.images:
        print("No images in output")
        return

    video_frames = req_output.images[0]

    # Convert to numpy array if needed
    if isinstance(video_frames, list):
        video_frames = np.stack([np.array(f) for f in video_frames])
    elif hasattr(video_frames, "numpy"):
        video_frames = video_frames.numpy()
    elif isinstance(video_frames, np.ndarray):
        pass  # Already numpy
    else:
        video_frames = np.array(video_frames)

    # Handle various output formats
    # Expected final shape: [T, H, W, C] for imageio
    if video_frames.ndim == 5:
        # [B, C, T, H, W] or [B, T, H, W, C] -> remove batch
        video_frames = video_frames[0]

    if video_frames.ndim == 4:
        # Check if channels first: [C, T, H, W] where C=3
        if video_frames.shape[0] == 3 and video_frames.shape[1] > 3:
            # [C, T, H, W] -> [T, H, W, C]
            video_frames = np.transpose(video_frames, (1, 2, 3, 0))
        # Check if channels last but wrong order: [T, C, H, W]
        elif video_frames.shape[1] == 3:
            # [T, C, H, W] -> [T, H, W, C]
            video_frames = np.transpose(video_frames, (0, 2, 3, 1))

    # Ensure correct dtype for video saving
    if video_frames.dtype == np.float32 or video_frames.dtype == np.float64:
        # Normalize from [-1, 1] or [0, 1] to [0, 255]
        if video_frames.min() < 0:
            video_frames = (video_frames + 1) / 2  # [-1, 1] -> [0, 1]
        video_frames = (video_frames * 255).clip(0, 255).astype(np.uint8)

    print(f"Video frames shape: {video_frames.shape}, dtype: {video_frames.dtype}")
    save_video(video_frames, args.output, args.fps)


if __name__ == "__main__":
    main()
