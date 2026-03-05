# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
VACE (Video Creation and Editing) Pipeline for Wan2.1.

VACE is an all-in-one model for video creation and editing supporting:
- R2V: Reference-to-Video generation
- V2V: Video-to-Video editing
- MV2V: Masked Video-to-Video editing
- T2V: Text-to-Video generation

This pipeline supports the official HuggingFace VACE models:
- Wan-AI/Wan2.1-VACE-1.3B-diffusers
- Wan-AI/Wan2.1-VACE-14B-diffusers
"""

from __future__ import annotations

import html
import logging
import os
import re
from collections.abc import Iterable

import ftfy
import numpy as np
import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    create_transformer_from_config,
    load_transformer_config,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)


def _basic_clean(text: str) -> str:
    """Clean text by fixing unicode and unescaping HTML entities."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def _whitespace_clean(text: str) -> str:
    """Normalize whitespace in text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _prompt_clean(text: str) -> str:
    """Clean prompt text (matching diffusers' prompt_clean)."""
    return _whitespace_clean(_basic_clean(text))


def get_wan22_vace_post_process_func(od_config: OmniDiffusionConfig):
    """Get post-processing function for VACE pipeline.

    Note: Post-processing runs on the main process, not workers.
    We move tensor to CPU to avoid OOM since workers hold GPU memory.

    Returns tensor in [B, C, F, H, W] format for downstream processing.
    """
    config_output_type = getattr(od_config, "output_type", "pt")

    def post_process_func(video: torch.Tensor, output_type: str | None = None):
        effective_output_type = output_type if output_type is not None else config_output_type
        if effective_output_type == "latent":
            return video

        # Move to CPU to avoid OOM (main process doesn't have GPU memory)
        # Return tensor directly in [B, C, F, H, W] format
        return video.cpu()

    return post_process_func


def get_wan22_vace_pre_process_func(od_config: OmniDiffusionConfig):
    """Pre-process function for VACE: handle reference images/videos and masks."""
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)

            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None

            # Handle reference image for R2V (Reference-to-Video)
            if multi_modal_data:
                ref_image = multi_modal_data.get("image") or multi_modal_data.get("reference_image")
                if ref_image is not None:
                    if isinstance(ref_image, str):
                        ref_image = PIL.Image.open(ref_image).convert("RGB")

                    # Calculate dimensions if not provided
                    if request.sampling_params.height is None or request.sampling_params.width is None:
                        max_area = 480 * 832  # Default for 480P
                        aspect_ratio = ref_image.height / ref_image.width
                        mod_value = 16
                        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

                        if request.sampling_params.height is None:
                            request.sampling_params.height = height
                        if request.sampling_params.width is None:
                            request.sampling_params.width = width

                    # Resize reference image
                    ref_image = ref_image.resize(
                        (request.sampling_params.width, request.sampling_params.height),
                        PIL.Image.Resampling.LANCZOS,
                    )

                    # Preprocess for VAE
                    prompt["additional_information"]["preprocessed_reference"] = video_processor.preprocess(
                        ref_image,
                        height=request.sampling_params.height,
                        width=request.sampling_params.width,
                    )

                # Handle mask for MV2V (Masked Video-to-Video)
                mask = multi_modal_data.get("mask")
                if mask is not None:
                    if isinstance(mask, str):
                        mask = PIL.Image.open(mask).convert("L")
                    prompt["additional_information"]["mask"] = mask

                # Handle source video for V2V
                source_video = multi_modal_data.get("video") or multi_modal_data.get("source_video")
                if source_video is not None:
                    prompt["additional_information"]["source_video"] = source_video

            request.prompts[i] = prompt
        return request

    return pre_process_func


class Wan22VACEPipeline(nn.Module, SupportImageInput, CFGParallelMixin):
    """
    VACE (Video Creation and Editing) Pipeline for Wan2.1.

    This pipeline supports various video generation and editing tasks:
    - T2V: Text-to-Video (prompt only)
    - R2V: Reference-to-Video (prompt + reference image)
    - V2V: Video-to-Video (prompt + source video)
    - MV2V: Masked Video-to-Video (prompt + source video + mask)
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config

        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Set up weights sources for transformer
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # Load tokenizer and text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model,
            subfolder="text_encoder",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(self.device)

        # Load VAE
        self.vae = DistributedAutoencoderKLWan.from_pretrained(
            model,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
            use_slicing=od_config.vae_use_slicing,
            use_tiling=od_config.vae_use_tiling,
        ).to(self.device)

        # Load transformer (with VACE blocks)
        self.transformer_config = load_transformer_config(model, "transformer", local_files_only)
        self.transformer = create_transformer_from_config(self.transformer_config)

        # Load scheduler
        # Default flow_shift=3.0 matches diffusers' WanVACEPipeline for 480p
        # Use 5.0 for 720p if needed
        flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 3.0
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            prediction_type="flow_prediction",
        )

        # VAE scale factors
        vae_config = self.vae.config
        self.vae_scale_factor_spatial = getattr(vae_config, "scale_factor_spatial", 8)
        self.vae_scale_factor_temporal = getattr(vae_config, "scale_factor_temporal", 4)

        # CFG state
        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
    ) -> torch.Tensor:
        """Encode text prompt to embeddings.

        Matches diffusers' _get_t5_prompt_embeds implementation:
        1. Clean prompts with ftfy and html unescape
        2. Tokenize with max_length
        3. Encode with text encoder
        4. Truncate embeddings to actual sequence length
        5. Re-pad with zeros to max_sequence_length

        Args:
            prompt: Text prompt(s) to encode
            device: Target device
            num_videos_per_prompt: Number of videos per prompt
            max_sequence_length: Maximum sequence length (default 226 matches diffusers)

        Returns:
            Text embeddings of shape [batch, max_sequence_length, hidden_dim]
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        # Clean prompts (matching diffusers)
        prompt = [_prompt_clean(p) for p in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        # Get actual sequence lengths (number of non-padding tokens)
        seq_lens = attention_mask.gt(0).sum(dim=1).long()

        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

        # Truncate to actual sequence length and re-pad with zeros
        # This matches diffusers' behavior exactly
        prompt_embeds_list = [emb[:length] for emb, length in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([emb, emb.new_zeros(max_sequence_length - emb.size(0), emb.size(1))])
                for emb in prompt_embeds_list
            ],
            dim=0,
        )

        # Duplicate for num_videos_per_prompt
        if num_videos_per_prompt > 1:
            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prepare latents for denoising."""
        # Calculate latent dimensions
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        shape = (batch_size, num_channels_latents, latent_frames, latent_height, latent_width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents

    def _create_default_vace_context(
        self,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Create default VACE context for T2V mode (no reference image).

        For T2V, we create an "empty" context following diffusers' approach:
        - Create a zero video in pixel space
        - Create an all-ones mask (generate everything)
        - Encode the zero video through VAE to get latents
        - Apply VAE normalization to the latents
        - Encode the mask spatially

        IMPORTANT: We MUST actually encode zero pixels through the VAE,
        NOT assume that zero pixels → zero latents. The VAE is a neural
        network, so encoding zeros gives some non-trivial latent output.

        Args:
            num_frames: Number of video frames to generate
            height: Output video height
            width: Output video width
            device: Target device
            dtype: Target dtype

        Returns:
            List containing a single VACE context tensor [C, T, H', W']
        """
        if self.transformer.vace_patch_embedding is None:
            return None

        # Get expected input channels
        vace_in_channels = self.transformer.vace_patch_embedding.weight.shape[1]

        # Calculate latent dimensions
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # VAE channels
        vae_channels = self.vae.config.z_dim

        # Create zero video in pixel space [1, 3, T, H, W]
        # Following diffusers: preprocess_conditions sets video = zeros when video=None
        zero_video = torch.zeros(1, 3, num_frames, height, width, device=device, dtype=dtype)

        # Create all-ones mask [1, 1, T, H, W]
        # For T2V: mask = ones means "generate everything"
        mask_pixel = torch.ones(1, 1, num_frames, height, width, device=device, dtype=dtype)

        # Following diffusers' prepare_video_latents:
        # For mask!=None case (T2V has mask=ones, not None):
        #   inactive = video * (1 - mask) = 0 * 0 = 0 (zeros)
        #   reactive = video * mask = 0 * 1 = 0 (zeros)
        # Both are zeros in pixel space, but we still need to encode through VAE

        # Prepare VAE normalization constants
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=torch.float32).view(
            1, vae_channels, 1, 1, 1
        )
        latents_std_inv = 1.0 / torch.tensor(self.vae.config.latents_std, device=device, dtype=torch.float32).view(
            1, vae_channels, 1, 1, 1
        )

        # Encode zero video through VAE
        # Note: VAE expects dtype matching its parameters
        vae_dtype = self.vae.dtype
        with torch.no_grad():
            zero_latent = self.vae.encode(zero_video.to(dtype=vae_dtype)).latent_dist.mode()

        # Apply VAE normalization: (latent - mean) * (1/std)
        zero_latent_normalized = ((zero_latent.float() - latents_mean) * latents_std_inv).to(dtype)

        # For T2V, both inactive and reactive are the same (encoded zero video)
        inactive_latent = zero_latent_normalized
        reactive_latent = zero_latent_normalized.clone()

        # Encode mask spatially using VAE stride (temporal=4, spatial=8)
        vae_stride = (self.vae_scale_factor_temporal, self.vae_scale_factor_spatial, self.vae_scale_factor_spatial)
        mask_encoded = self._encode_mask_spatial(mask_pixel, vae_stride=vae_stride)
        # mask_encoded is now [1, 64, T_latent, H_latent, W_latent]

        # Concatenate: inactive (16) + reactive (16) + mask (64) = 96
        vace_ctx = torch.cat([inactive_latent, reactive_latent, mask_encoded], dim=1)

        # Adjust to expected channels if needed
        if vace_ctx.shape[1] != vace_in_channels:
            if vace_ctx.shape[1] > vace_in_channels:
                vace_ctx = vace_ctx[:, :vace_in_channels, :, :, :]
            else:
                padding = torch.zeros(
                    1,
                    vace_in_channels - vace_ctx.shape[1],
                    num_latent_frames,
                    latent_height,
                    latent_width,
                    device=device,
                    dtype=dtype,
                )
                vace_ctx = torch.cat([vace_ctx, padding], dim=1)

        # Remove batch dimension
        vace_ctx = vace_ctx[0]
        return [vace_ctx]

    def _encode_mask_spatial(
        self,
        mask: torch.Tensor,
        vae_stride: tuple[int, int, int] = (4, 8, 8),
    ) -> torch.Tensor:
        """Encode mask using spatial stride sampling (8x8 patches -> 64 channels).

        Following the official ali-vilab VACE implementation:
        1. Remove channel dim: [C, T, H, W]
        2. Compute latent dimensions
        3. Reshape: [T, H_latent, stride, W_latent, stride]
        4. Permute to: [stride, stride, T, H_latent, W_latent]
        5. Reshape to: [64, T, H_latent, W_latent]
        6. Interpolate temporally to latent frames

        Args:
            mask: Binary mask tensor [B, 1, T, H, W] in pixel space
            vae_stride: VAE stride (temporal, spatial_h, spatial_w), typically (4, 8, 8)

        Returns:
            Encoded mask tensor [B, 64, T_latent, H_latent, W_latent]
        """
        B, C, T, H, W = mask.shape

        # Compute latent dimensions (matching official implementation)
        T_latent = (T + vae_stride[0] - 1) // vae_stride[0]
        H_latent = 2 * (H // (vae_stride[1] * 2))  # = H // stride
        W_latent = 2 * (W // (vae_stride[2] * 2))  # = W // stride

        result_masks = []
        for b in range(B):
            # Get single mask [T, H, W]
            m = mask[b, 0, :, :, :]

            # Reshape to separate spatial stride patches
            # [T, H, W] -> [T, H_latent, stride, W_latent, stride]
            m = m.view(T, H_latent, vae_stride[1], W_latent, vae_stride[2])

            # Permute: [stride, stride, T, H_latent, W_latent]
            m = m.permute(2, 4, 0, 1, 3)

            # Reshape to channels: [64, T, H_latent, W_latent]
            m = m.reshape(vae_stride[1] * vae_stride[2], T, H_latent, W_latent)

            # Interpolate temporally to latent frame count
            # Use 'nearest-exact' mode to match diffusers implementation
            m = torch.nn.functional.interpolate(
                m.unsqueeze(0),  # [1, 64, T, H_latent, W_latent]
                size=(T_latent, H_latent, W_latent),
                mode="nearest-exact",
            ).squeeze(0)  # [64, T_latent, H_latent, W_latent]

            result_masks.append(m)

        # Stack batch: [B, 64, T_latent, H_latent, W_latent]
        return torch.stack(result_masks, dim=0)

    def _create_vace_context_from_reference(
        self,
        preprocessed_ref: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Create VACE context from a reference image.

        VACE context structure (96 channels for standard VACE models):
        - Inactive latents (16ch): VAE-encoded regions NOT to change (reference frame)
        - Reactive latents (16ch): VAE-encoded regions TO change (zeros for R2V)
        - Mask encoding (64ch): 8x8 spatial stride mask encoding

        The channel count is read dynamically from the model's vace_patch_embedding.

        Args:
            preprocessed_ref: Preprocessed reference image tensor [1, C, H, W] from VideoProcessor
            num_frames: Number of video frames to generate
            height: Output video height
            width: Output video width
            device: Target device
            dtype: Target dtype

        Returns:
            List containing a single VACE context tensor [C, T, H', W']
        """
        # Check what in_channels the VACE model expects
        if self.transformer.vace_patch_embedding is None:
            logger.warning("VACE patch embedding not initialized, cannot create VACE context")
            return None

        # Get expected input channels from vace_patch_embedding weight [out, in, kT, kH, kW]
        vace_in_channels = self.transformer.vace_patch_embedding.weight.shape[1]

        # Calculate latent dimensions
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # VAE latent channels (typically 16)
        vae_channels = self.vae.config.z_dim

        # Encode reference image through VAE
        # preprocessed_ref is [1, C, H, W] from VideoProcessor (C=3 for RGB)
        ref_tensor = preprocessed_ref.to(device=device, dtype=self.vae.dtype)
        # Expand to video format [B, C, F, H, W] - single frame
        ref_tensor = ref_tensor.unsqueeze(2)  # [1, C, 1, H, W]

        # Create full video for VAE encoding
        # For R2V: first frame is reference, rest are black/zeros
        video_with_ref = torch.cat(
            [ref_tensor, ref_tensor.new_zeros(1, ref_tensor.shape[1], num_frames - 1, height, width)],
            dim=2,
        )

        # Create binary mask in PIXEL space for R2V: 0 for first frame, 1 for rest
        # Shape: [1, 1, num_frames, height, width] (pixel space for 8x8 encoding)
        mask_pixel = torch.ones(1, 1, num_frames, height, width, device=device, dtype=dtype)
        mask_pixel[:, :, 0] = 0  # First frame is conditioned (reference)

        # CRITICAL: Apply mask in PIXEL space BEFORE encoding, not in latent space!
        # This matches diffusers: encode(video * mask) != encode(video) * mask
        # because VAE is non-linear.
        mask_pixel_expanded = mask_pixel.expand(-1, 3, -1, -1, -1).to(self.vae.dtype)  # [1, 3, T, H, W]

        # Inactive: regions to preserve (reference frame)
        inactive_video = video_with_ref * (1 - mask_pixel_expanded)  # [1, 3, T, H, W]
        # Reactive: regions to generate (zeros, will be generated)
        reactive_video = video_with_ref * mask_pixel_expanded  # [1, 3, T, H, W]

        # Encode through VAE to get latents
        # Use .mode() (not .sample()) to match diffusers' sample_mode="argmax" behavior
        with torch.no_grad():
            inactive_latent = self.vae.encode(inactive_video).latent_dist.mode()
            reactive_latent = self.vae.encode(reactive_video).latent_dist.mode()
        # latent shapes: [1, 16, T_latent, H', W']

        # Normalize latents using VAE config
        # Compute normalization constants in float32 for precision, then convert result
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=torch.float32).view(
            1, vae_channels, 1, 1, 1
        )
        latents_std_inv = 1.0 / torch.tensor(self.vae.config.latents_std, device=device, dtype=torch.float32).view(
            1, vae_channels, 1, 1, 1
        )
        inactive_latent = ((inactive_latent.float() - latents_mean) * latents_std_inv).to(dtype)
        reactive_latent = ((reactive_latent.float() - latents_mean) * latents_std_inv).to(dtype)

        # Encode mask spatially using VAE stride -> 64 channels
        # _encode_mask_spatial handles both spatial (8x8) and temporal encoding
        vae_stride = (self.vae_scale_factor_temporal, self.vae_scale_factor_spatial, self.vae_scale_factor_spatial)
        mask_encoded = self._encode_mask_spatial(mask_pixel, vae_stride=vae_stride)
        # mask_encoded is now [1, 64, T_latent, H_latent, W_latent]

        # Concatenate all components to create VACE context:
        # inactive (16) + reactive (16) + mask_encoding (64) = 96 channels
        vace_ctx = torch.cat(
            [
                inactive_latent,  # vae_channels (16)
                reactive_latent,  # vae_channels (16)
                mask_encoded,  # 64 channels from 8x8 spatial encoding
            ],
            dim=1,
        )  # [1, vace_in_channels, T, H', W']

        # Verify channel count matches expected
        if vace_ctx.shape[1] != vace_in_channels:
            logger.warning(
                f"VACE context channels mismatch: got {vace_ctx.shape[1]}, expected {vace_in_channels}. "
                f"Adjusting with padding/truncation."
            )
            if vace_ctx.shape[1] > vace_in_channels:
                vace_ctx = vace_ctx[:, :vace_in_channels, :, :, :]
            else:
                padding = torch.zeros(
                    1,
                    vace_in_channels - vace_ctx.shape[1],
                    num_latent_frames,
                    latent_height,
                    latent_width,
                    device=device,
                    dtype=dtype,
                )
                vace_ctx = torch.cat([vace_ctx, padding], dim=1)

        # Remove batch dimension for VACE context format [C, T, H', W']
        vace_ctx = vace_ctx[0]
        return [vace_ctx]

    def check_inputs(
        self,
        prompt: str | None,
        height: int,
        width: int,
        negative_prompt: str | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
    ):
        """Validate inputs before generation."""
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 16 but are {height} and {width}.")

        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`. Cannot leave both undefined.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`. Please provide only one.")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "Cannot forward both `negative_prompt` and `negative_prompt_embeds`. Please provide only one."
            )

    def forward(
        self,
        req: OmniDiffusionRequest,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: str | None = None,
        generator: torch.Generator | None = None,
        output_type: str | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        """Generate or edit video using VACE.

        Args:
            req: Diffusion request containing prompt and optional multi-modal data
            height: Output video height
            width: Output video width
            num_frames: Number of output frames
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            negative_prompt: Negative prompt for CFG
            generator: Random generator for reproducibility
            output_type: Output format ("np", "pt", or "latent"). Defaults to config value.

        Returns:
            DiffusionOutput containing the generated video
        """
        device = self.device

        # Use output_type from config if not explicitly passed
        if output_type is None:
            output_type = getattr(self.od_config, "output_type", "np")

        # Get parameters from request
        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames = req.sampling_params.num_frames or num_frames
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        guidance_scale = req.sampling_params.guidance_scale or guidance_scale
        generator = req.sampling_params.generator or generator

        # Extract prompt and pre-computed embeddings
        prompt = None
        prompt_embeds = None
        negative_prompt_embeds = None

        if len(req.prompts) > 0:
            first_prompt = req.prompts[0]
            if isinstance(first_prompt, str):
                prompt = first_prompt
            else:
                prompt = first_prompt.get("prompt", "")
                # Treat empty string as None (allows prompt_embeds to be used)
                if prompt == "":
                    prompt = None
                negative_prompt = negative_prompt or first_prompt.get("negative_prompt")
                # Extract pre-computed embeddings if provided
                prompt_embeds = first_prompt.get("prompt_embeds")
                negative_prompt_embeds = first_prompt.get("negative_prompt_embeds")

        # Validate inputs
        self.check_inputs(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
        )

        # Set CFG state
        self._guidance_scale = guidance_scale
        dtype = torch.bfloat16

        # Encode prompts only if pre-computed embeddings not provided
        if prompt_embeds is None:
            if prompt is None:
                raise ValueError("Either prompt or prompt_embeds must be provided.")
            prompt_embeds = self.encode_prompt(prompt, device)
            prompt_embeds = prompt_embeds.to(dtype)
            # For CFG, encode negative prompt (use empty string if not provided)
            if guidance_scale > 1.0:
                neg_prompt = negative_prompt if negative_prompt else ""
                negative_prompt_embeds = self.encode_prompt(neg_prompt, device)
                negative_prompt_embeds = negative_prompt_embeds.to(dtype)
        else:
            # Use pre-computed embeddings
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
            # For CFG with pre-computed embeddings, encode empty negative if not provided
            elif guidance_scale > 1.0:
                negative_prompt_embeds = self.encode_prompt("", device)
                negative_prompt_embeds = negative_prompt_embeds.to(dtype)

        # Get VACE context - either pre-computed or from reference image
        vace_context = None
        vace_context_scale = req.sampling_params.vace_context_scale or 1.0
        vace_seq_len = req.sampling_params.vace_context_seq_len

        # First, check for pre-computed VACE context in sampling params
        if req.sampling_params.vace_context is not None:
            ctx = req.sampling_params.vace_context
            if isinstance(ctx, list) and len(ctx) > 0:
                vace_context = [t.to(device, dtype=dtype) for t in ctx]
            elif isinstance(ctx, torch.Tensor):
                vace_context = [ctx.to(device, dtype=dtype)]

        # If no pre-computed context, check for preprocessed reference image
        if vace_context is None and len(req.prompts) > 0:
            first_prompt = req.prompts[0]
            if isinstance(first_prompt, dict):
                additional_info = first_prompt.get("additional_information", {})
                preprocessed_ref = additional_info.get("preprocessed_reference")
                if preprocessed_ref is not None:
                    vace_context = self._create_vace_context_from_reference(
                        preprocessed_ref=preprocessed_ref,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        device=device,
                        dtype=dtype,
                    )

        # For T2V mode (no reference), create default VACE context
        if vace_context is None and self.transformer.vace_patch_embedding is not None:
            vace_context = self._create_default_vace_context(
                num_frames=num_frames,
                height=height,
                width=width,
                device=device,
                dtype=dtype,
            )

        # Prepare latents
        num_channels_latents = self.transformer_config.get("in_channels", 16)
        latents = self.prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=req.sampling_params.latents,
        )

        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # Denoising loop
        for i, t in enumerate(timesteps):
            self._current_timestep = t
            latent_model_input = latents.to(dtype)
            timestep = t.expand(latents.shape[0])

            # Forward through transformer
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                vace_context=vace_context,
                vace_seq_len=vace_seq_len,
                vace_context_scale=vace_context_scale,
                return_dict=False,
            )[0]

            # CFG
            if guidance_scale > 1.0 and negative_prompt_embeds is not None:
                noise_pred_uncond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    vace_context=vace_context,
                    vace_seq_len=vace_seq_len,
                    vace_context_scale=vace_context_scale,
                    return_dict=False,
                )[0]

                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        self._current_timestep = None

        # Clear cache
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()

        # Decode
        if output_type == "latent":
            output = latents
        else:
            latents = latents.to(self.vae.dtype)
            # Denormalize latents
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            output = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(output=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights, detecting and initializing VACE modules from transformer weights.

        VACE weights (vace_blocks.*, vace_patch_embedding.*) are expected to be
        merged into the transformer weights file, consistent with HuggingFace format.
        """
        # Collect all weights and separate VACE weights
        weights_list = list(weights)
        vace_weights = {}
        non_vace_weights = []

        for name, tensor in weights_list:
            # Check for VACE weight keys (with transformer. prefix from weights_sources)
            clean_name = name.removeprefix("transformer.")
            if clean_name.startswith("vace_blocks.") or clean_name.startswith("vace_patch_embedding."):
                vace_weights[clean_name] = tensor
            else:
                non_vace_weights.append((name, tensor))

        # Load non-VACE weights first
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(iter(non_vace_weights))

        # Initialize and load VACE weights if present
        if vace_weights:
            self.transformer.load_vace_weights(vace_weights)
            logger.debug(f"VACE modules initialized and loaded ({len(vace_weights)} tensors)")

        return loaded
