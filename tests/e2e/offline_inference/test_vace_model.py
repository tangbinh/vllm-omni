# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from tests.utils import hardware_test
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Use 1.3B model for faster CI testing; 14B model for comprehensive tests
VACE_MODELS = ["Wan-AI/Wan2.1-VACE-1.3B-diffusers"]


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 1, "rocm": 2})
@pytest.mark.parametrize("model_name", VACE_MODELS)
def test_vace_t2v_generation(model_name: str, run_level):
    """Test basic Text-to-Video generation with VACE model."""
    m = None
    try:
        m = Omni(
            model=model_name,
            vae_use_tiling=True,
        )

        # Use minimal settings for testing
        # num_frames must satisfy: (num_frames - 1) % vae_scale_factor_temporal == 0
        # For Wan2.1, vae_scale_factor_temporal=4, so valid values are 5, 9, 13, 17, ...
        height = 480
        width = 832
        num_frames = 5

        outputs = m.generate(
            prompts="A cat sitting on a table",
            sampling_params_list=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=2,
                guidance_scale=5.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
            ),
        )

        # Validate output structure
        first_output = outputs[0]
        assert first_output.final_output_type == "image"

        if not hasattr(first_output, "request_output") or not first_output.request_output:
            raise ValueError("No request_output found in OmniRequestOutput")

        req_out = first_output.request_output[0]
        if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
            raise ValueError("Invalid request_output structure or missing 'images' key")

        frames = req_out.images[0]

        assert frames is not None
        # Output can be either:
        # - numpy array with shape (batch, num_frames, height, width, channels)
        # - list of PIL Images
        if hasattr(frames, "shape"):
            # numpy array format
            assert frames.shape[1] == num_frames
            assert frames.shape[2] == height
            assert frames.shape[3] == width
        elif isinstance(frames, list):
            # list of PIL Images
            from PIL import Image
            assert len(frames) == num_frames
            assert all(isinstance(f, Image.Image) for f in frames)
            assert frames[0].size == (width, height)
        else:
            raise ValueError(f"Unexpected frames type: {type(frames)}")
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 1, "rocm": 2})
@pytest.mark.parametrize("model_name", VACE_MODELS)
def test_vace_r2v_generation(model_name: str, run_level):
    """Test Reference-to-Video pipeline with a reference image.

    Note: This test validates the pipeline handles multi_modal_data correctly
    and doesn't crash. Full R2V generation requires pre-computed VACE context
    (96-channel latents from VACE encoder), which is beyond this structural test.
    """
    m = None
    try:
        m = Omni(
            model=model_name,
            vae_use_tiling=True,
        )

        height = 480
        width = 832
        num_frames = 5

        # Create a synthetic reference image for testing
        reference_image = _create_synthetic_reference_image(height, width)

        outputs = m.generate(
            prompts={
                "prompt": "Camera slowly zooms out from the scene",
                "multi_modal_data": {
                    "image": reference_image,
                },
            },
            sampling_params_list=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=2,
                guidance_scale=5.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
            ),
        )

        # Validate output structure
        first_output = outputs[0]
        assert first_output.final_output_type == "image"

        if not hasattr(first_output, "request_output") or not first_output.request_output:
            raise ValueError("No request_output found in OmniRequestOutput")

        req_out = first_output.request_output[0]
        if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
            raise ValueError("Invalid request_output structure or missing 'images' key")

        frames = req_out.images[0]

        assert frames is not None
        # Output can be either:
        # - numpy array with shape (batch, num_frames, height, width, channels)
        # - list of PIL Images
        if hasattr(frames, "shape"):
            # numpy array format
            assert frames.shape[1] == num_frames
            assert frames.shape[2] == height
            assert frames.shape[3] == width
        elif isinstance(frames, list):
            # list of PIL Images
            from PIL import Image
            assert len(frames) == num_frames
            assert all(isinstance(f, Image.Image) for f in frames)
            assert frames[0].size == (width, height)
        else:
            raise ValueError(f"Unexpected frames type: {type(frames)}")
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


def _create_synthetic_reference_image(height: int, width: int) -> Image.Image:
    """Create a synthetic gradient image for testing."""
    import numpy as np

    # Use numpy for fast gradient creation
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = np.tile(x, (height, 1))  # Red: horizontal gradient
    arr[:, :, 1] = np.tile(y.reshape(-1, 1), (1, width))  # Green: vertical gradient
    arr[:, :, 2] = (arr[:, :, 0].astype(np.uint16) + arr[:, :, 1]) // 2  # Blue: average
    return Image.fromarray(arr)
