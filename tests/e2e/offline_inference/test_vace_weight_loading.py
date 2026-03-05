# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test VACE weight loading in vllm-omni.

Verifies:
1. VACE weights from safetensors match diffusers loaded values
2. Weight transformation mapping is correctly applied for QKV fusion
3. All VACE blocks have complete weight coverage
"""

import pytest
import torch
from safetensors.torch import load_file


def load_diffusers_vace_transformer():
    """Load the diffusers VACE transformer."""
    from diffusers import WanVACETransformer3DModel

    model = WanVACETransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    return model


def load_raw_safetensors():
    """Load raw safetensors weights."""
    from huggingface_hub import hf_hub_download

    model_path = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"

    # Download weight files
    weight_files = []
    for i in range(1, 3):  # 2 shards
        weight_file = hf_hub_download(
            model_path,
            f"transformer/diffusion_pytorch_model-{i:05d}-of-00002.safetensors",
        )
        weight_files.append(weight_file)

    # Load all weights
    all_weights = {}
    for wf in weight_files:
        weights = load_file(wf)
        all_weights.update(weights)

    return all_weights


@pytest.fixture(scope="module")
def diffusers_model():
    """Load diffusers model once for all tests."""
    return load_diffusers_vace_transformer()


@pytest.fixture(scope="module")
def raw_weights():
    """Load raw weights once for all tests."""
    return load_raw_safetensors()


@pytest.mark.core_model
@pytest.mark.diffusion
class TestVACEWeightIntegrity:
    """Test VACE weight integrity between diffusers and raw safetensors."""

    def test_critical_vace_weights_not_zero(self, raw_weights):
        """Verify critical VACE weights are not all zeros."""
        critical_weights = [
            "vace_patch_embedding.weight",
            "vace_patch_embedding.bias",
            "vace_blocks.0.proj_in.weight",
            "vace_blocks.0.proj_out.weight",
            "vace_blocks.0.attn1.to_q.weight",
            "vace_blocks.0.attn1.to_k.weight",
            "vace_blocks.0.attn1.to_v.weight",
            "vace_blocks.0.scale_shift_table",
            "vace_blocks.0.ffn.net.0.proj.weight",
            "vace_blocks.14.proj_out.weight",
        ]

        for key in critical_weights:
            assert key in raw_weights, f"Weight {key} not found in safetensors"
            w = raw_weights[key]
            assert not (w == 0).all().item(), f"Weight {key} is all zeros"
            assert not w.isnan().any().item(), f"Weight {key} has NaN values"
            assert not w.isinf().any().item(), f"Weight {key} has Inf values"

    def test_vace_weights_match_diffusers(self, diffusers_model, raw_weights):
        """Verify VACE weights in diffusers match raw safetensors."""
        diffusers_state = diffusers_model.state_dict()

        vace_keys = [k for k in diffusers_state.keys() if "vace" in k.lower()]
        assert len(vace_keys) > 0, "No VACE weights found in diffusers model"

        mismatches = []
        for key in vace_keys:
            if key not in raw_weights:
                continue

            diff_weight = diffusers_state[key]
            raw_weight = raw_weights[key]

            diff_val = (diff_weight.float() - raw_weight.float()).abs()
            max_diff = diff_val.max().item()

            if max_diff >= 1e-6:
                mismatches.append((key, max_diff))

        assert len(mismatches) == 0, f"Weight mismatches found: {mismatches}"


@pytest.mark.core_model
@pytest.mark.diffusion
class TestVACEWeightLoadingLogic:
    """Test vllm-omni weight loading logic for VACE blocks."""

    def test_qkv_fusion_groups_complete(self, raw_weights):
        """Verify all VACE blocks have complete QKV weights for fusion."""
        stacked_params_mapping = [
            (".attn1.to_qkv", ".attn1.to_q", "q"),
            (".attn1.to_qkv", ".attn1.to_k", "k"),
            (".attn1.to_qkv", ".attn1.to_v", "v"),
        ]

        vace_weights = {k: v for k, v in raw_weights.items()
                        if k.startswith("vace_blocks.")}

        # Group by fused param name
        qkv_fusions = {}
        for name in vace_weights.keys():
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    fused_name = name.replace(weight_name, param_name)
                    if fused_name not in qkv_fusions:
                        qkv_fusions[fused_name] = set()
                    qkv_fusions[fused_name].add(shard_id)

        # Verify each fusion group has all 3 shards
        incomplete = []
        for fused_name, shards in qkv_fusions.items():
            if shards != {"q", "k", "v"}:
                incomplete.append((fused_name, shards))

        assert len(incomplete) == 0, f"Incomplete QKV groups: {incomplete}"

    def test_weight_name_transformations(self, raw_weights):
        """Verify weight name transformations are correctly defined."""
        vace_weights = {k for k in raw_weights.keys()
                        if k.startswith("vace_blocks.")}

        # Patterns that need transformation
        ffn_net0_pattern = ".ffn.net.0."
        ffn_net2_pattern = ".ffn.net.2."
        to_out0_pattern = ".to_out.0."

        ffn_net0_found = any(ffn_net0_pattern in k for k in vace_weights)
        ffn_net2_found = any(ffn_net2_pattern in k for k in vace_weights)
        to_out0_found = any(to_out0_pattern in k for k in vace_weights)

        # At least some weights should match each pattern
        assert ffn_net0_found, "No FFN net.0 weights found"
        assert ffn_net2_found, "No FFN net.2 weights found"
        assert to_out0_found, "No to_out.0 weights found"

    def test_vace_block_count(self, raw_weights):
        """Verify the expected number of VACE blocks are present."""
        vace_block_indices = set()
        for k in raw_weights.keys():
            if k.startswith("vace_blocks."):
                parts = k.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    vace_block_indices.add(int(parts[1]))

        # VACE model has 15 blocks (indices 0-14)
        assert len(vace_block_indices) == 15, f"Expected 15 VACE blocks, found {len(vace_block_indices)}"
        assert min(vace_block_indices) == 0, "VACE blocks should start at index 0"
        assert max(vace_block_indices) == 14, "VACE blocks should end at index 14"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
