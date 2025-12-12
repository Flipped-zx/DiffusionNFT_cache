import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch


def prompt_key_from_ids_row(
    prompt_ids_row: torch.Tensor,
    *,
    method: str = "bytes_hash",
) -> int:
    """
    Turn one prompt_ids row ([seq_len]) into a stable-ish integer key.
    This is used to group samples by prompt.

    Args:
        prompt_ids_row: shape [seq_len], can be on GPU/CPU.
        method:
            - "bytes_hash": Python hash over raw bytes of numpy array (fast).
            - "sha1": stronger but slower.

    Returns:
        int key
    """
    x = prompt_ids_row.detach()
    if x.is_cuda:
        x = x.to("cpu", non_blocking=True)
    x = x.contiguous()

    arr = x.numpy()  # int64/int32 depending on tokenizer output
    if method == "bytes_hash":
        return hash(arr.tobytes())

    if method == "sha1":
        import hashlib

        h = hashlib.sha1(arr.tobytes()).hexdigest()
        # shrink to int
        return int(h[:16], 16)

    raise ValueError(f"Unknown method={method}")


@dataclass
class SnapshotPromptLatentCache:
    """
    A per-epoch snapshot cache:
      - build once after sampling
      - read-only during training
      - replace entirely next epoch sampling

    Stores:
      latents_clean_all: [N, *latent_shape] on CPU (pinned) by default
      key_to_indices: prompt_key -> list of indices in latents_clean_all
      keys: list of prompt_key for sampling
    """

    latents_clean_all: torch.Tensor  # CPU pinned recommended
    key_to_indices: Dict[int, List[int]]
    keys: List[int]

    @staticmethod
    def build_from_collated_samples(
        collated_samples: Dict,
        *,
        store_on_cpu: bool = True,
        pin_memory: bool = True,
        store_dtype: torch.dtype = torch.float16,
        key_method: str = "bytes_hash",
        seed: Optional[int] = None,
    ) -> "SnapshotPromptLatentCache":
        """
        Build snapshot from your collated_samples.

        Expected fields:
            collated_samples["latents_clean"]: [N, *latent_shape]
            collated_samples["prompt_ids"]:    [N, seq_len]

        Note:
            - If store_on_cpu=True, we move latents to CPU (optionally pinned).
            - During training, sample_k() moves selected latents to GPU.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        lat = collated_samples["latents_clean"].detach()
        if lat.dtype != store_dtype:
            lat = lat.to(store_dtype)

        if store_on_cpu:
            # move once, training will fetch slices back to GPU
            lat = lat.to("cpu", non_blocking=True)
            if pin_memory:
                lat = lat.pin_memory()

        prompt_ids_all = collated_samples["prompt_ids"]
        N = int(prompt_ids_all.shape[0])

        key_to_indices: Dict[int, List[int]] = {}
        # compute keys
        for i in range(N):
            k = prompt_key_from_ids_row(prompt_ids_all[i], method=key_method)
            if k not in key_to_indices:
                key_to_indices[k] = [i]
            else:
                key_to_indices[k].append(i)

        keys = list(key_to_indices.keys())
        return SnapshotPromptLatentCache(
            latents_clean_all=lat,
            key_to_indices=key_to_indices,
            keys=keys,
        )

    def sample_k(
        self,
        k: int,
        *,
        exclude_keys: Set[int],
        device: torch.device | str = "cuda",
    ) -> Tuple[Optional[torch.Tensor], List[int]]:
        """
        Sample up to K latents:
        - each from a different prompt
        - prompt not in exclude_keys
        - no duplication
        - if candidates < K, return fewer

        Returns:
        latents: [M, *latent_shape] on device, where M <= K
        keys:    list of prompt keys (length M)
        """
        # 可用 prompt keys
        candidates = [
            kk for kk in self.keys
            if kk not in exclude_keys and len(self.key_to_indices[kk]) > 0
        ]

        if len(candidates) == 0:
            return None, []

        # 抽最多 K 个，不重复
        picked_keys = random.sample(candidates, k=min(k, len(candidates)))

        # 每个 prompt 抽 1 个 latent
        idx_list = [random.choice(self.key_to_indices[kk]) for kk in picked_keys]

        lat_cpu = self.latents_clean_all[idx_list]
        lat_gpu = lat_cpu.to(device, non_blocking=True)

        return lat_gpu, picked_keys

    def make_exclude_keys_from_prompt_ids(
        self,
        prompt_ids: torch.Tensor,
        *,
        key_method: str = "bytes_hash",
    ) -> Set[int]:
        """
        Convenience: build exclude_keys set from a [B, seq_len] prompt_ids tensor.
        """
        B = int(prompt_ids.shape[0])
        out: Set[int] = set()
        for b in range(B):
            out.add(prompt_key_from_ids_row(prompt_ids[b], method=key_method))
        return out