import random
from typing import Dict, List, Optional, Sequence

import torch

class HistoricalLatentStorage:
    """
    一个按周期（epoch）工作的历史 latent 快照存储。
    - 在采样（sampling）阶段通过 add_batch 添加数据。
    - 添加完所有数据后调用一次 finalize_and_prepare。
    - 在训练（training）阶段是只读的。
    - 在下一个 epoch 开始时通过 clear 清空。

    数据按 prompt (字符串) 分组。
    """

    def __init__(self, *, store_dtype: torch.dtype = torch.float16, pin_memory: bool = True):
        """
        初始化存储。
        Args:
            store_dtype: 在 CPU 上存储时使用的数据类型，float16 可以节省一半内存。
            pin_memory: 是否使用锁页内存，可以加速 CPU 到 GPU 的数据传输。
        """
        self._storage: Dict[str, List[torch.Tensor]] = {}
        self._prompt_keys: List[str] = []
        self._latent_shape: Optional[torch.Size] = None
        self._store_dtype = store_dtype
        self._pin_memory = pin_memory

    def clear(self):
        """清空所有数据，为新周期做准备。"""
        self._storage.clear()
        self._prompt_keys.clear()
        self._latent_shape = None

    @torch.no_grad()
    def add_batch(self, prompts: Sequence[str], latents_clean: torch.Tensor):
        """
        添加一个批次的数据。
        Args:
            prompts: 长度为 B 的字符串 prompt 列表。
            latents_clean: 形状为 [B, *latent_shape] 的张量，可以在任何设备上。
        """
        if latents_clean.ndim < 2:
            raise ValueError(f"latents_clean 应该是 [B,...] 形状, 但得到 {tuple(latents_clean.shape)}")

        B = latents_clean.shape[0]
        if len(prompts) != B:
            raise ValueError(f"prompts 长度 {len(prompts)} 与 batch_size {B} 不匹配")

        # 从第一个批次中记录 latent 的形状，用于后续返回空的张量
        if self._latent_shape is None:
            self._latent_shape = latents_clean.shape[1:]

        # 将数据转移到 CPU 进行快照存储
        lat_cpu = latents_clean.detach()
        if lat_cpu.dtype != self._store_dtype:
            lat_cpu = lat_cpu.to(self._store_dtype) # 转换类型以节省内存
        if lat_cpu.is_cuda:
            lat_cpu = lat_cpu.to("cpu", non_blocking=True) # 异步移至 CPU
        if self._pin_memory:
            lat_cpu = lat_cpu.pin_memory() # 使用锁页内存以加速传输

        for i, p in enumerate(prompts):
            self._storage.setdefault(p, []).append(lat_cpu[i])

    def finalize_and_prepare(self):
        """在所有数据添加完毕后，创建用于高效采样的 prompt key 列表。"""
        self._prompt_keys = list(self._storage.keys())

    @torch.no_grad()
    def sample(
        self,
        k: int,
        *,
        exclude_prompts: Sequence[str],
        device: torch.device | str,
        samples_per_prompt: int = 1, # <-- 每个prompt采样多少latents
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            k (int): 要选择的 *独特 prompts* 的数量。
            exclude_prompts (Sequence[str]): 需要排除的 prompt 列表。
            device (torch.device | str): 目标设备。
            samples_per_prompt (int): 对于每个被选中的 prompt，要采样多少个 latent 样本。
            non_blocking (bool): 是否使用异步传输
        """
        if self._latent_shape is None:
            # 如果从未添加过数据
            return torch.empty((0,), device=device)

        if not self._prompt_keys:
            # 兼容未调用 finalize_and_prepare 的情况
            self._prompt_keys = list(self._storage.keys())
        if not self._prompt_keys:
            return torch.empty((0, *self._latent_shape), device=device)

        exclude_set = set(exclude_prompts)
        available = [p for p in self._prompt_keys if p not in exclude_set]

        if len(available) == 0:
            # 严格模式：如果没有可用的 prompts，直接返回空张量
            return torch.empty((0, *self._latent_shape), device=device)

        # 核心逻辑：从可用 prompts 中无放回地采样 min(k, len(available)) 个
        chosen_prompts = random.sample(available, k=min(k, len(available)))
        # 2. 为每个选中的 prompt, 无放回地采样最多 x 个 latents
        sampled = []
        for p in chosen_prompts:
            latent_list_for_prompt = self._storage[p]

            # 核心逻辑：如果请求的数量 (samples_per_prompt) 超过了列表长度，
            # random.sample 会自动只采样列表中的所有元素，但顺序是随机的。
            # 为了更明确和安全，我们仍然使用 min 来确定采样数量。
            num_to_sample = min(samples_per_prompt, len(latent_list_for_prompt))

            # random.sample 执行无放回抽样
            chosen_latents = random.sample(latent_list_for_prompt, k=num_to_sample)
            sampled.extend(chosen_latents)

        if not sampled:
            return torch.empty((0, *self._latent_shape), device=device)

        lat = torch.stack(sampled, dim=0)  # 在 CPU 上的 (pinned) tensor
        # 异步地将最终的 stack tensor 移至目标设备
        return lat.to(device, non_blocking=non_blocking)
    def log_stats(self):
        """打印关于当前存储内容的详细统计信息。"""
        if not self._storage:
            print("Historical Storage is empty.")
            return
        
        num_unique_prompts = len(self._storage)
        
        latents_per_prompt = [len(v) for v in self._storage.values()]
        total_latents = sum(latents_per_prompt)
        
        print("\n--- Historical Latent Storage Stats ---")
        print(f"  - Unique Prompts Stored: {num_unique_prompts}")
        print(f"  - Total Latents Stored:  {total_latents}")
        print("---------------------------------------\n")