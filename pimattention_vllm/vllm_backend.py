from __future__ import annotations
from dataclasses import dataclass
import os
import torch
from vllm.v1.attention.backend import AttentionBackend, AttentionImpl, AttentionMetadataBuilder, AttentionType, CommonAttentionMetadata, AttentionCGSupport, MultipleOf
from vllm.v1.kv_cache_interface import KVCacheLayout
from .core import OfflinePolicy, PIMAttentionRuntime, append_kv
_SHARED_RUNTIMES: dict[OfflinePolicy, PIMAttentionRuntime] = {}

@dataclass
class PIMMetadata:
    num_actual_tokens: int
    block_size: int
    query_starts_cpu: tuple[int, ...]
    seq_lengths_cpu: tuple[int, ...]
    block_table_cpu: tuple[tuple[int, ...], ...]
    slot_mapping: torch.Tensor

class PIMMetadataBuilder(AttentionMetadataBuilder[PIMMetadata]):
    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool=False) -> PIMMetadata:
        c = common_attn_metadata
        nreq = c.num_reqs
        n = c.num_actual_tokens
        starts = tuple((int(x) for x in c.query_start_loc_cpu[:nreq + 1].tolist()))
        lengths = tuple((int(x) for x in c.seq_lens[:nreq].cpu().tolist()))
        blocks = tuple((tuple((int(b) for b in row)) for row in c.block_table_tensor[:nreq].cpu().tolist()))
        if not isinstance(c.causal, bool) or not c.causal:
            raise NotImplementedError('Only causal decoder attention is supported')
        return PIMMetadata(n, self.block_size, starts, lengths, blocks, c.slot_mapping[:n])

class PIMAttentionImpl(AttentionImpl[PIMMetadata]):

    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: int | None=None, alibi_slopes: list[float] | None=None, sliding_window: int | None=None, kv_cache_dtype: str='auto', logits_soft_cap: float | None=None, attn_type: str=AttentionType.DECODER, kv_sharing_target_layer_name: str | None=None, **kwargs):
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_heads % num_kv_heads:
            raise ValueError('Q head count must be divisible by KV head count')
        if alibi_slopes is not None or kv_sharing_target_layer_name is not None:
            raise NotImplementedError('ALiBi and explicit KV layer sharing unsupported')
        if kv_cache_dtype not in ('auto', 'float16', 'bfloat16'):
            raise NotImplementedError('Quantized KV cache unsupported')
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = scale
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap
        policy_path = os.environ.get('PIMATTENTION_POLICY_PATH')
        self.policy = OfflinePolicy.load(policy_path) if policy_path else OfflinePolicy()
        if self.policy not in _SHARED_RUNTIMES:
            _SHARED_RUNTIMES[self.policy] = PIMAttentionRuntime(self.policy, keep_trace=os.environ.get('PIMATTENTION_TRACE', '') == '1')
        self.runtime = _SHARED_RUNTIMES[self.policy]

    def forward(self, layer, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor, attn_metadata: PIMMetadata | None, output: torch.Tensor, output_scale: torch.Tensor | None=None, output_block_scale: torch.Tensor | None=None) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError('Fused output quantization unsupported')
        if attn_metadata is None:
            return output.zero_()
        if attn_metadata.block_size != self.policy.block_size:
            raise ValueError('vLLM --block-size must equal offline policy B')
        n = attn_metadata.num_actual_tokens
        if n > query.size(0) or n > key.size(0) or n > value.size(0):
            raise ValueError('Actual token count exceeds Q/K/V rows')
        append_kv(kv_cache, key[:n], value[:n], attn_metadata.slot_mapping)
        result = self.runtime.execute(query[:n], kv_cache, attn_metadata.query_starts_cpu, attn_metadata.seq_lengths_cpu, attn_metadata.block_table_cpu, scale=self.scale, sliding_window=self.sliding_window, logits_soft_cap=self.logits_soft_cap)
        output[:n].copy_(result)
        return output

class PIMAttentionBackend(AttentionBackend):
    forward_includes_kv_cache_update = True
    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    supported_kv_cache_dtypes = ['auto', 'float16', 'bfloat16']

    @staticmethod
    def get_name() -> str:
        return 'PIMATTENTION'

    @staticmethod
    def get_impl_cls() -> type[PIMAttentionImpl]:
        return PIMAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[PIMMetadataBuilder]:
        return PIMMetadataBuilder

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBNHC,)

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [MultipleOf(1)]

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False
