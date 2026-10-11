from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from typing import Protocol
import json
import math
import threading
import torch

@dataclass(frozen=True)
class OfflinePolicy:
    block_size: int = 16
    num_domains: int = 4
    head_bind_group: int = 1
    pi_k: str = 'token_major'
    pi_v: str = 'token_major'
    tau: str = 'domain_grouped'
    max_head_blocks_per_domain: int = 100000
    pim_base_address: int = 1 << 32
    domain_stride_bytes: int = 1 << 42

    def validate(self, head_dim: int, dtype_bytes: int) -> None:
        if self.block_size <= 0 or self.num_domains <= 0:
            raise ValueError('block_size and num_domains must be positive')
        if self.head_bind_group <= 0 or self.max_head_blocks_per_domain <= 0:
            raise ValueError('head_bind_group and capacity must be positive')
        if self.pi_k != 'token_major' or self.pi_v != 'token_major':
            raise NotImplementedError('This prototype only implements token_major K/V')
        if self.tau not in ('block_order', 'domain_grouped'):
            raise ValueError('Unsupported execution schedule tau')
        if dtype_bytes <= 0 or head_dim <= 0:
            raise ValueError('Invalid dtype/head dimension')
        bytes_per_head_block = 2 * self.block_size * head_dim * dtype_bytes
        if bytes_per_head_block * self.max_head_blocks_per_domain > self.domain_stride_bytes:
            raise ValueError('PIM domain address regions would overlap')

    @classmethod
    def load(cls, path: str) -> 'OfflinePolicy':
        with open(path, encoding='utf-8') as handle:
            obj = json.load(handle)
        return cls(**obj)

@dataclass(frozen=True)
class PIMBlockAddress:
    cache_namespace: int
    vllm_physical_block_id: int
    kv_head: int
    pim_domain: int
    pim_slot: int
    k_address: int
    v_address: int

@dataclass(frozen=True)
class PIMCommand:
    request_index: int
    absolute_q_position: int
    logical_kv_block: int
    physical_kv_block: int
    kv_head: int
    pim_domain: int
    k_address: int
    v_address: int
    valid_token_offset: int
    valid_token_count: int

@dataclass
class PartialSoftmax:
    maximum: torch.Tensor
    denominator: torch.Tensor
    numerator: torch.Tensor

class PIMDevice(Protocol):

    def partial_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float, logits_soft_cap: float | None, address: PIMBlockAddress, token_offset: int) -> PartialSoftmax:
        ...

class PIMTransport(Protocol):

    def upload(self, address: PIMBlockAddress, token_offset: int, k: torch.Tensor, v: torch.Tensor) -> None:
        ...

    def execute(self, q: torch.Tensor, address: PIMBlockAddress, token_offset: int, token_count: int, scale: float, logits_soft_cap: float | None) -> PartialSoftmax:
        ...

class TransportPIMDevice:

    def __init__(self, transport: PIMTransport):
        self.transport = transport

    def partial_attention(self, q, k, v, *, scale, logits_soft_cap, address, token_offset):
        self.transport.upload(address, token_offset, k, v)
        return self.transport.execute(q, address, token_offset, k.shape[0], scale, logits_soft_cap)

class TorchPIMDevice:

    def partial_attention(self, q, k, v, *, scale, logits_soft_cap, address, token_offset) -> PartialSoftmax:
        if k.size(0) == 0:
            raise ValueError('No valid KV tokens')
        scores = k.float() @ q.float() * scale
        if logits_soft_cap is not None:
            scores = logits_soft_cap * torch.tanh(scores / logits_soft_cap)
        m = scores.max()
        weights = torch.exp(scores - m)
        l = weights.sum()
        o = weights @ v.float()
        return PartialSoftmax(m, l, o)

class PIMPhysicalTable:

    def __init__(self, policy: OfflinePolicy, head_dim: int, dtype_bytes: int):
        policy.validate(head_dim, dtype_bytes)
        self.policy = policy
        self.head_block_bytes = policy.block_size * head_dim * dtype_bytes
        self._next_slot = [0] * policy.num_domains
        self._bindings: dict[tuple[int, int, int], PIMBlockAddress] = {}
        self._lock = threading.Lock()

    def get_or_create(self, namespace: int, physical: int, head: int, logical: int) -> PIMBlockAddress:
        if physical < 0 or head < 0 or logical < 0:
            raise ValueError('Negative physical ID/head/logical block')
        key = (namespace, physical, head)
        with self._lock:
            found = self._bindings.get(key)
            if found is not None:
                return found
            p = self.policy
            domain = (logical + head // p.head_bind_group) % p.num_domains
            slot = self._next_slot[domain]
            if slot >= p.max_head_blocks_per_domain:
                raise MemoryError(f'PIM domain {domain} is out of head-block slots')
            self._next_slot[domain] += 1
            base = p.pim_base_address + domain * p.domain_stride_bytes + slot * 2 * self.head_block_bytes
            mapping = PIMBlockAddress(namespace, physical, head, domain, slot, base, base + self.head_block_bytes)
            self._bindings[key] = mapping
            return mapping

    def snapshot(self) -> list[PIMBlockAddress]:
        with self._lock:
            return list(self._bindings.values())

def append_kv(cache: torch.Tensor, key: torch.Tensor, value: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    if cache.ndim != 4 or key.ndim != 3 or value.shape != key.shape:
        raise ValueError('Invalid KV cache / input tensor rank')
    nb, hk, bs, two_d = cache.shape
    n, hk2, d = key.shape
    if hk2 != hk or two_d != 2 * d or slot_mapping.numel() < n:
        raise ValueError('KV head/dimension/slot mapping mismatch')
    flat = cache.transpose(1, 2).reshape(nb * bs, hk, two_d)
    if flat.data_ptr() != cache.data_ptr():
        raise ValueError('KV cache must have vLLM LBNHC stride order')
    slot = slot_mapping[:n].to(device=cache.device, dtype=torch.long)
    valid = slot >= 0
    ids = slot[valid]
    if ids.numel() == 0:
        return
    if (ids >= nb * bs).any().item():
        raise IndexError('KV slot is outside the physical cache')
    vals = torch.cat((key[valid], value[valid]), dim=-1)
    flat.index_copy_(0, ids, vals)

def combine(a: PartialSoftmax | None, b: PartialSoftmax) -> PartialSoftmax:
    if a is None:
        return b
    m = torch.maximum(a.maximum, b.maximum)
    wa = torch.exp(a.maximum - m)
    wb = torch.exp(b.maximum - m)
    return PartialSoftmax(m, wa * a.denominator + wb * b.denominator, wa * a.numerator + wb * b.numerator)

class PIMAttentionRuntime:

    def __init__(self, policy: OfflinePolicy, device: PIMDevice | None=None, keep_trace: bool=False):
        self.policy = policy
        self.device: PIMDevice = device if device is not None else TorchPIMDevice()
        self.keep_trace = keep_trace
        self.trace: list[PIMCommand] = []
        self._address_tables: dict[tuple[int, int], PIMPhysicalTable] = {}

    def _table(self, head_dim: int, dtype_bytes: int) -> PIMPhysicalTable:
        shape_key = (head_dim, dtype_bytes)
        if shape_key not in self._address_tables:
            shape_index = len(self._address_tables)
            adjusted = replace(self.policy, pim_base_address=self.policy.pim_base_address + shape_index * self.policy.num_domains * self.policy.domain_stride_bytes)
            self._address_tables[shape_key] = PIMPhysicalTable(adjusted, head_dim, dtype_bytes)
        return self._address_tables[shape_key]

    @torch.no_grad()
    def execute(self, query: torch.Tensor, cache: torch.Tensor, query_starts: tuple[int, ...], seq_lengths: tuple[int, ...], block_table: tuple[tuple[int, ...], ...], *, scale: float, logits_soft_cap: float | None=None, sliding_window: int | None=None) -> torch.Tensor:
        if query.ndim != 3 or cache.ndim != 4:
            raise ValueError('query/cache rank mismatch')
        nt, hq, d = query.shape
        nb, hk, bs, two_d = cache.shape
        if hq == 0 or hk == 0 or hq % hk or (two_d != 2 * d):
            raise ValueError('GQA/MHA head or dimension mismatch')
        if bs != self.policy.block_size:
            raise ValueError('vLLM block size must equal offline policy B')
        if len(seq_lengths) != len(block_table) or len(query_starts) != len(seq_lengths) + 1:
            raise ValueError('Invalid request metadata')
        if query_starts[0] != 0 or query_starts[-1] != nt:
            raise ValueError('Query starts inconsistent with token count')
        if sliding_window is not None and sliding_window <= 0:
            raise ValueError('sliding_window must be positive')
        if logits_soft_cap is not None and logits_soft_cap <= 0:
            raise ValueError('logits_soft_cap must be positive')
        element_bytes = cache.element_size()
        addresses = self._table(d, element_bytes)
        namespace = int(cache.data_ptr())
        groups = hq // hk
        out = torch.zeros_like(query)
        for ri, seq_len in enumerate(seq_lengths):
            qs, qe = query_starts[ri:ri + 2]
            nq = qe - qs
            if nq < 0 or seq_len < nq:
                raise ValueError('Query count exceeds valid KV length')
            needed = (seq_len + bs - 1) // bs
            if len(block_table[ri]) < needed:
                raise ValueError('Insufficient physical KV block IDs')
            for qi in range(nq):
                absolute_q = seq_len - nq + qi
                earliest_k = 0 if sliding_window is None else max(0, absolute_q - sliding_window + 1)
                first_block = earliest_k // bs
                last_block = absolute_q // bs
                for qh in range(hq):
                    kvh = qh // groups
                    partitions: list[tuple[int, PartialSoftmax]] = []
                    for logical in range(first_block, last_block + 1):
                        physical = block_table[ri][logical]
                        if not 0 <= physical < nb:
                            raise IndexError('Invalid physical KV block ID')
                        start = max(0, earliest_k - logical * bs)
                        stop = min(bs, absolute_q - logical * bs + 1)
                        if stop <= start:
                            continue
                        address = addresses.get_or_create(namespace, physical, kvh, logical)
                        k = cache[physical, kvh, start:stop, :d]
                        v = cache[physical, kvh, start:stop, d:]
                        partial = self.device.partial_attention(query[qs + qi, qh], k, v, scale=scale, logits_soft_cap=logits_soft_cap, address=address, token_offset=start)
                        partitions.append((address.pim_domain, partial))
                        if self.keep_trace:
                            self.trace.append(PIMCommand(ri, absolute_q, logical, physical, kvh, address.pim_domain, address.k_address, address.v_address, start, stop - start))
                    if not partitions:
                        raise ValueError('No reachable KV tokens for attention')
                    if self.policy.tau == 'domain_grouped':
                        partitions.sort(key=lambda pair: pair[0])
                    merged = None
                    for _, partial in partitions:
                        merged = combine(merged, partial)
                    assert merged is not None
                    out[qs + qi, qh] = (merged.numerator / merged.denominator).to(out.dtype)
        return out

    def trace_as_jsonl(self) -> str:
        return ''.join((json.dumps(asdict(item), sort_keys=True) + '\n' for item in self.trace))

    def physical_address_snapshot(self) -> list[PIMBlockAddress]:
        return [addr for table in self._address_tables.values() for addr in table.snapshot()]
