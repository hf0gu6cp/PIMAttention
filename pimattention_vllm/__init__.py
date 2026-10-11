def register():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend(AttentionBackendEnum.CUSTOM, 'pimattention_vllm.vllm_backend.PIMAttentionBackend')
