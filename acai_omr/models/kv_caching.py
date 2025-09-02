import torch
import torch.nn.functional as F
from torch import nn

class KVCache(nn.Module):
    """
    NOTE: this is just torchtune's implementation (https://docs.pytorch.org/torchtune/stable/_modules/torchtune/modules/kv_cache.html#KVCache)
    with some small modifications to make it more useful for my case: 
        update() returns only filled in parts of the cache each call
        The cache is initialized with a max batch size, but smaller batches can be passed into it (eg like the last batch of
        the dataloader which is usually smaller) and only the right parts of the cache will be updated/indexed. Things will break
        if you use update() with changing batch sizes within the same batch (eg if you drop completed sequences)

    Standalone ``nn.Module`` containing a kv-cache to cache past key and values during inference.

    Args:
        batch_size (int): batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        num_kv_heads (int): number of key/value heads.
        head_dim (int): per-attention head embedding dimension
        dtype (torch.dtype): dtype for the caches. This manually has to be set to a lower precision type
        if using autocast
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        cache_shape = (max_batch_size, num_kv_heads, max_seq_len, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "cache_pos", torch.arange(0, cache_shape[2]), persistent=False
        )
        self.max_batch_size = max_batch_size

    def reset(self) -> None:
        """Reset the cache to zero."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos -= self.size

    @property
    def size(self) -> int:
        return self.cache_pos[0].item()

    def update(
        self, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache with the new ``k_val``, ``v_val`` and return the updated cache.

        Note:
            When updating the KV cache, it is assumed that subsequent updates should update key-value
            positions in consecutive sequence positions. If you wish to update cache values which have
            already been filled, use ``.reset()``, which will reset the cache to the zero-th position.

        Args:
            k_val (torch.Tensor): Current key tensor with shape [B, H, S, D]
            v_val (torch.Tensor): Current value tensor with shape [B, H, S, D]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated key and value cache tensors, respectively.

        Raises:
            ValueError: if the batch size of the new key (or value) tensor is greater than the batch size
                used during cache setup.

        Note:
            This function will raise an ``AssertionError`` if the sequence length of ``k_val``
                is longer than the maximum cache sequence length.

        """
        cur_bsz, _, seq_len, _ = k_val.shape
        if cur_bsz > self.k_cache.shape[0]:
            raise ValueError(
                f"The current cache has been setup with a max batch size of {self.k_cache.shape[0]}"
                f", but found new key tensors with batch size {k_val.shape[0]}!"
            )

        assert (self.cache_pos[0] + seq_len) <= self.k_cache.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache

        k_out[:cur_bsz, :, self.cache_pos[:seq_len], :] = k_val
        v_out[:cur_bsz, :, self.cache_pos[:seq_len], :] = v_val

        # forward cache_pos seq_len positions along
        # cache_pos starts at (0, 1, 2, 3, 4, 5, ...)
        # an update of seq_len = 5 tokens brings it to
        # (5, 6, 7, 8, 9, ...)
        # this allows us to track the current position in the cache
        # after the last update in a compile-friendly way without any dynamism
        # e.g. relying on an int size tracker, or re-creating cache_pos every time
        self.cache_pos.add_(seq_len)

        k_out = k_out[:cur_bsz, :, :self.cache_pos[0]]
        v_out = v_out[:cur_bsz, :, :self.cache_pos[0]]

        return k_out, v_out

# the regular forwards of all these cached subclasses are untouched so that we have an uncached pathway that can be used 
# for things like generating teacher forced probs over a whole input sequence (which GRPO needs)

class CachedMultiheadAttention(nn.MultiheadAttention):
    """
    Inputs:
        q_t: (B, H, 1, E_h), this step's query vector split across heads
        K_t: (B, H, T_t, E_h), this step's key matrix, comprising all keys from past steps + this step's key vector
        V_t: (B, H, T_t, E_h), this step's value matrix, comprising all values from past steps + this step's value vector
        memory_key_padding_mask: (B, T_latent) if not None, the cross attention mask for the encoder memory. True = 
        mask this element from participating in attention
    """
    def cached_forward(self, q_t, K_t, V_t, memory_key_padding_mask=None):
        # to prepare the memory mask for sdpa, we have to prepare it for broadcasting across heads and rows and also invert it since
        # the functional sdpa uses False = mask this element from participating in attention (instead of True)
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.unsqueeze(1).unsqueeze(1) # (B, 1, 1, S)
            memory_key_padding_mask = ~memory_key_padding_mask

        # functional sdpa also requires manually setting dropout to 0 if in eval mode 
        attention_output = F.scaled_dot_product_attention(
            q_t, K_t, V_t, attn_mask=memory_key_padding_mask, dropout_p=self.dropout if self.training else 0.0,
        )

        # concatenate head outputs
        attention_output = attention_output.transpose(1, 2).flatten(start_dim=-2, end_dim=-1)

        # final out projection
        attention_output = self.out_proj(attention_output) 
        return attention_output

class CachedTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation = F.gelu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias, device, dtype)

        self.hidden_dim = d_model
        self.num_heads = nhead 
        self.self_attn = CachedMultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.head_dim = self.self_attn.head_dim
        self.multihead_attn = CachedMultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    """
    We assume this cached pathway is being used for batch autoregressive inference, so no tgt padding (can generate junk for sequences that
    finished earlier), no causal masking (just interested in next token prediction), and likely memory masking
    Inputs
        tgt_t: (B, 1, E), embedding for this step's token from which to predict the next token
        self_attn_kv_cache: KVCache instance storing cached keys and values from previous steps and to cache new keys and values into
        cached_kv_mem: tuple of (cached_k_m, cached_v_m), the cached keys and values, respectively, from the 
        memory_key_padding_mask: (B, T_latent) if not None, the mask marking which positions in the latent tensor are padding. True = mask this
        position from attention
    """
    def cached_forward(self, tgt_t, self_attn_kv_cache: KVCache, cached_kv_mem: tuple[torch.Tensor, torch.Tensor], memory_key_padding_mask=None):
        # SELF ATTENTION
        # project this step's embedding into concatenated qkv vector, then split into q, k, v vectors
        qkv_t = F.linear(tgt_t, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q_t, k_t, v_t = qkv_t.chunk(3, dim=-1)

        # split across heads, add k and v to kv cache as (B, H, 1, D_h) tensors
        batch_size = tgt_t.shape[0]
        num_heads = self.num_heads
        head_dim = self.head_dim
        q_t = q_t.view(batch_size, num_heads, 1, head_dim)
        k_t = k_t.view(batch_size, num_heads, 1, head_dim)
        v_t = v_t.view(batch_size, num_heads, 1, head_dim)

        K_t, V_t = self_attn_kv_cache.update(k_t, v_t)

        self_attn_output = self.self_attn.cached_forward(q_t, K_t, V_t)

        tgt_t = self.norm1(tgt_t + self_attn_output)

        # CROSS ATTENTION
        # chunk/split return multiple views that can't be used as parameters in F.linear, so slice tensors instead 
        W_q_cross = self.multihead_attn.in_proj_weight[:self.hidden_dim, :]
        b_q_cross = self.multihead_attn.in_proj_bias[:self.hidden_dim]

        q_t_cross = F.linear(tgt_t, W_q_cross, b_q_cross)
        q_t_cross = q_t_cross.view(batch_size, num_heads, 1, head_dim)
        K_cross, V_cross = cached_kv_mem

        cross_attn_output = self.multihead_attn.cached_forward(q_t_cross, K_cross, V_cross, memory_key_padding_mask=memory_key_padding_mask)
        tgt_t = self.norm2(tgt_t + cross_attn_output)

        tgt_t = self.norm3(tgt_t + self._ff_block(tgt_t))
        return tgt_t

# cache for encoder memory for cross attention. Each instance is assigned to a layer and will use that layer's 
# in_proj_weight/bias to store a K_cross and V_cross for a given encoder memory tensor
class MemoryCache(nn.Module):
    def __init__(self, layer: CachedTransformerDecoderLayer):
        super().__init__()
        # make shallow "views" that stay tied to weights (so when old_policy is synced, these are also synced)
        self.register_buffer("W_kv", layer.multihead_attn.in_proj_weight[layer.hidden_dim:, :], persistent=False)
        self.register_buffer("b_kv", layer.multihead_attn.in_proj_bias[layer.hidden_dim:], persistent=False)
        self.num_heads = layer.num_heads
        self.head_dim = layer.head_dim
        self.register_buffer("K_cross", None, persistent=False)
        self.register_buffer("b_kv", None, persistent=False)

    # called per batch with a (B, T_latent, E) memory tensor to reset/update the cached K_cross, V_cross. Stores
    # K_cross and V_cross as (B, H, T_latent, E) tensors
    def cache_memory_keys_and_vals(self, memory: torch.Tensor):
        batch_size = memory.shape[0]
        memory_len = memory.shape[1]

        KV_cross = F.linear(memory, self.W_kv, self.b_kv)
        K_cross, V_cross = KV_cross.chunk(2, dim=-1)
        # split the last (embedding) dimension into heads, then transpose latent_len and num_heads to match sdpa's expected dim order.
        # Can't just do this all in one view() because it's not easy to specify one unambiguous shape configuration that matches the intention
        # of splitting heads/transposing in one step
        K_cross = K_cross.view(batch_size, memory_len, self.num_heads, self.head_dim).transpose(1, 2)
        V_cross = V_cross.view(batch_size, memory_len, self.num_heads, self.head_dim).transpose(1, 2)

        self.K_cross = K_cross
        self.V_cross = V_cross

    def get_cached_keys_and_vals(self):
        return self.K_cross, self.V_cross

class CachedTransformerDecoder(nn.TransformerDecoder):
    """
    Notes:
    max_batch_size and max_decoder_seq_len should be the largest values that will ever be seen throughout all inference. This way,
    we avoid having to continually reinitialize caches.
    The inference loop for this module is for each batch:
        1. Call prepare_caches() to reset caches while passing the encoder memory in to cache it
        2. Call cached_generate() with one token at a time, starting with <bos>
    """
    def __init__(
            self, 
            decoder_layer: CachedTransformerDecoderLayer, 
            num_layers: int, 
            max_batch_size: int, 
            max_decoder_seq_len: int,
            cache_dtype, 
            norm: nn.Module = None):
        assert isinstance(decoder_layer, CachedTransformerDecoderLayer), "Can't use uncached TransformerDecoderLayer in a cached TransformerDecoder"
        super().__init__(decoder_layer, num_layers, norm)
       
        # initialize caches for each layer 
        self.self_attn_caches = nn.ModuleList([KVCache(max_batch_size, max_decoder_seq_len, layer.num_heads, layer.head_dim, dtype=cache_dtype) for layer in self.layers])
        self.cross_attn_caches = nn.ModuleList([MemoryCache(layer) for layer in self.layers])

    # reset each layer's self and cross attention caches. Fill cross attention caches using this batch's memory
    def prepare_caches(self, encoder_memory):
        for self_attn_cache in self.self_attn_caches:
            self_attn_cache.reset()
        for cross_attn_cache in self.cross_attn_caches:
            cross_attn_cache.cache_memory_keys_and_vals(encoder_memory)

    # carries out one autoregressive inference step, predicting the next token after embedding_t given 
    # the past sequence KVs cached in self.self_attn_caches and encoder memory cached in self.cross_attn_caches. 
    # This function assumes the caches have been reset/prepared properly by the caller
    def cached_generate(self, embedding_t, memory_key_padding_mask=None):
        x = embedding_t
        for i, layer in enumerate(self.layers):
            self_attn_kv_cache = self.self_attn_caches[i]
            cached_kv_mem = self.cross_attn_caches[i].get_cached_keys_and_vals()
            x = layer.cached_forward(x, self_attn_kv_cache, cached_kv_mem, memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
