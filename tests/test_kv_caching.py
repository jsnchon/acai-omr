import torch
import torch.nn.functional as F
from torch import nn
from acai_omr.models.kv_caching import KVCache, CachedMultiheadAttention, CachedTransformerDecoderLayer

# this test simulates a whole sequence being fed through cached mha and pytorch's native mha. We generate one token of
# the "original" sequence at a time, prepare it for cached mha, and feed it through. The tokens are stored in a tensor representing
# the original sequence which is then fed into torch's MHA to be turned into queries, keys, and values then used in attention
def test_cached_mha():
    head_dim = 6
    num_heads = 2
    embed_dim = head_dim * num_heads
    mha_kwargs = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "dropout": 0,
        "batch_first": True
    }
    uncached_mha = nn.MultiheadAttention(**mha_kwargs)
    cached_mha = CachedMultiheadAttention(**mha_kwargs)
    cached_mha.load_state_dict(uncached_mha.state_dict())
    uncached_mha.eval()
    cached_mha.eval()

    batch_size = 2
    max_seq_len = 3
    self_attn_kv_cache = KVCache(batch_size, max_seq_len, num_heads, head_dim, dtype=torch.float)

    time_steps = 3
    full_sequence = torch.empty([batch_size, time_steps, embed_dim])

    cached_out = torch.empty([batch_size, time_steps, embed_dim])
    for t in range(time_steps):
        # create an original input token, store it for uncached MHA, then prepare for cached MHA
        input_token = torch.rand([batch_size, 1, embed_dim])
        full_sequence[:, t, :] = input_token.squeeze(1)
        qkv_t = F.linear(input_token, cached_mha.in_proj_weight, cached_mha.in_proj_bias)
        q_t, k_t, v_t = qkv_t.chunk(3, dim=-1)

        q_t = q_t.view(batch_size, num_heads, 1, head_dim)
        k_t = k_t.view(batch_size, num_heads, 1, head_dim)
        v_t = v_t.view(batch_size, num_heads, 1, head_dim)

        K_t, V_t = self_attn_kv_cache.update(k_t, v_t)
        attention_output = cached_mha(q_t, K_t, V_t)
        cached_out[:, t, :] = attention_output.squeeze(1)

    # compare to MHA's outputs (with causal masking) at each time step over the whole input sequence. In
    # self-attention, torch's MHA expects the original input sequence to be passed in as q, k, and v, the 
    # embeddings to derive queries, keys, and values from
    causal_mask = torch.triu(torch.ones(time_steps, time_steps), diagonal=1).bool() 
    uncached_out = uncached_mha(full_sequence, full_sequence, full_sequence, attn_mask=causal_mask)
    assert torch.allclose(cached_out, uncached_out[0])

def test_cached_cross_attn():
    hidden_dim = 12
    num_heads = 2
    decoder_layer_kwargs = {
        "d_model": hidden_dim,
        "nhead": num_heads,
        "dim_feedforward": 48,
        "dropout": 0.0,
        "activation": "gelu",
        "batch_first": True
    }
    head_dim = int(hidden_dim / num_heads)
    uncached_layer = nn.TransformerDecoderLayer(**decoder_layer_kwargs)
    cached_layer = CachedTransformerDecoderLayer(**decoder_layer_kwargs)
    cached_layer.load_state_dict(uncached_layer.state_dict())
    uncached_layer.eval()
    cached_layer.eval()

    batch_size = 1
    input_seq_len = 1
    latent_len = 2

    # simple test for just the cross attention 
    input_token = torch.rand([batch_size, input_seq_len, hidden_dim])
    latent = torch.rand([batch_size, latent_len, hidden_dim])

    # get the query for this step's token
    W_q_cross = cached_layer.multihead_attn.in_proj_weight[:hidden_dim, :]
    b_q_cross = cached_layer.multihead_attn.in_proj_bias[:hidden_dim]

    q_t_cross = F.linear(input_token, W_q_cross, b_q_cross)
    q_t_cross = q_t_cross.view(batch_size, num_heads, 1, head_dim)
 
    W_kv = cached_layer.multihead_attn.in_proj_weight[hidden_dim:, :]
    b_kv = cached_layer.multihead_attn.in_proj_bias[hidden_dim:]
    KV_cross = F.linear(latent, W_kv, b_kv)
    K_cross, V_cross = KV_cross.chunk(2, dim=-1)
    # split the last (embedding) dimension into heads, then transpose latent_len and num_heads to match sdpa's expected dim order.
    # Can't just do this all in one view() because it's not easy to specify one unambiguous shape configuration that matches the intention
    # of splitting heads/transposing in one step
    K_cross = K_cross.view(batch_size, latent_len, num_heads, head_dim).transpose(1, 2)
    V_cross = V_cross.view(batch_size, latent_len, num_heads, head_dim).transpose(1, 2)

    cached_cross_attn_out = cached_layer.multihead_attn(q_t_cross, K_cross, V_cross)
    uncached_cross_attn_out = uncached_layer.multihead_attn(input_token, latent, latent, need_weights=False)
    assert torch.allclose(cached_cross_attn_out, uncached_cross_attn_out[0])

    # with latent masking

def test_cached_decoder_layer():
    hidden_dim = 12
    num_heads = 2
    decoder_layer_kwargs = {
        "d_model": hidden_dim,
        "nhead": num_heads,
        "dim_feedforward": 48,
        "dropout": 0.0,
        "activation": "gelu",
        "batch_first": True
    }
    head_dim = int(hidden_dim / num_heads)
    uncached_layer = nn.TransformerDecoderLayer(**decoder_layer_kwargs)
    cached_layer = CachedTransformerDecoderLayer(**decoder_layer_kwargs)
    cached_layer.load_state_dict(uncached_layer.state_dict())

    batch_size = 2

    max_seq_len = 200
    self_attn_kv_cache = KVCache(batch_size, max_seq_len, num_heads, head_dim, dtype=torch.float)

    latent_len = 8
    latent = torch.rand([batch_size, latent_len, hidden_dim])
    # cache the keys/values derived from the "encoder latent"
    
    W_kv = cached_layer.multihead_attn.in_proj_weight[hidden_dim:, :]
    b_kv = cached_layer.multihead_attn.in_proj_bias[hidden_dim:]
    KV_cross = F.linear(latent, W_kv, b_kv)
    K_cross, V_cross = KV_cross.chunk(2, dim=-1)
    K_cross = K_cross.view(batch_size, latent_len, num_heads, head_dim).transpose(1, 2)
    V_cross = V_cross.view(batch_size, latent_len, num_heads, head_dim).transpose(1, 2)

    cached_kv_mem = (K_cross, V_cross)

    time_steps = 3
    # basic inference test with no masking
    full_sequence = torch.empty([batch_size, time_steps, hidden_dim])
    cached_out = torch.empty([batch_size, time_steps, hidden_dim])
    for t in range(time_steps):
        input_token = torch.rand([batch_size, 1, hidden_dim])
        full_sequence[:, t, :] = input_token.squeeze(1)

        x = cached_layer(input_token, self_attn_kv_cache, cached_kv_mem)
        cached_out[:, t, :] = x.squeeze(1)

    causal_mask = torch.triu(torch.ones(time_steps, time_steps), diagonal=1).bool() 
    uncached_out = uncached_layer(full_sequence, memory=latent, tgt_mask=causal_mask)
    print(cached_out, uncached_out)

# final test, deterministic uncached/cached decoders (ie no dropout), run on same input for same amount of steps
# and should get same logits at each step

if __name__ == "__main__":
    test_cached_cross_attn()
