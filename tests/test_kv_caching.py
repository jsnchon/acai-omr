import torch
import torch.nn.functional as F
from torch import nn
from acai_omr.models.kv_caching import KVCache, CachedMultiheadAttention, CachedTransformerDecoderLayer, CachedTransformerDecoder, MemoryCache
from acai_omr.models.models import OMRDecoder
import pytest

LMX_VOCAB_PATH = "lmx_vocab.txt"

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
        "dropout": 0.1,
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
        attention_output = cached_mha.cached_forward(q_t, K_t, V_t)
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
        "dropout": 0.1,
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

    cached_cross_attn_out = cached_layer.multihead_attn.cached_forward(q_t_cross, K_cross, V_cross)
    uncached_cross_attn_out = uncached_layer.multihead_attn(input_token, latent, latent, need_weights=False)
    assert torch.allclose(cached_cross_attn_out, uncached_cross_attn_out[0])

    # test with latent masking
    latent_mask = torch.tensor([False, True]).unsqueeze(0).repeat(batch_size, 1)

    cached_cross_attn_out = cached_layer.multihead_attn.cached_forward(q_t_cross, K_cross, V_cross, memory_key_padding_mask=latent_mask)
    uncached_cross_attn_out = uncached_layer.multihead_attn(input_token, latent, latent, attn_mask=latent_mask, need_weights=False)
    assert torch.allclose(cached_cross_attn_out, uncached_cross_attn_out[0])

def test_cached_decoder_layer():
    hidden_dim = 12
    num_heads = 2
    decoder_layer_kwargs = {
        "d_model": hidden_dim,
        "nhead": num_heads,
        "dim_feedforward": 48,
        "dropout": 0.1,
        "activation": "gelu",
        "batch_first": True
    }
    head_dim = int(hidden_dim / num_heads)
    uncached_layer = nn.TransformerDecoderLayer(**decoder_layer_kwargs)
    cached_layer = CachedTransformerDecoderLayer(**decoder_layer_kwargs)
    cached_layer.load_state_dict(uncached_layer.state_dict())
    uncached_layer.eval()
    cached_layer.eval()

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

        x = cached_layer.cached_forward(input_token, self_attn_kv_cache, cached_kv_mem)
        cached_out[:, t, :] = x.squeeze(1)

    causal_mask = torch.triu(torch.ones(time_steps, time_steps), diagonal=1).bool() 
    uncached_out = uncached_layer(full_sequence, memory=latent, tgt_mask=causal_mask)
    assert torch.allclose(cached_out, uncached_out, atol=1e-6, rtol=1e-5)

    # test with latent masking
    latent_mask = torch.full([batch_size, latent_len], fill_value=False)
    latent_mask[0, 6:] = True
    latent_mask[1, 3:] = True

    # have to initialize fresh KV caches between batches
    self_attn_kv_cache = KVCache(batch_size, max_seq_len, num_heads, head_dim, dtype=torch.float) 

    time_steps = 3
    full_sequence = torch.empty([batch_size, time_steps, hidden_dim])
    cached_out = torch.empty([batch_size, time_steps, hidden_dim])
    for t in range(time_steps):
        input_token = torch.rand([batch_size, 1, hidden_dim])
        full_sequence[:, t, :] = input_token.squeeze(1)

        x = cached_layer.cached_forward(input_token, self_attn_kv_cache, cached_kv_mem, memory_key_padding_mask=latent_mask)
        cached_out[:, t, :] = x.squeeze(1)

    causal_mask = torch.triu(torch.ones(time_steps, time_steps), diagonal=1).bool() 
    uncached_out = uncached_layer(full_sequence, memory=latent, tgt_mask=causal_mask, memory_key_padding_mask=latent_mask)
    assert torch.allclose(cached_out, uncached_out, atol=1e-6, rtol=1e-5)

def test_memory_cache():
    hidden_dim = 12
    num_heads = 2
    decoder_layer_kwargs = {
        "d_model": hidden_dim,
        "nhead": num_heads,
        "dim_feedforward": 48,
        "dropout": 0.1,
        "activation": "gelu",
        "batch_first": True
    }
    head_dim = int(hidden_dim / num_heads)
    layer = CachedTransformerDecoderLayer(**decoder_layer_kwargs)
    layer.eval()

    batch_size = 2

    latent_len = 8
    latents = [torch.rand([batch_size, latent_len, hidden_dim]), 
               torch.rand([batch_size, latent_len, hidden_dim])]
    
    # test with multiple latents to make sure caches are resetting
    for latent in latents:
        W_kv = layer.multihead_attn.in_proj_weight[hidden_dim:, :]
        b_kv = layer.multihead_attn.in_proj_bias[hidden_dim:]
        KV_cross = F.linear(latent, W_kv, b_kv)
        expected_K_cross, expected_V_cross = KV_cross.chunk(2, dim=-1)
        expected_K_cross = expected_K_cross.view(batch_size, latent_len, num_heads, head_dim).transpose(1, 2)
        expected_V_cross = expected_V_cross.view(batch_size, latent_len, num_heads, head_dim).transpose(1, 2)

        memory_cache = MemoryCache()
        memory_cache.cache_memory_keys_and_vals(latent, layer)
        K_cross, V_cross = memory_cache.get_cached_keys_and_vals()

        assert torch.allclose(K_cross, expected_K_cross)
        assert torch.allclose(V_cross, expected_V_cross)

def test_cached_transformer_decoder():
    hidden_dim = 12
    num_heads = 2
    decoder_layer_kwargs = {
        "d_model": hidden_dim,
        "nhead": num_heads,
        "dim_feedforward": 48,
        "dropout": 0.1,
        "activation": "gelu",
        "batch_first": True
    }
    uncached_layer = nn.TransformerDecoderLayer(**decoder_layer_kwargs)
    cached_layer = CachedTransformerDecoderLayer(**decoder_layer_kwargs)

    decoder_kwargs = {
        "num_layers": 3,
        "norm": nn.LayerNorm(hidden_dim, eps=1e-6)
    }

    batch_size = 4
    max_decoder_seq_len = 200

    uncached_decoder = nn.TransformerDecoder(uncached_layer, **decoder_kwargs)
    cached_decoder = CachedTransformerDecoder(cached_layer, max_batch_size=batch_size, max_decoder_seq_len=max_decoder_seq_len, cache_dtype=torch.float, **decoder_kwargs)
    cached_decoder.load_state_dict(uncached_decoder.state_dict())
    uncached_decoder.eval()
    cached_decoder.eval()

    # basic test: one batch, no masking
    latent_len = 8
    latent = torch.rand([batch_size, latent_len, hidden_dim])
    cached_decoder.prepare_caches(latent)

    time_steps = 3
    full_sequence = torch.empty([batch_size, time_steps, hidden_dim])
    cached_out = torch.empty([batch_size, time_steps, hidden_dim])
    for t in range(time_steps):
        input_token = torch.rand([batch_size, 1, hidden_dim])
        full_sequence[:, t, :] = input_token.squeeze(1)

        x = cached_decoder.cached_generate(input_token)
        cached_out[:, t, :] = x.squeeze(1)

    causal_mask = torch.triu(torch.ones(time_steps, time_steps), diagonal=1).bool() 
    uncached_out = uncached_decoder(full_sequence, memory=latent, tgt_mask=causal_mask)
    assert torch.allclose(cached_out, uncached_out, atol=1e-6, rtol=1e-5)

    # test with masking
    latent_mask = torch.full([batch_size, latent_len], fill_value=False)
    latent_mask[1, 3:] = True
    latent_mask[2, 6:] = True
    latent_mask[3, 6:] = True
    
    time_steps = 3
    cached_decoder.prepare_caches(latent) # reset caches
    full_sequence = torch.empty([batch_size, time_steps, hidden_dim])
    cached_out = torch.empty([batch_size, time_steps, hidden_dim])
    for t in range(time_steps):
        input_token = torch.rand([batch_size, 1, hidden_dim])
        full_sequence[:, t, :] = input_token.squeeze(1)

        x = cached_decoder.cached_generate(input_token, memory_key_padding_mask=latent_mask)
        cached_out[:, t, :] = x.squeeze(1)

    causal_mask = torch.triu(torch.ones(time_steps, time_steps), diagonal=1).bool() 
    uncached_out = uncached_decoder(full_sequence, memory=latent, tgt_mask=causal_mask, memory_key_padding_mask=latent_mask)
    assert torch.allclose(cached_out, uncached_out, atol=1e-6, rtol=1e-5)

    # test with less examples than max_batch_size
    batch_size = 3

    latent = torch.rand([batch_size, latent_len, hidden_dim])
    cached_decoder.prepare_caches(latent)
    latent_mask = torch.full([batch_size, latent_len], fill_value=False)
    latent_mask[0, 3:] = True
    latent_mask[2, 6:] = True
    
    time_steps = 3
    full_sequence = torch.empty([batch_size, time_steps, hidden_dim])
    cached_out = torch.empty([batch_size, time_steps, hidden_dim])
    for t in range(time_steps):
        input_token = torch.rand([batch_size, 1, hidden_dim])
        full_sequence[:, t, :] = input_token.squeeze(1)

        x = cached_decoder.cached_generate(input_token, memory_key_padding_mask=latent_mask)
        cached_out[:, t, :] = x.squeeze(1)

    causal_mask = torch.triu(torch.ones(time_steps, time_steps), diagonal=1).bool() 
    uncached_out = uncached_decoder(full_sequence, memory=latent, tgt_mask=causal_mask, memory_key_padding_mask=latent_mask)
    assert torch.allclose(cached_out, uncached_out, atol=1e-6, rtol=1e-5)

    # test uncached path works fine, eg for teacher forced decoding: cached_decoder's forward with the same args as 
    # uncached_out should be equal (they should be calling the same function)
    cached_out_tf = cached_decoder(full_sequence, memory=latent, tgt_mask=causal_mask, memory_key_padding_mask=latent_mask)
    assert torch.allclose(cached_out_tf, uncached_out, atol=1e-6, rtol=1e-5)

def test_omr_decoder_with_caching():
    max_lmx_seq_len = 15
    hidden_dim = 24
    batch_size = 8
    omr_decoder_kwargs = {
        "max_lmx_seq_len": max_lmx_seq_len,
        "lmx_vocab_path": LMX_VOCAB_PATH,
        "num_layers": 5,
        "hidden_dim": hidden_dim,
        "num_heads": 4,
        "mlp_dim": 48
    }
    uncached_decoder = OMRDecoder(**omr_decoder_kwargs)
    cached_decoder = OMRDecoder(**omr_decoder_kwargs, use_caching=True, max_batch_size=batch_size, cache_dtype=torch.float)
    uncached_decoder.eval()
    cached_decoder.eval()
    cached_decoder.load_state_dict(uncached_decoder.state_dict())
    assert torch.equal(uncached_decoder.decoder_blocks.layers[0].self_attn.in_proj_weight,
                       cached_decoder.decoder_blocks.layers[0].self_attn.in_proj_weight)

    latent_len = 20

    # complete basic capabilities test: multiple batches with masking
    latent = torch.rand([batch_size, latent_len, hidden_dim])
    cached_decoder.prepare_caches(latent)
    latent_mask = torch.full([batch_size, latent_len], fill_value=False)
    latent_mask[0, 16:] = True
    latent_mask[2, 13:] = True
    latent_mask[3, 14:] = True
    
    time_steps = 10
    full_sequence = torch.empty([batch_size, time_steps], dtype=torch.int)
    cached_out = torch.empty([batch_size, time_steps, cached_decoder.vocab_size])
    cached_decoder.prepare_caches(latent)
    for t in range(time_steps):
        input_token = torch.randint(0, cached_decoder.vocab_size, [batch_size, 1])
        full_sequence[:, t] = input_token.squeeze(1)

        x = cached_decoder.cached_generate(input_token, t, latent_attention_mask=latent_mask)
        cached_out[:, t, :] = x.squeeze(1)

    uncached_out = uncached_decoder.generate(full_sequence, latent, latent_attention_mask=latent_mask)
    assert torch.allclose(cached_out, uncached_out, atol=1e-6, rtol=1e-5)

    # test uncached path works fine
    cached_out_tf = cached_decoder.generate(full_sequence, latent, latent_attention_mask=latent_mask)
    assert(cached_out_tf.requires_grad)
    cached_out_tf = cached_decoder.forward(full_sequence, latent, None, latent_attention_mask=latent_mask)
    assert(cached_out_tf.requires_grad)
    assert torch.allclose(cached_out_tf, uncached_out, atol=1e-6, rtol=1e-5)

    # test trying to go past max decoder limit
    cached_decoder.prepare_caches(latent)
    with pytest.raises(RuntimeError):
        time_steps = 16 
        for t in range(time_steps):
            input_token = torch.randint(0, cached_decoder.vocab_size, [batch_size, 1])
            x = cached_decoder.cached_generate(input_token, t, latent_attention_mask=latent_mask)

if __name__ == "__main__":
    # test_cached_transformer_decoder()
    test_omr_decoder_with_caching()
