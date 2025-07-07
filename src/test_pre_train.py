import torch
from torch import nn
from models import Encoder, MAEEncoder

def test_encoder_batchify():
    encoder = Encoder(patch_size=2)
    x = [torch.rand(3, 4, 4), torch.rand(3, 4, 8)]
    batch, mask = encoder.batchify(x)
    print(batch)
    print(mask)
    first_ex_mask = (torch.arange(8) >= 4).unsqueeze(0)
    second_ex_mask = (torch.arange(8) >= 8).unsqueeze(0)
    assert torch.equal(mask, torch.cat((first_ex_mask, second_ex_mask)))

def test_encoder_forward():
    hidden_dim = 200
    encoder = Encoder(num_layers=2, num_heads=2, hidden_dim=hidden_dim, mlp_dim=500, patch_size=2)
    x = [torch.rand(3, 4, 4), torch.rand(3, 4, 8)]
    x, mask = encoder(x)
    print(x)
    print(mask)
    assert x.shape == torch.Size([2, 8, hidden_dim])

def test_mask_sequence():
    encoder = MAEEncoder(0.50, num_heads=1, hidden_dim=1)
    SEQ_LEN = 4
    x = torch.arange(SEQ_LEN).unsqueeze(0).repeat(12, 1).unsqueeze(0) # simulate patch size 2 (so CP^2 = 12) w/ sequence of length 4, label patches to check shuffle/unshuffle
    print(f"x before shuffle/mask: {x}, shape: {x.shape}")
    pe_num_grid = torch.arange(4, dtype=torch.float).reshape(2, 2) # pes labeled in order to check alignment with shuffle
    pe_filler = torch.zeros(2, 4) - 1 # should not appear in final slice to be added to embeddings
    encoder.pos_embedding = nn.Parameter(
        torch.cat((pe_num_grid, pe_filler), dim=1).unsqueeze(-1)
    )
    print(f"Unsliced positional embedding grid: {encoder.pos_embedding}")
    t_masked, pos_embed_slice, len_keep, seq_mask, ids_restore = encoder.mask_sequence(x, 2, 2)
    print(f"Output:\nt_masked: {t_masked}, shape: {t_masked.shape}\npos_embed_slice: {pos_embed_slice}, shape: {pos_embed_slice.shape}\nlen_keep: {len_keep}\nseq_mask: {seq_mask}\nids_restore: {ids_restore}")
    assert t_masked.shape == torch.Size([1, 12, 2])
    assert len_keep == 2
    t_masked = torch.concat((t_masked, (torch.zeros(1, 12, 3) - 1)), dim=-1) # append mask tokens of -1 tensors
    undo = t_masked.index_select(dim=-1, index=ids_restore.squeeze(0)) 
    print(f"After appending mask tokens and unshuffling: {undo}")
    assert undo.shape == x.shape

def test_masked_encoder_batchify():
    encoder = MAEEncoder(0.50, patch_size=2)
    x = [torch.rand(3, 4, 4), torch.rand(3, 4, 6)]
    batch, attn_mask, seq_masks, ids_restores = encoder.batchify(x)
    print(f"Output:\nEmbeddings {batch}\nAttention mask: {attn_mask}\nSequence masks: {seq_masks}\nRestore tensor: {ids_restores}")
    first_ex_mask = (torch.arange(3) >= 2).unsqueeze(0)
    second_ex_mask = (torch.arange(3) >= 3).unsqueeze(0)
    assert torch.equal(attn_mask, torch.cat((first_ex_mask, second_ex_mask)))

def test_masked_encoder_forward():
    hidden_dim = 200
    encoder = MAEEncoder(0.50, num_layers=2, num_heads=2, hidden_dim=hidden_dim, mlp_dim=500, patch_size=2)
    x = [torch.rand(3, 4, 4), torch.rand(3, 4, 8)]
    x, attn_mask, seq_masks, ids_restores = encoder(x)
    print(f"Output:\nEmbeddings {x}\nAttention mask: {attn_mask}\nSequence masks: {seq_masks}\nRestore tensor: {ids_restores}")
    assert x.shape == torch.Size([2, 4, hidden_dim])

if __name__ == "__main__":
    test_masked_encoder_forward()