import torch
from torch import nn
from models import Encoder, MAEEncoder

def test_encoder_batchify():
    encoder = Encoder(patch_size=2)
    x = [torch.rand(3, 4, 4), torch.rand(3, 4, 8)]
    batch, mask = encoder.batchify(x)
    print(batch)
    print(mask)
    # assert torch.equal(batch.offsets(), torch.tensor([0, 4, 12]))
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
    encoder = MAEEncoder(0.50)
    seq_len = 4
    x = torch.arange(seq_len).unsqueeze(0).repeat(12, 1).unsqueeze(0)
    print(f"x before shuffle/mask: {x}, shape: {x.shape}")
    t_masked, len_keep, seq_mask, ids_restore = encoder.mask_sequence(x)
    print(f"Output:\nt_masked: {t_masked}, shape: {t_masked.shape}\nlen_keep: {len_keep}\nseq_mask: {seq_mask}\nids_restore: {ids_restore}")
    assert t_masked.shape == torch.Size([1, 12, 2])
    assert len_keep == 2
    t_masked = torch.concat((t_masked, (torch.zeros(1, 12, 3) - 1)), dim=-1)
    undo = t_masked.index_select(dim=-1, index=ids_restore.squeeze(0)) 
    print(f"After appending \"mask tokens\" and unshuffling: {undo}")
    assert undo.shape == x.shape

if __name__ == "__main__":
    test_mask_sequence()